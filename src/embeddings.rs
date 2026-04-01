use crate::types::Concept;
use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::api::sync::ApiBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

const MODEL_ID: &str = "BAAI/bge-small-en-v1.5";

struct BgeModel {
    model: BertModel,
    tokenizer: Tokenizer,
}

impl BgeModel {
    fn load(cache_dir: Option<&PathBuf>) -> Result<Self> {
        let device = Device::Cpu;

        let mut builder = ApiBuilder::from_env();
        if let Some(dir) = cache_dir {
            builder = builder.with_cache_dir(dir.clone());
        }
        let api = builder.with_progress(true).build()?;
        let repo = api.model(MODEL_ID.to_string());

        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo.get("model.safetensors")?;

        let config: BertConfig =
            serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 512,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("failed to set truncation: {e}"))?;

        let weights = std::fs::read(&weights_path)?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
        let model = BertModel::load(vb, &config)?;

        Ok(BgeModel { model, tokenizer })
    }

    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let device = &self.model.device;
        let encodings = self
            .tokenizer
            .encode_batch(texts, true)
            .map_err(|e| anyhow!("tokenization failed: {e}"))?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        let token_ids: Vec<u32> = encodings
            .iter()
            .flat_map(|e| e.get_ids().iter().copied())
            .collect();
        let type_ids: Vec<u32> = encodings
            .iter()
            .flat_map(|e| e.get_type_ids().iter().copied())
            .collect();
        let mask: Vec<u32> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().copied())
            .collect();

        let token_ids = Tensor::from_vec(token_ids, (batch_size, seq_len), device)?;
        let type_ids = Tensor::from_vec(type_ids, (batch_size, seq_len), device)?;
        let mask = Tensor::from_vec(mask, (batch_size, seq_len), device)?;

        // Forward pass -> [batch, seq_len, hidden_size]
        let output = self.model.forward(&token_ids, &type_ids, Some(&mask))?;

        // CLS pooling: first token -> [batch, hidden_size]
        let cls = output.narrow(1, 0, 1)?.squeeze(1)?;

        // L2 normalize
        let norms = cls.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = cls.broadcast_div(&norms)?;

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            results.push(normalized.get(i)?.to_vec1::<f32>()?);
        }
        Ok(results)
    }
}

#[derive(Default)]
pub struct EmbeddingIndex {
    model: Option<BgeModel>,
    vectors: HashMap<u64, Vec<f32>>,
    cache_dir: Option<PathBuf>,
}

impl EmbeddingIndex {
    /// Initialize embedding index with the BGE-small model.
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self> {
        let model = BgeModel::load(cache_dir.as_ref())?;
        Ok(Self {
            model: Some(model),
            vectors: HashMap::new(),
            cache_dir,
        })
    }

    /// Create an empty embedding index (no model, for when embeddings are disabled).
    pub fn empty() -> Self {
        Self::default()
    }
}

impl EmbeddingIndex {
    /// Load the embedding model into an index that was deserialized from
    /// cache (which has vectors but no model). Needed for runtime queries
    /// like embed_text.
    pub fn load_model(&mut self) -> Result<()> {
        if self.model.is_some() {
            return Ok(());
        }
        let model = BgeModel::load(self.cache_dir.as_ref())?;
        self.model = Some(model);
        Ok(())
    }

    /// Build the text representation for a concept embedding.
    fn concept_text(concept: &Concept) -> String {
        let mut parts = vec![concept.canonical.clone()];
        for subtoken in &concept.subtokens {
            if !parts.contains(subtoken) {
                parts.push(subtoken.clone());
            }
        }
        let example_ids: Vec<String> = concept
            .occurrences
            .iter()
            .map(|o| o.identifier.clone())
            .take(5)
            .collect();
        for id in &example_ids {
            if !parts.contains(id) {
                parts.push(id.clone());
            }
        }
        parts.join(" ")
    }

    /// Embed a single concept using its canonical name + subtokens + example identifiers.
    #[allow(dead_code)]
    pub fn embed_concept(&mut self, concept: &Concept) -> Result<Vec<f32>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("no embedding model loaded (index is empty)"))?;

        let text = Self::concept_text(concept);
        let embeddings = model.embed(vec![text])?;
        let vector = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("embedding model returned no vectors"))?;

        self.vectors.insert(concept.id, vector.clone());
        Ok(vector)
    }

    /// Embed all concepts in a single batched model call.
    pub fn embed_concepts_batch(&mut self, concepts: &[Concept]) -> Result<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("no embedding model loaded (index is empty)"))?;

        if concepts.is_empty() {
            return Ok(());
        }

        let texts: Vec<String> = concepts
            .iter()
            .map(Self::concept_text)
            .collect();
        let all_embeddings = model.embed(texts)?;

        for (concept, vector) in concepts.iter().zip(all_embeddings) {
            self.vectors.insert(concept.id, vector);
        }
        Ok(())
    }

    /// Embed multiple texts in a single batched model call.
    /// Returns a Vec parallel to the input: one embedding per text.
    pub fn embed_texts_batch(&self, texts: &[String]) -> Option<Vec<Vec<f32>>> {
        let model = self.model.as_ref()?;
        model.embed(texts.to_vec()).ok()
    }

    /// Find top-k concepts most similar to the query by cosine similarity.
    pub fn find_similar(&self, query: &[f32], top_k: usize) -> Vec<(u64, f32)> {
        let mut scored: Vec<(u64, f32)> = self
            .vectors
            .iter()
            .map(|(&id, vec)| (id, cosine_similarity(query, vec)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Embed arbitrary text. Returns None if no model is loaded.
    pub fn embed_text(&self, text: &str) -> Option<Vec<f32>> {
        let model = self.model.as_ref()?;
        model
            .embed(vec![text.to_string()])
            .ok()
            .and_then(|vecs| vecs.into_iter().next())
    }

    /// Look up a stored embedding vector by concept ID.
    pub fn get_vector(&self, concept_id: u64) -> Option<&Vec<f32>> {
        self.vectors.get(&concept_id)
    }

    /// Set the model cache directory (used after deserialization from cache).
    pub fn set_cache_dir(&mut self, dir: PathBuf) {
        self.cache_dir = Some(dir);
    }

    /// Return the set of concept IDs that already have embeddings.
    pub fn vector_ids(&self) -> std::collections::HashSet<u64> {
        self.vectors.keys().copied().collect()
    }

    /// Insert a single embedding vector.
    pub fn insert_vector(&mut self, id: u64, vector: Vec<f32>) {
        self.vectors.insert(id, vector);
    }

    /// Number of stored embedding vectors.
    pub fn nb_vectors(&self) -> usize {
        self.vectors.len()
    }
}

impl Serialize for EmbeddingIndex {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.vectors.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EmbeddingIndex {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let vectors = HashMap::deserialize(deserializer)?;
        Ok(Self {
            model: None,
            vectors,
            cache_dir: None,
        })
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_find_similar_empty() {
        let index = EmbeddingIndex::empty();
        let results = index.find_similar(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_similar_ordering() {
        let mut index = EmbeddingIndex::empty();
        index.vectors.insert(1, vec![1.0, 0.0, 0.0]);
        index.vectors.insert(2, vec![0.9, 0.1, 0.0]);
        index.vectors.insert(3, vec![0.0, 1.0, 0.0]);

        let results = index.find_similar(&[1.0, 0.0, 0.0], 3);
        assert_eq!(results[0].0, 1); // most similar
        assert_eq!(results[2].0, 3); // least similar
    }

    #[test]
    fn test_find_similar_top_k_truncation() {
        let mut index = EmbeddingIndex::empty();
        index.vectors.insert(1, vec![1.0, 0.0]);
        index.vectors.insert(2, vec![0.9, 0.1]);
        index.vectors.insert(3, vec![0.5, 0.5]);

        let results = index.find_similar(&[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_get_vector() {
        let mut index = EmbeddingIndex::empty();
        assert!(index.get_vector(1).is_none());

        index.vectors.insert(1, vec![1.0, 2.0, 3.0]);
        let v = index.get_vector(1).unwrap();
        assert_eq!(v, &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_embed_concept_no_model() {
        use crate::types::{Concept, EntityType, Occurrence};
        use std::collections::HashSet;
        use std::path::PathBuf;

        let mut index = EmbeddingIndex::empty();
        let concept = Concept {
            id: 1,
            canonical: "transform".to_string(),
            subtokens: vec!["transform".to_string()],
            occurrences: vec![Occurrence {
                file: PathBuf::from("test.py"),
                line: 1,
                identifier: "spatial_transform".to_string(),
                entity_type: EntityType::Function,
            }],
            entity_types: HashSet::from([EntityType::Function]),
            embedding: None,
            cluster_id: None,
            subconcepts: Vec::new(),
        };

        let result = index.embed_concept(&concept);
        assert!(result.is_err());
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut index = EmbeddingIndex::empty();
        index.vectors.insert(1, vec![1.0, 2.0, 3.0]);
        index.vectors.insert(2, vec![4.0, 5.0, 6.0]);

        let json = serde_json::to_string(&index).unwrap();
        let deserialized: EmbeddingIndex = serde_json::from_str(&json).unwrap();

        assert_eq!(
            deserialized.get_vector(1).unwrap(),
            &vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            deserialized.get_vector(2).unwrap(),
            &vec![4.0, 5.0, 6.0]
        );
        // Model should be None after deserialization
        assert!(deserialized.model.is_none());
    }

    #[test]
    #[ignore] // requires model download
    fn test_embed_text_produces_valid_vector() {
        let index = EmbeddingIndex::new(None).unwrap();
        let vector = index.embed_text("spatial transform").unwrap();
        assert_eq!(vector.len(), 384); // BGE-Small dimension
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01); // BGE-Small outputs are normalized
    }

    #[test]
    #[ignore] // requires model download
    fn test_similar_texts_have_higher_similarity() {
        let index = EmbeddingIndex::new(None).unwrap();
        let a = index.embed_text("spatial transform").unwrap();
        let b = index.embed_text("affine transformation").unwrap();
        let c = index.embed_text("database connection pool").unwrap();

        let sim_ab = cosine_similarity(&a, &b);
        let sim_ac = cosine_similarity(&a, &c);
        assert!(sim_ab > sim_ac); // related terms closer than unrelated
    }

    #[test]
    #[ignore] // requires model download
    fn test_embed_concepts_batch_stores_vectors() {
        use crate::types::{Concept, EntityType, Occurrence};
        use std::collections::HashSet;

        let mut index = EmbeddingIndex::new(None).unwrap();
        let concepts = vec![
            Concept {
                id: 1,
                canonical: "transform".to_string(),
                subtokens: vec!["transform".to_string()],
                occurrences: vec![Occurrence {
                    file: PathBuf::from("test.py"),
                    line: 1,
                    identifier: "spatial_transform".to_string(),
                    entity_type: EntityType::Function,
                }],
                entity_types: HashSet::from([EntityType::Function]),
                embedding: None,
                cluster_id: None,
                subconcepts: Vec::new(),
            },
            Concept {
                id: 2,
                canonical: "connection".to_string(),
                subtokens: vec!["connection".to_string()],
                occurrences: vec![Occurrence {
                    file: PathBuf::from("test.py"),
                    line: 10,
                    identifier: "db_connection".to_string(),
                    entity_type: EntityType::Variable,
                }],
                entity_types: HashSet::from([EntityType::Variable]),
                embedding: None,
                cluster_id: None,
                subconcepts: Vec::new(),
            },
        ];

        index.embed_concepts_batch(&concepts).unwrap();
        assert_eq!(index.nb_vectors(), 2);
        assert_eq!(index.get_vector(1).unwrap().len(), 384);
        assert_eq!(index.get_vector(2).unwrap().len(), 384);
    }
}
