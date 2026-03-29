use crate::types::Concept;
use anyhow::{anyhow, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

pub struct EmbeddingIndex {
    model: Option<TextEmbedding>,
    vectors: HashMap<u64, Vec<f32>>,
    cache_dir: Option<PathBuf>,
}

impl EmbeddingIndex {
    /// Initialize embedding index with the BGE-small model.
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self> {
        let mut opts = InitOptions::new(EmbeddingModel::BGESmallENV15)
            .with_show_download_progress(true);
        if let Some(ref dir) = cache_dir {
            opts = opts.with_cache_dir(dir.clone());
        }
        let model = TextEmbedding::try_new(opts)?;
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

impl Default for EmbeddingIndex {
    fn default() -> Self {
        Self {
            model: None,
            vectors: HashMap::new(),
            cache_dir: None,
        }
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
        let mut opts = InitOptions::new(EmbeddingModel::BGESmallENV15)
            .with_show_download_progress(true);
        if let Some(ref dir) = self.cache_dir {
            opts = opts.with_cache_dir(dir.clone());
        }
        let model = TextEmbedding::try_new(opts)?;
        self.model = Some(model);
        Ok(())
    }

    /// Embed a concept using its canonical name + subtokens + example identifiers.
    pub fn embed_concept(&mut self, concept: &Concept) -> Result<Vec<f32>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("no embedding model loaded (index is empty)"))?;

        // Build text: canonical + subtokens + up to 5 example identifier names
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
        let text = parts.join(" ");

        let embeddings = model.embed(vec![text], None)?;
        let vector = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("embedding model returned no vectors"))?;

        self.vectors.insert(concept.id, vector.clone());
        Ok(vector)
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
            .embed(vec![text.to_string()], None)
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
}
