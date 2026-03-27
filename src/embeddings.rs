use crate::types::Concept;
use anyhow::{anyhow, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct EmbeddingIndex {
    model: Option<TextEmbedding>,
    vectors: HashMap<u64, Vec<f32>>,
}

impl EmbeddingIndex {
    /// Initialize embedding index with the BGE-small model.
    pub fn new() -> Result<Self> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_show_download_progress(true),
        )?;
        Ok(Self {
            model: Some(model),
            vectors: HashMap::new(),
        })
    }

    /// Create an empty embedding index (no model, for when embeddings are disabled).
    pub fn empty() -> Self {
        Self {
            model: None,
            vectors: HashMap::new(),
        }
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

    /// Cluster all concepts into groups by similarity threshold.
    ///
    /// Uses union-find for single-linkage clustering: any pair with
    /// cosine similarity > threshold gets merged into the same cluster.
    pub fn cluster(&self, threshold: f32) -> Vec<Vec<u64>> {
        let ids: Vec<u64> = self.vectors.keys().copied().collect();
        if ids.is_empty() {
            return Vec::new();
        }

        // Map concept IDs to contiguous indices for union-find
        let id_to_idx: HashMap<u64, usize> =
            ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let mut parent: Vec<usize> = (0..ids.len()).collect();

        // Union-find helpers
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]]; // path compression
                x = parent[x];
            }
            x
        }
        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        }

        // Merge pairs above threshold
        for (i, &id_a) in ids.iter().enumerate() {
            let vec_a = &self.vectors[&id_a];
            for &id_b in &ids[i + 1..] {
                let vec_b = &self.vectors[&id_b];
                if cosine_similarity(vec_a, vec_b) > threshold {
                    union(&mut parent, id_to_idx[&id_a], id_to_idx[&id_b]);
                }
            }
        }

        // Group by root
        let mut clusters: HashMap<usize, Vec<u64>> = HashMap::new();
        for (i, &id) in ids.iter().enumerate() {
            let root = find(&mut parent, i);
            clusters.entry(root).or_default().push(id);
        }

        clusters.into_values().collect()
    }

    /// Look up a stored embedding vector by concept ID.
    pub fn get_vector(&self, concept_id: u64) -> Option<&Vec<f32>> {
        self.vectors.get(&concept_id)
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
    fn test_cluster_basic() {
        let mut index = EmbeddingIndex::empty();
        // Two similar vectors, one different
        index.vectors.insert(1, vec![1.0, 0.0]);
        index.vectors.insert(2, vec![0.95, 0.05]);
        index.vectors.insert(3, vec![0.0, 1.0]);

        let clusters = index.cluster(0.9);
        // Concepts 1 and 2 should cluster together, 3 separate
        assert!(clusters.len() >= 2);

        // Find which cluster has concept 3
        let cluster_with_3 = clusters.iter().find(|c| c.contains(&3)).unwrap();
        assert_eq!(cluster_with_3.len(), 1);

        // Concepts 1 and 2 should share a cluster
        let cluster_with_1 = clusters.iter().find(|c| c.contains(&1)).unwrap();
        assert!(cluster_with_1.contains(&2));
    }

    #[test]
    fn test_cluster_empty() {
        let index = EmbeddingIndex::empty();
        let clusters = index.cluster(0.9);
        assert!(clusters.is_empty());
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
