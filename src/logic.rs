// L4: Logic embedding index and behavioral clustering.

use crate::embeddings::{self, EmbeddingModel};
use crate::types::{LogicCluster, Pseudocode, PseudocodeStep};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Default)]
pub struct LogicIndex {
    model: Option<Box<dyn EmbeddingModel>>,
    vectors: HashMap<u64, Vec<f32>>,
    cache_dir: Option<PathBuf>,
    model_id: String,
}

impl LogicIndex {
    /// Initialize logic index with a specific model.
    pub fn new_with_model(model_id: &str, cache_dir: Option<PathBuf>) -> Result<Self> {
        let model = embeddings::load_model(model_id, cache_dir.as_ref())?;
        Ok(Self {
            model: Some(model),
            vectors: HashMap::new(),
            cache_dir,
            model_id: model_id.to_string(),
        })
    }

    /// Initialize logic index with the default BGE-small model.
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self> {
        Self::new_with_model(embeddings::BGE_SMALL_ID, cache_dir)
    }

    /// Create an empty logic index (no model, empty vectors).
    pub fn empty() -> Self {
        Self::default()
    }

    /// Embed pseudocode text for a single entity.
    pub fn embed_pseudocode(
        &mut self,
        entity_id: u64,
        text: &str,
    ) -> Result<Vec<f32>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("no embedding model loaded"))?;
        let vecs = model.embed(vec![text.to_string()])?;
        let vector = vecs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("embedding model returned no vectors"))?;
        self.vectors.insert(entity_id, vector.clone());
        Ok(vector)
    }

    /// Batch embed pseudocode for multiple entities.
    pub fn embed_batch(&mut self, items: Vec<(u64, String)>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("no embedding model loaded"))?;
        let ids: Vec<u64> = items.iter().map(|(id, _)| *id).collect();
        let texts: Vec<String> = items.into_iter().map(|(_, t)| t).collect();
        let vecs = model.embed(texts)?;
        for (id, vec) in ids.into_iter().zip(vecs) {
            self.vectors.insert(id, vec);
        }
        Ok(())
    }

    /// Find entities with similar logic to the query vector.
    pub fn find_similar(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Vec<(u64, f32)> {
        let mut scored: Vec<(u64, f32)> = self
            .vectors
            .iter()
            .map(|(&id, vec)| (id, cosine_similarity(query, vec)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        scored
    }

    /// Find entities with similar logic to a given entity.
    pub fn find_similar_to_entity(
        &self,
        entity_id: u64,
        top_k: usize,
    ) -> Vec<(u64, f32)> {
        let query = match self.vectors.get(&entity_id) {
            Some(v) => v.clone(),
            None => return Vec::new(),
        };
        self.find_similar(&query, top_k + 1)
            .into_iter()
            .filter(|(id, _)| *id != entity_id)
            .take(top_k)
            .collect()
    }

    /// Get vector for an entity.
    pub fn get_vector(&self, entity_id: u64) -> Option<&Vec<f32>> {
        self.vectors.get(&entity_id)
    }

    /// Get all entity IDs that have vectors.
    pub fn vector_ids(&self) -> Vec<u64> {
        self.vectors.keys().copied().collect()
    }

    /// Number of stored vectors.
    pub fn nb_vectors(&self) -> usize {
        self.vectors.len()
    }

    /// Reload model (for use after deserialization).
    pub fn load_model(&mut self) -> Result<()> {
        if self.model.is_some() {
            return Ok(());
        }
        let id = if self.model_id.is_empty() {
            embeddings::BGE_SMALL_ID
        } else {
            &self.model_id
        };
        let model = embeddings::load_model(id, self.cache_dir.as_ref())?;
        self.model = Some(model);
        Ok(())
    }

    /// Set cache directory for model weights.
    pub fn set_cache_dir(&mut self, dir: PathBuf) {
        self.cache_dir = Some(dir);
    }

    /// Insert a single vector directly (for testing and merge operations).
    pub fn insert_vector(&mut self, id: u64, vec: Vec<f32>) {
        self.vectors.insert(id, vec);
    }
}

// -- Custom serde: only serialize vectors, skip model and cache_dir --

impl Serialize for LogicIndex {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        self.vectors.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LogicIndex {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        let vectors = HashMap::deserialize(deserializer)?;
        Ok(Self {
            model: None,
            vectors,
            cache_dir: None,
            model_id: String::new(),
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

// -- Logic clustering --

/// Cluster entities by logic embedding similarity using agglomerative
/// clustering with average linkage.
pub fn cluster_logic(
    logic_index: &LogicIndex,
    entity_ids: &[u64],
    distance_threshold: f32,
) -> Vec<LogicCluster> {
    // Filter to entities that have vectors
    let ids_with_vecs: Vec<u64> = entity_ids
        .iter()
        .filter(|id| logic_index.get_vector(**id).is_some())
        .copied()
        .collect();
    let n = ids_with_vecs.len();

    if n == 0 {
        return Vec::new();
    }

    // Extract vectors for cache-friendly access.
    // ids_with_vecs is pre-filtered to IDs with vectors, so filter_map
    // is a safety net — it should never actually drop an entry.
    let vecs: Vec<&Vec<f32>> = ids_with_vecs
        .iter()
        .filter_map(|id| logic_index.get_vector(*id))
        .collect();

    // Compute pairwise cosine distance matrix (flat array)
    let mut dist = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = 1.0 - cosine_similarity(vecs[i], vecs[j]);
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }

    // Initialize each entity as its own cluster
    let mut cluster_of: Vec<usize> = (0..n).collect();
    let mut cluster_size: Vec<usize> = vec![1; n];
    let mut active: Vec<bool> = vec![true; n];

    // Merge loop: find closest pair, merge if below threshold
    loop {
        let mut best_dist = f32::MAX;
        let mut best_i = 0;
        let mut best_j = 0;

        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                if dist[i * n + j] < best_dist {
                    best_dist = dist[i * n + j];
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if best_dist >= distance_threshold {
            break;
        }

        // Merge best_j into best_i (Lance-Williams average linkage)
        let size_i = cluster_size[best_i];
        let size_j = cluster_size[best_j];
        let new_size = size_i + size_j;

        for k in 0..n {
            if !active[k] || k == best_i || k == best_j {
                continue;
            }
            let new_d = (size_i as f32 * dist[best_i * n + k]
                + size_j as f32 * dist[best_j * n + k])
                / new_size as f32;
            dist[best_i * n + k] = new_d;
            dist[k * n + best_i] = new_d;
        }

        // Update membership
        let merged_label = cluster_of[best_j];
        let target_label = cluster_of[best_i];
        for label in cluster_of.iter_mut() {
            if *label == merged_label {
                *label = target_label;
            }
        }
        cluster_size[best_i] = new_size;
        active[best_j] = false;
    }

    // Collect clusters and compute centroids
    let mut cluster_members: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, &label) in cluster_of.iter().enumerate() {
        cluster_members.entry(label).or_default().push(idx);
    }

    let dim = vecs.first().map(|v| v.len()).unwrap_or(0);
    let mut result: Vec<LogicCluster> = cluster_members
        .into_iter()
        .enumerate()
        .map(|(cluster_id, (_label, members))| {
            let entity_ids: Vec<u64> =
                members.iter().map(|&idx| ids_with_vecs[idx]).collect();
            let centroid = compute_centroid(
                &members.iter().map(|&idx| vecs[idx].as_slice()).collect::<Vec<_>>(),
                dim,
            );
            LogicCluster {
                id: cluster_id,
                entity_ids,
                centroid,
                behavioral_label: None,
            }
        })
        .collect();

    // Stable ordering by first entity ID
    result.sort_by_key(|c| c.entity_ids.first().copied().unwrap_or(0));
    for (i, c) in result.iter_mut().enumerate() {
        c.id = i;
    }

    result
}

fn compute_centroid(vectors: &[&[f32]], dim: usize) -> Vec<f32> {
    if vectors.is_empty() || dim == 0 {
        return Vec::new();
    }
    let mut centroid = vec![0.0_f32; dim];
    for v in vectors {
        for (c, x) in centroid.iter_mut().zip(v.iter()) {
            *c += x;
        }
    }
    let n = vectors.len() as f32;
    for c in centroid.iter_mut() {
        *c /= n;
    }
    centroid
}

/// Label logic clusters by the most common callee across member pseudocode.
pub fn label_clusters(
    clusters: &mut [LogicCluster],
    pseudocode: &HashMap<u64, Pseudocode>,
) {
    for cluster in clusters.iter_mut() {
        let mut callee_counts: HashMap<&str, usize> = HashMap::new();
        for entity_id in &cluster.entity_ids {
            if let Some(pc) = pseudocode.get(entity_id) {
                collect_callees(&pc.steps, &mut callee_counts);
            }
        }
        // Find the most common callee
        cluster.behavioral_label = callee_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(callee, _)| callee.to_string());
    }
}

fn collect_callees<'a>(
    steps: &'a [PseudocodeStep],
    counts: &mut HashMap<&'a str, usize>,
) {
    for step in steps {
        match step {
            PseudocodeStep::Call { callee, .. } => {
                *counts.entry(callee.as_str()).or_insert(0) += 1;
            }
            PseudocodeStep::Conditional { branches } => {
                for branch in branches {
                    collect_callees(&branch.body, counts);
                }
            }
            PseudocodeStep::Loop { body, .. } => {
                collect_callees(body, counts);
            }
            _ => {}
        }
    }
}

/// Compute Jaccard overlap between logic clusters and concept clusters.
/// Returns (logic_cluster_id, concept_cluster_id, overlap) tuples
/// exceeding `min_overlap`.
pub fn cross_reference(
    logic_clusters: &[LogicCluster],
    concept_cluster_assignments: &HashMap<u64, usize>,
    min_overlap: f32,
) -> Vec<(usize, usize, f32)> {
    // Invert concept_cluster_assignments: cluster_id -> set of entity_ids
    let mut concept_clusters: HashMap<usize, Vec<u64>> = HashMap::new();
    for (&entity_id, &cluster_id) in concept_cluster_assignments {
        concept_clusters.entry(cluster_id).or_default().push(entity_id);
    }

    let mut results = Vec::new();
    for lc in logic_clusters {
        let lc_set: std::collections::HashSet<u64> =
            lc.entity_ids.iter().copied().collect();
        for (&cc_id, cc_members) in &concept_clusters {
            let cc_set: std::collections::HashSet<u64> =
                cc_members.iter().copied().collect();
            let intersection = lc_set.intersection(&cc_set).count();
            let union = lc_set.union(&cc_set).count();
            if union == 0 {
                continue;
            }
            let jaccard = intersection as f32 / union as f32;
            if jaccard >= min_overlap {
                results.push((lc.id, cc_id, jaccard));
            }
        }
    }

    results.sort_by(|a, b| {
        b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ConditionalBranch;

    #[test]
    fn test_empty_index_has_zero_vectors() {
        let idx = LogicIndex::empty();
        assert_eq!(idx.nb_vectors(), 0);
    }

    #[test]
    fn test_insert_and_get_vector() {
        let mut idx = LogicIndex::empty();
        idx.insert_vector(1, vec![1.0, 0.0, 0.0]);
        assert_eq!(idx.nb_vectors(), 1);
        assert_eq!(idx.get_vector(1).unwrap(), &vec![1.0, 0.0, 0.0]);
        assert!(idx.get_vector(2).is_none());
    }

    #[test]
    fn test_vector_ids() {
        let mut idx = LogicIndex::empty();
        idx.insert_vector(10, vec![1.0]);
        idx.insert_vector(20, vec![2.0]);
        let mut ids = idx.vector_ids();
        ids.sort();
        assert_eq!(ids, vec![10, 20]);
    }

    #[test]
    fn test_find_similar_returns_ranked_results() {
        let mut idx = LogicIndex::empty();
        idx.insert_vector(1, vec![1.0, 0.0, 0.0]);
        idx.insert_vector(2, vec![0.9, 0.1, 0.0]);
        idx.insert_vector(3, vec![0.0, 1.0, 0.0]);

        let results = idx.find_similar(&[1.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        // ID 1 is most similar (identical), ID 3 is least (orthogonal)
        assert_eq!(results[0].0, 1);
        assert_eq!(results[2].0, 3);
        assert!(results[0].1 > results[1].1);
        assert!(results[1].1 > results[2].1);
    }

    #[test]
    fn test_find_similar_top_k_truncation() {
        let mut idx = LogicIndex::empty();
        idx.insert_vector(1, vec![1.0, 0.0]);
        idx.insert_vector(2, vec![0.9, 0.1]);
        idx.insert_vector(3, vec![0.5, 0.5]);

        let results = idx.find_similar(&[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_find_similar_to_entity_excludes_self() {
        let mut idx = LogicIndex::empty();
        idx.insert_vector(1, vec![1.0, 0.0, 0.0]);
        idx.insert_vector(2, vec![0.9, 0.1, 0.0]);
        idx.insert_vector(3, vec![0.0, 1.0, 0.0]);

        let results = idx.find_similar_to_entity(1, 5);
        // Should not contain entity 1
        assert!(results.iter().all(|(id, _)| *id != 1));
        assert_eq!(results.len(), 2);
        // Entity 2 should be more similar than entity 3
        assert_eq!(results[0].0, 2);
        assert_eq!(results[1].0, 3);
    }

    #[test]
    fn test_find_similar_to_entity_missing_returns_empty() {
        let idx = LogicIndex::empty();
        let results = idx.find_similar_to_entity(999, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_cluster_logic_identical_vectors_same_cluster() {
        let mut idx = LogicIndex::empty();
        idx.insert_vector(1, vec![1.0, 0.0, 0.0]);
        idx.insert_vector(2, vec![1.0, 0.0, 0.0]);
        idx.insert_vector(3, vec![1.0, 0.0, 0.0]);

        let clusters = cluster_logic(&idx, &[1, 2, 3], 0.5);
        // All identical vectors should merge into one cluster
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].entity_ids.len(), 3);
    }

    #[test]
    fn test_cluster_logic_orthogonal_vectors_different_clusters() {
        let mut idx = LogicIndex::empty();
        idx.insert_vector(1, vec![1.0, 0.0, 0.0]);
        idx.insert_vector(2, vec![0.0, 1.0, 0.0]);
        idx.insert_vector(3, vec![0.0, 0.0, 1.0]);

        // Orthogonal vectors have distance 1.0, threshold 0.5 prevents merge
        let clusters = cluster_logic(&idx, &[1, 2, 3], 0.5);
        assert_eq!(clusters.len(), 3);
    }

    #[test]
    fn test_cluster_logic_two_groups() {
        let mut idx = LogicIndex::empty();
        // Group A: similar vectors
        idx.insert_vector(1, vec![1.0, 0.0, 0.0]);
        idx.insert_vector(2, vec![0.95, 0.05, 0.0]);
        // Group B: similar vectors, far from A
        idx.insert_vector(3, vec![0.0, 1.0, 0.0]);
        idx.insert_vector(4, vec![0.05, 0.95, 0.0]);

        let clusters = cluster_logic(&idx, &[1, 2, 3, 4], 0.25);
        assert_eq!(clusters.len(), 2);

        // Find which cluster each entity is in
        let cluster_of = |eid: u64| -> usize {
            clusters
                .iter()
                .find(|c| c.entity_ids.contains(&eid))
                .unwrap()
                .id
        };
        assert_eq!(cluster_of(1), cluster_of(2));
        assert_eq!(cluster_of(3), cluster_of(4));
        assert_ne!(cluster_of(1), cluster_of(3));
    }

    #[test]
    fn test_cluster_logic_empty_input() {
        let idx = LogicIndex::empty();
        let clusters = cluster_logic(&idx, &[], 0.5);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_cluster_logic_skips_missing_vectors() {
        let mut idx = LogicIndex::empty();
        idx.insert_vector(1, vec![1.0, 0.0]);
        // Entity 2 has no vector — should be skipped
        idx.insert_vector(3, vec![0.95, 0.05]);

        let clusters = cluster_logic(&idx, &[1, 2, 3], 0.25);
        // Only entities 1 and 3 should appear
        let all_ids: Vec<u64> = clusters
            .iter()
            .flat_map(|c| c.entity_ids.iter().copied())
            .collect();
        assert!(all_ids.contains(&1));
        assert!(!all_ids.contains(&2));
        assert!(all_ids.contains(&3));
    }

    #[test]
    fn test_cluster_centroids_computed() {
        let mut idx = LogicIndex::empty();
        idx.insert_vector(1, vec![1.0, 0.0]);
        idx.insert_vector(2, vec![0.0, 1.0]);

        // With a very large threshold, everything merges
        let clusters = cluster_logic(&idx, &[1, 2], 2.0);
        assert_eq!(clusters.len(), 1);
        let centroid = &clusters[0].centroid;
        assert_eq!(centroid.len(), 2);
        assert!((centroid[0] - 0.5).abs() < 1e-6);
        assert!((centroid[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_label_clusters_extracts_most_common_callee() {
        let mut pseudocode = HashMap::new();
        pseudocode.insert(
            1,
            Pseudocode {
                entity_id: 1,
                steps: vec![
                    PseudocodeStep::Call {
                        callee: "process".to_string(),
                        args: vec![],
                    },
                    PseudocodeStep::Call {
                        callee: "validate".to_string(),
                        args: vec![],
                    },
                    PseudocodeStep::Call {
                        callee: "process".to_string(),
                        args: vec![],
                    },
                ],
                body_hash: 0,
                omitted_count: 0,
            },
        );
        pseudocode.insert(
            2,
            Pseudocode {
                entity_id: 2,
                steps: vec![PseudocodeStep::Call {
                    callee: "process".to_string(),
                    args: vec![],
                }],
                body_hash: 0,
                omitted_count: 0,
            },
        );

        let mut clusters = vec![LogicCluster {
            id: 0,
            entity_ids: vec![1, 2],
            centroid: vec![],
            behavioral_label: None,
        }];

        label_clusters(&mut clusters, &pseudocode);
        assert_eq!(
            clusters[0].behavioral_label.as_deref(),
            Some("process")
        );
    }

    #[test]
    fn test_label_clusters_with_nested_calls() {
        let mut pseudocode = HashMap::new();
        pseudocode.insert(
            1,
            Pseudocode {
                entity_id: 1,
                steps: vec![
                    PseudocodeStep::Conditional {
                        branches: vec![ConditionalBranch {
                            condition: Some("x > 0".to_string()),
                            body: vec![
                                PseudocodeStep::Call {
                                    callee: "inner".to_string(),
                                    args: vec![],
                                },
                                PseudocodeStep::Call {
                                    callee: "inner".to_string(),
                                    args: vec![],
                                },
                            ],
                        }],
                    },
                    PseudocodeStep::Call {
                        callee: "outer".to_string(),
                        args: vec![],
                    },
                ],
                body_hash: 0,
                omitted_count: 0,
            },
        );

        let mut clusters = vec![LogicCluster {
            id: 0,
            entity_ids: vec![1],
            centroid: vec![],
            behavioral_label: None,
        }];

        label_clusters(&mut clusters, &pseudocode);
        // "inner" appears twice, "outer" once
        assert_eq!(
            clusters[0].behavioral_label.as_deref(),
            Some("inner")
        );
    }

    #[test]
    fn test_label_clusters_no_calls_means_no_label() {
        let mut pseudocode = HashMap::new();
        pseudocode.insert(
            1,
            Pseudocode {
                entity_id: 1,
                steps: vec![PseudocodeStep::Return {
                    value: Some("42".to_string()),
                }],
                body_hash: 0,
                omitted_count: 0,
            },
        );

        let mut clusters = vec![LogicCluster {
            id: 0,
            entity_ids: vec![1],
            centroid: vec![],
            behavioral_label: None,
        }];

        label_clusters(&mut clusters, &pseudocode);
        assert!(clusters[0].behavioral_label.is_none());
    }

    #[test]
    fn test_cross_reference_computes_jaccard() {
        let logic_clusters = vec![
            LogicCluster {
                id: 0,
                entity_ids: vec![1, 2, 3],
                centroid: vec![],
                behavioral_label: None,
            },
            LogicCluster {
                id: 1,
                entity_ids: vec![4, 5],
                centroid: vec![],
                behavioral_label: None,
            },
        ];

        // Concept cluster 0 has entities {1, 2}, concept cluster 1 has {3, 4, 5}
        let mut concept_assignments = HashMap::new();
        concept_assignments.insert(1_u64, 0_usize);
        concept_assignments.insert(2, 0);
        concept_assignments.insert(3, 1);
        concept_assignments.insert(4, 1);
        concept_assignments.insert(5, 1);

        let results = cross_reference(&logic_clusters, &concept_assignments, 0.0);
        assert!(!results.is_empty());

        // Logic cluster 0 ({1,2,3}) vs concept cluster 0 ({1,2}):
        //   intersection = {1,2} = 2, union = {1,2,3} = 3, jaccard = 2/3
        let lc0_cc0 = results
            .iter()
            .find(|(lc, cc, _)| *lc == 0 && *cc == 0)
            .unwrap();
        assert!((lc0_cc0.2 - 2.0 / 3.0).abs() < 1e-6);

        // Logic cluster 1 ({4,5}) vs concept cluster 1 ({3,4,5}):
        //   intersection = {4,5} = 2, union = {3,4,5} = 3, jaccard = 2/3
        let lc1_cc1 = results
            .iter()
            .find(|(lc, cc, _)| *lc == 1 && *cc == 1)
            .unwrap();
        assert!((lc1_cc1.2 - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_cross_reference_min_overlap_filter() {
        let logic_clusters = vec![LogicCluster {
            id: 0,
            entity_ids: vec![1, 2, 3],
            centroid: vec![],
            behavioral_label: None,
        }];

        let mut concept_assignments = HashMap::new();
        concept_assignments.insert(1_u64, 0_usize);
        // Jaccard = 1/3 ≈ 0.33

        let results = cross_reference(&logic_clusters, &concept_assignments, 0.5);
        // 1/3 < 0.5, so nothing passes
        assert!(results.is_empty());
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut idx = LogicIndex::empty();
        idx.insert_vector(1, vec![1.0, 2.0, 3.0]);
        idx.insert_vector(2, vec![4.0, 5.0, 6.0]);

        let json = serde_json::to_string(&idx).unwrap();
        let deserialized: LogicIndex = serde_json::from_str(&json).unwrap();

        assert_eq!(
            deserialized.get_vector(1).unwrap(),
            &vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            deserialized.get_vector(2).unwrap(),
            &vec![4.0, 5.0, 6.0]
        );
        assert_eq!(deserialized.nb_vectors(), 2);
        // Model should be None after deserialization
        assert!(deserialized.model.is_none());
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    #[ignore] // requires model download
    fn test_embed_pseudocode_produces_vector() {
        let mut idx = LogicIndex::new(None).unwrap();
        let vec = idx
            .embed_pseudocode(1, "CALL process(item)\nRETURN result")
            .unwrap();
        assert_eq!(vec.len(), 384); // BGE-Small dimension
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    #[ignore] // requires model download
    fn test_embed_batch_stores_vectors() {
        let mut idx = LogicIndex::new(None).unwrap();
        let items = vec![
            (1, "CALL process(item)\nRETURN result".to_string()),
            (2, "FOR x IN items:\n  CALL validate(x)".to_string()),
        ];
        idx.embed_batch(items).unwrap();
        assert_eq!(idx.nb_vectors(), 2);
        assert_eq!(idx.get_vector(1).unwrap().len(), 384);
        assert_eq!(idx.get_vector(2).unwrap().len(), 384);
    }

    #[test]
    fn test_embed_pseudocode_no_model() {
        let mut idx = LogicIndex::empty();
        let result = idx.embed_pseudocode(1, "CALL foo()");
        assert!(result.is_err());
    }

    #[test]
    fn test_embed_batch_no_model() {
        let mut idx = LogicIndex::empty();
        let result = idx.embed_batch(vec![(1, "CALL foo()".to_string())]);
        assert!(result.is_err());
    }

    #[test]
    fn test_embed_batch_empty_is_ok() {
        let mut idx = LogicIndex::empty();
        let result = idx.embed_batch(vec![]);
        assert!(result.is_ok());
    }
}
