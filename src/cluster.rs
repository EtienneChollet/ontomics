// Agglomerative clustering with average linkage using a priority queue.
// Groups similar concepts into clusters, replacing the naive pairwise
// threshold approach that caused transitive chaining.

use crate::embeddings::EmbeddingIndex;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Newtype for f32 that implements total ordering via `f32::total_cmp`.
/// Required because f32 doesn't implement Ord (NaN), but our distances
/// are always in [0.0, 1.0] so NaN never appears.
#[derive(Clone, Copy, PartialEq)]
struct Dist(f32);

impl Eq for Dist {}

impl Ord for Dist {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for Dist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
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

pub struct ClusterResult {
    pub assignments: HashMap<u64, usize>,
    pub nb_clusters: usize,
}

/// Agglomerative clustering with average linkage.
///
/// Groups concepts into clusters based on embedding similarity. Uses a
/// priority queue (min-heap) for O(N^2 log N) runtime instead of the
/// naive O(N^3) merge loop.
///
/// Parameters
/// ----------
/// concept_ids : &[u64]
///     The concept IDs to cluster.
/// embeddings : &EmbeddingIndex
///     Embedding vectors for similarity computation. Concepts without
///     embeddings are treated as distance 1.0 from everything (singletons).
/// distance_threshold : f32
///     Maximum average-linkage distance for merging two clusters.
///     Derived from similarity threshold as `1.0 - similarity_threshold`.
pub fn agglomerative_cluster(
    concept_ids: &[u64],
    embeddings: &EmbeddingIndex,
    distance_threshold: f32,
) -> ClusterResult {
    let n = concept_ids.len();

    if n == 0 {
        return ClusterResult {
            assignments: HashMap::new(),
            nb_clusters: 0,
        };
    }

    if n == 1 {
        let mut assignments = HashMap::new();
        assignments.insert(concept_ids[0], 0);
        return ClusterResult {
            assignments,
            nb_clusters: 1,
        };
    }

    // Phase 1: Build initial state
    // cluster_of[i] = cluster label for concept at index i
    let mut cluster_of: Vec<usize> = (0..n).collect();
    // cluster_members[label] = set of concept indices in that cluster
    let mut cluster_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut active_clusters: HashSet<usize> = (0..n).collect();

    // Precompute NxN pairwise cosine distance matrix (flat)
    let mut dist = vec![1.0_f32; n * n];
    for i in 0..n {
        dist[i * n + i] = 0.0;
        let emb_i = embeddings.get_vector(concept_ids[i]);
        for j in (i + 1)..n {
            let emb_j = embeddings.get_vector(concept_ids[j]);
            let d = match (emb_i, emb_j) {
                (Some(a), Some(b)) => 1.0 - cosine_similarity(a, b),
                _ => 1.0,
            };
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }

    // Phase 2: Initialize min-heap with all pairs below threshold
    let mut heap: BinaryHeap<Reverse<(Dist, usize, usize)>> = BinaryHeap::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if dist[i * n + j] < distance_threshold {
                heap.push(Reverse((Dist(dist[i * n + j]), i, j)));
            }
        }
    }

    // Phase 3: Merge loop
    while let Some(Reverse((_d, ca, cb))) = heap.pop() {
        // Skip stale entries
        if !active_clusters.contains(&ca) || !active_clusters.contains(&cb) {
            continue;
        }

        // Recompute average linkage between current cluster members
        let members_a = &cluster_members[ca];
        let members_b = &cluster_members[cb];
        let mut sum = 0.0_f32;
        let count = members_a.len() * members_b.len();
        for &i in members_a {
            for &j in members_b {
                sum += dist[i * n + j];
            }
        }
        let actual_dist = sum / count as f32;

        if actual_dist > distance_threshold {
            continue;
        }

        // Merge cb into ca
        let cb_members: Vec<usize> = cluster_members[cb].clone();
        for &idx in &cb_members {
            cluster_of[idx] = ca;
        }
        cluster_members[ca].extend(cb_members);
        cluster_members[cb].clear();
        active_clusters.remove(&cb);

        // Push updated distances from merged cluster to all remaining
        let ca_members = cluster_members[ca].clone();
        for &other in &active_clusters {
            if other == ca {
                continue;
            }
            let other_members = &cluster_members[other];
            let mut s = 0.0_f32;
            let c = ca_members.len() * other_members.len();
            for &i in &ca_members {
                for &j in other_members {
                    s += dist[i * n + j];
                }
            }
            let avg_d = s / c as f32;
            if avg_d < distance_threshold {
                heap.push(Reverse((Dist(avg_d), ca, other)));
            }
        }
    }

    // Phase 4: Compact labels to contiguous 0..nb_clusters
    let mut label_map: HashMap<usize, usize> = HashMap::new();
    let mut next_label = 0usize;
    let mut assignments = HashMap::new();

    for (idx, &cluster) in cluster_of.iter().enumerate() {
        let compact = *label_map.entry(cluster).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        assignments.insert(concept_ids[idx], compact);
    }

    ClusterResult {
        assignments,
        nb_clusters: next_label,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let embeddings = EmbeddingIndex::empty();
        let result = agglomerative_cluster(&[], &embeddings, 0.25);
        assert_eq!(result.nb_clusters, 0);
        assert!(result.assignments.is_empty());
    }

    #[test]
    fn test_all_singletons() {
        // 4 concepts with orthogonal embeddings -- all pairwise sim ~0.0,
        // distance ~1.0, well above any reasonable threshold.
        let mut embeddings = EmbeddingIndex::empty();
        embeddings.insert_vector(1, vec![1.0, 0.0, 0.0]);
        embeddings.insert_vector(2, vec![0.0, 1.0, 0.0]);
        embeddings.insert_vector(3, vec![0.0, 0.0, 1.0]);
        embeddings.insert_vector(4, vec![-1.0, 0.0, 0.0]);

        let ids = vec![1, 2, 3, 4];
        let result = agglomerative_cluster(&ids, &embeddings, 0.25);

        assert_eq!(result.nb_clusters, 4);
        // Each concept should be in its own cluster
        let labels: HashSet<usize> = result.assignments.values().copied().collect();
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_two_clusters() {
        // (A,B) similar, (C,D) similar, cross-pair distant
        // A=[1,0,0], B=[0.95,0.05,0] → sim ~0.998, dist ~0.002
        // C=[0,1,0], D=[0.05,0.95,0] → sim ~0.998, dist ~0.002
        // A~C: [1,0,0] vs [0,1,0] → sim 0.0, dist 1.0
        let mut embeddings = EmbeddingIndex::empty();
        embeddings.insert_vector(10, vec![1.0, 0.0, 0.0]);
        embeddings.insert_vector(11, vec![0.95, 0.05, 0.0]);
        embeddings.insert_vector(20, vec![0.0, 1.0, 0.0]);
        embeddings.insert_vector(21, vec![0.05, 0.95, 0.0]);

        let ids = vec![10, 11, 20, 21];
        // threshold=0.25 means distance must be < 0.25 to merge
        let result = agglomerative_cluster(&ids, &embeddings, 0.25);

        assert_eq!(result.nb_clusters, 2);
        // A and B must share a cluster
        assert_eq!(result.assignments[&10], result.assignments[&11]);
        // C and D must share a cluster
        assert_eq!(result.assignments[&20], result.assignments[&21]);
        // The two clusters must be different
        assert_ne!(result.assignments[&10], result.assignments[&20]);
    }

    #[test]
    fn test_chaining_prevented() {
        // A~B (sim 0.95, dist 0.05), A~C (sim 0.3, dist 0.7), B~C (sim 0.285, dist 0.715)
        // With threshold 0.5: A-B merge first (dist 0.05 is clearly smallest).
        // Then avg dist({A,B}, {C}) = (0.7 + 0.715) / 2 = 0.7075 > 0.5, no merge.
        // Result: {A,B} and {C} -- chaining prevented.
        let mut embeddings = EmbeddingIndex::empty();
        embeddings.insert_vector(1, vec![1.0, 0.0, 0.0]);
        embeddings.insert_vector(2, vec![0.95, 0.3122, 0.0]);
        embeddings.insert_vector(3, vec![0.3, 0.0, 0.954]);

        let ids = vec![1, 2, 3];
        let result = agglomerative_cluster(&ids, &embeddings, 0.5);

        assert_eq!(result.nb_clusters, 2);
        // A and B in same cluster
        assert_eq!(result.assignments[&1], result.assignments[&2]);
        // C in its own cluster
        assert_ne!(result.assignments[&1], result.assignments[&3]);
    }

    #[test]
    fn test_no_embedding_is_singleton() {
        // Concept 1 has an embedding, concept 2 does not, concept 3 has one
        // similar to concept 1. Concept 2 must remain a singleton.
        let mut embeddings = EmbeddingIndex::empty();
        embeddings.insert_vector(1, vec![1.0, 0.0, 0.0]);
        // No embedding for concept 2
        embeddings.insert_vector(3, vec![0.99, 0.1, 0.0]);

        let ids = vec![1, 2, 3];
        let result = agglomerative_cluster(&ids, &embeddings, 0.25);

        // Concept 2 must be in its own cluster
        let c2_label = result.assignments[&2];
        assert_ne!(c2_label, result.assignments[&1]);
        assert_ne!(c2_label, result.assignments[&3]);

        // Concepts 1 and 3 should be together (very similar)
        assert_eq!(result.assignments[&1], result.assignments[&3]);
    }
}
