// Agglomerative clustering with average linkage using a priority queue.
// Groups similar concepts into clusters, replacing the naive pairwise
// threshold approach that caused transitive chaining.

use crate::embeddings::EmbeddingIndex;
use rayon::prelude::*;
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

pub struct ClusterResult {
    pub assignments: HashMap<u64, usize>,
    #[allow(dead_code)]
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
    let mut cluster_size: Vec<usize> = vec![1; n];
    let mut active_clusters: HashSet<usize> = (0..n).collect();

    // Pre-extract, L2-normalize, and pack embeddings into a contiguous flat
    // array for cache-friendly dot products. After normalization,
    // cosine_sim(a, b) = dot(a, b), avoiding sqrt per pair.
    let dim = concept_ids
        .iter()
        .find_map(|&id| embeddings.get_vector(id).map(|v| v.len()))
        .unwrap_or(0);
    let mut has_emb = vec![false; n];
    let mut flat = vec![0.0_f32; n * dim];
    for (i, &id) in concept_ids.iter().enumerate() {
        if let Some(v) = embeddings.get_vector(id) {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                has_emb[i] = true;
                let row = &mut flat[i * dim..(i + 1) * dim];
                for (dst, src) in row.iter_mut().zip(v.iter()) {
                    *dst = src / norm;
                }
            }
        }
    }

    // Precompute NxN pairwise cosine distance matrix (flat).
    // Parallelized across rows via rayon. Contiguous embedding layout
    // enables auto-vectorization and good cache utilization.
    // After clustering, entries are updated via the Lance-Williams formula.
    let row_distances: Vec<Vec<(usize, f32)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut pairs = Vec::new();
            if has_emb[i] {
                let a = &flat[i * dim..(i + 1) * dim];
                for j in (i + 1)..n {
                    if has_emb[j] {
                        let b = &flat[j * dim..(j + 1) * dim];
                        let dot: f32 =
                            a.iter().zip(b).map(|(x, y)| x * y).sum();
                        pairs.push((j, 1.0 - dot));
                    }
                }
            }
            pairs
        })
        .collect();

    let mut dist = vec![1.0_f32; n * n];
    for i in 0..n {
        dist[i * n + i] = 0.0;
        for &(j, d) in &row_distances[i] {
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
        // Skip stale entries (clusters already merged into something else)
        if !active_clusters.contains(&ca) || !active_clusters.contains(&cb) {
            continue;
        }

        // Check current cluster-level distance (heap entry may be stale
        // if either cluster was updated by an earlier merge)
        let current_dist = dist[ca * n + cb];
        if current_dist > distance_threshold {
            continue;
        }

        // Merge cb into ca
        let size_a = cluster_size[ca];
        let size_b = cluster_size[cb];
        let new_size = size_a + size_b;

        // Update cluster_of for all members of cb
        for label in cluster_of.iter_mut() {
            if *label == cb {
                *label = ca;
            }
        }
        cluster_size[ca] = new_size;
        cluster_size[cb] = 0;
        active_clusters.remove(&cb);

        // Lance-Williams update for average linkage:
        //   d(A∪B, C) = (|A| * d(A,C) + |B| * d(B,C)) / (|A| + |B|)
        // This is O(1) per cluster pair instead of O(|A∪B| * |C|).
        for &other in &active_clusters {
            if other == ca {
                continue;
            }
            let d_ca = dist[ca * n + other];
            let d_cb = dist[cb * n + other];
            let new_d = (size_a as f32 * d_ca + size_b as f32 * d_cb)
                / new_size as f32;
            dist[ca * n + other] = new_d;
            dist[other * n + ca] = new_d;
            if new_d < distance_threshold {
                let (lo, hi) = if ca < other {
                    (ca, other)
                } else {
                    (other, ca)
                };
                heap.push(Reverse((Dist(new_d), lo, hi)));
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
