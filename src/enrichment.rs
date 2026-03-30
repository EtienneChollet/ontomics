use crate::embeddings::EmbeddingIndex;
use crate::graph::ConceptGraph;
use crate::types::Concept;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// Determine which concepts still need embedding vectors.
/// Returns only concepts whose IDs are NOT in `already_embedded`.
pub fn compute_embedding_plan(
    all_concepts: Vec<Concept>,
    already_embedded: &HashSet<u64>,
) -> Vec<Concept> {
    all_concepts
        .into_iter()
        .filter(|c| !already_embedded.contains(&c.id))
        .collect()
}

/// Merge newly-computed vectors from `source` into `graph.embeddings`.
/// Returns count of vectors merged.
pub fn merge_vectors_into_graph(
    graph: &mut ConceptGraph,
    source: &EmbeddingIndex,
    concept_ids: &[u64],
) -> usize {
    let mut merged = 0;
    for &id in concept_ids {
        if let Some(v) = source.get_vector(id) {
            graph.embeddings.insert_vector(id, v.clone());
            merged += 1;
        }
    }
    merged
}

/// Check if the graph generation has changed since this thread started.
/// Returns true if the thread should stop (graph was replaced).
pub fn is_generation_stale(
    current: &AtomicU64,
    my_generation: u64,
) -> bool {
    current.load(Ordering::SeqCst) != my_generation
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EntityType, Occurrence};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn make_concept(id: u64, canonical: &str) -> Concept {
        Concept {
            id,
            canonical: canonical.to_string(),
            subtokens: vec![canonical.to_string()],
            occurrences: vec![Occurrence {
                file: PathBuf::from("test.py"),
                line: 1,
                identifier: canonical.to_string(),
                entity_type: EntityType::Function,
            }],
            entity_types: HashSet::from([EntityType::Function]),
            embedding: None,
            subconcepts: Vec::new(),
        }
    }

    fn make_empty_graph(nb_concepts: u64) -> ConceptGraph {
        let concepts = (0..nb_concepts)
            .map(|i| (i, make_concept(i, &format!("concept_{i}"))))
            .collect();
        ConceptGraph {
            concepts,
            relationships: Vec::new(),
            conventions: Vec::new(),
            embeddings: EmbeddingIndex::empty(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
            entities: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_compute_plan_filters_existing() {
        let concepts: Vec<Concept> =
            (0..10).map(|i| make_concept(i, &format!("c_{i}"))).collect();
        let already: HashSet<u64> = (0..5).collect();

        let plan = compute_embedding_plan(concepts, &already);

        assert_eq!(plan.len(), 5);
        let plan_ids: HashSet<u64> = plan.iter().map(|c| c.id).collect();
        assert_eq!(plan_ids, (5..10).collect::<HashSet<u64>>());
    }

    #[test]
    fn test_compute_plan_all_embedded() {
        let concepts: Vec<Concept> =
            (0..5).map(|i| make_concept(i, &format!("c_{i}"))).collect();
        let already: HashSet<u64> = (0..5).collect();

        let plan = compute_embedding_plan(concepts, &already);
        assert!(plan.is_empty());
    }

    #[test]
    fn test_merge_vectors_into_graph() {
        let mut graph = make_empty_graph(5);
        assert_eq!(graph.embeddings.nb_vectors(), 0);

        let mut source = EmbeddingIndex::empty();
        source.insert_vector(0, vec![1.0, 0.0, 0.0]);
        source.insert_vector(1, vec![0.0, 1.0, 0.0]);
        source.insert_vector(2, vec![0.0, 0.0, 1.0]);

        let merged = merge_vectors_into_graph(
            &mut graph,
            &source,
            &[0, 1, 2],
        );

        assert_eq!(merged, 3);
        assert_eq!(graph.embeddings.nb_vectors(), 3);
        assert!(graph.embeddings.get_vector(0).is_some());
        assert!(graph.embeddings.get_vector(3).is_none());
    }

    #[test]
    fn test_generation_stale_detection() {
        let gen = AtomicU64::new(0);

        assert!(!is_generation_stale(&gen, 0));

        gen.store(1, Ordering::SeqCst);
        assert!(is_generation_stale(&gen, 0));
        assert!(!is_generation_stale(&gen, 1));
    }

    #[test]
    fn test_stale_thread_does_not_merge() {
        let mut graph = make_empty_graph(4);
        let gen = AtomicU64::new(0);
        let my_gen = gen.load(Ordering::SeqCst);

        // Batch 1: embed concepts 0,1 — generation matches
        let mut source = EmbeddingIndex::empty();
        source.insert_vector(0, vec![1.0, 0.0]);
        source.insert_vector(1, vec![0.0, 1.0]);

        assert!(!is_generation_stale(&gen, my_gen));
        merge_vectors_into_graph(&mut graph, &source, &[0, 1]);
        assert_eq!(graph.embeddings.nb_vectors(), 2);

        // Simulate watcher replacing graph
        gen.store(1, Ordering::SeqCst);

        // Batch 2: would embed concepts 2,3 — but generation is stale
        source.insert_vector(2, vec![1.0, 1.0]);
        source.insert_vector(3, vec![0.5, 0.5]);

        assert!(is_generation_stale(&gen, my_gen));
        // Thread should NOT merge — simulating the check
        if !is_generation_stale(&gen, my_gen) {
            merge_vectors_into_graph(&mut graph, &source, &[2, 3]);
        }

        // Only batch 1 vectors present
        assert_eq!(graph.embeddings.nb_vectors(), 2);
        assert!(graph.embeddings.get_vector(0).is_some());
        assert!(graph.embeddings.get_vector(2).is_none());
    }

    /// Simulate the retry loop: thread 1 holds the embedding lock,
    /// thread 2 retries until thread 1 releases it.
    #[test]
    fn test_lock_retry_acquires_after_release() {
        use crate::cache::IndexCache;

        let dir = std::path::Path::new(
            "/tmp/semex_test_lock_retry",
        );
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        let cache = IndexCache::open(dir).unwrap();
        assert!(cache.try_acquire_embedding_lock());

        let gen = Arc::new(AtomicU64::new(0));
        let my_gen = gen.load(Ordering::SeqCst);

        // Spawn thread that retries lock acquisition (mirrors main.rs logic)
        let dir2 = dir.to_path_buf();
        let gen2 = Arc::clone(&gen);
        let handle = std::thread::spawn(move || {
            let cache2 = IndexCache::open(&dir2).unwrap();
            let mut attempts = 0;
            while !cache2.try_acquire_embedding_lock() {
                attempts += 1;
                if attempts > 10 {
                    return false; // gave up
                }
                if is_generation_stale(&gen2, my_gen) {
                    return false; // superseded
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            cache2.release_embedding_lock();
            true // acquired successfully
        });

        // Release after a short delay — thread should pick it up
        std::thread::sleep(std::time::Duration::from_millis(100));
        cache.release_embedding_lock();

        let acquired = handle.join().unwrap();
        assert!(acquired, "retry loop should acquire lock after release");

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Simulate the retry loop giving up when generation changes
    /// while waiting for the lock (another watcher event superseded us).
    #[test]
    fn test_lock_retry_stops_when_generation_changes() {
        use crate::cache::IndexCache;

        let dir = std::path::Path::new(
            "/tmp/semex_test_lock_retry_gen",
        );
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        let cache = IndexCache::open(dir).unwrap();
        assert!(cache.try_acquire_embedding_lock());

        let gen = Arc::new(AtomicU64::new(0));
        let my_gen = gen.load(Ordering::SeqCst);

        // Spawn thread that retries lock acquisition
        let dir2 = dir.to_path_buf();
        let gen2 = Arc::clone(&gen);
        let handle = std::thread::spawn(move || {
            let cache2 = IndexCache::open(&dir2).unwrap();
            let mut attempts = 0;
            while !cache2.try_acquire_embedding_lock() {
                attempts += 1;
                if attempts > 30 {
                    return "timeout";
                }
                if is_generation_stale(&gen2, my_gen) {
                    return "superseded";
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            cache2.release_embedding_lock();
            "acquired"
        });

        // Increment generation while lock is still held — thread
        // should notice and stop retrying.
        std::thread::sleep(std::time::Duration::from_millis(100));
        gen.fetch_add(1, Ordering::SeqCst);

        let result = handle.join().unwrap();
        assert_eq!(
            result, "superseded",
            "retry loop should stop when generation changes"
        );

        cache.release_embedding_lock();
        let _ = std::fs::remove_dir_all(dir);
    }
}
