// L4: Structural centrality scoring (PageRank) on entity graph.

use crate::types::{CentralityScore, Entity, Relationship, RelationshipKind};
use std::collections::{HashMap, HashSet};

/// Compute PageRank centrality for all entities in the graph.
///
/// Only `InheritsFrom` and `Uses` relationships are considered
/// structural edges. Other relationship kinds (CoOccurs, SimilarTo,
/// etc.) are ignored.
///
/// Parameters
/// ----------
/// entities : &HashMap<u64, Entity>
///     All entities keyed by ID.
/// relationships : &[Relationship]
///     Full relationship list (filtered internally).
/// damping : f32
///     PageRank damping factor, typically 0.85.
/// iterations : usize
///     Maximum number of power-iteration steps.
pub fn compute_centrality(
    entities: &HashMap<u64, Entity>,
    relationships: &[Relationship],
    damping: f32,
    iterations: usize,
) -> HashMap<u64, CentralityScore> {
    if entities.is_empty() {
        return HashMap::new();
    }

    let entity_ids: HashSet<u64> = entities.keys().copied().collect();

    // Filter to structural edges only, both endpoints must be entities.
    let edges: Vec<(u64, u64)> = relationships
        .iter()
        .filter(|r| is_structural(&r.kind))
        .filter(|r| {
            entity_ids.contains(&r.source)
                && entity_ids.contains(&r.target)
        })
        .map(|r| (r.source, r.target))
        .collect();

    // Forward adjacency: source -> set of targets.
    let mut forward: HashMap<u64, Vec<u64>> = HashMap::new();
    // Reverse adjacency: target -> set of sources.
    let mut reverse: HashMap<u64, Vec<u64>> = HashMap::new();
    let mut in_degree: HashMap<u64, usize> = HashMap::new();
    let mut out_degree: HashMap<u64, usize> = HashMap::new();

    for &id in &entity_ids {
        forward.entry(id).or_default();
        reverse.entry(id).or_default();
        in_degree.entry(id).or_insert(0);
        out_degree.entry(id).or_insert(0);
    }

    for &(src, tgt) in &edges {
        forward.entry(src).or_default().push(tgt);
        reverse.entry(tgt).or_default().push(src);
        *in_degree.entry(tgt).or_insert(0) += 1;
        *out_degree.entry(src).or_insert(0) += 1;
    }

    // PageRank via power iteration.
    let n = entity_ids.len() as f32;
    let init = 1.0 / n;
    let teleport = (1.0 - damping) / n;

    let mut rank: HashMap<u64, f32> =
        entity_ids.iter().map(|&id| (id, init)).collect();

    for _ in 0..iterations {
        // Dangling node mass: nodes with no outgoing edges
        // redistribute their rank evenly to all nodes.
        let dangling_sum: f32 = entity_ids
            .iter()
            .filter(|id| out_degree[id] == 0)
            .map(|id| rank[id])
            .sum();
        let dangling_share = damping * dangling_sum / n;

        let mut new_rank: HashMap<u64, f32> = HashMap::new();
        let mut max_delta: f32 = 0.0;

        for &id in &entity_ids {
            let incoming = &reverse[&id];
            let sum: f32 = incoming
                .iter()
                .map(|&src| {
                    let od = out_degree[&src];
                    if od > 0 {
                        rank[&src] / od as f32
                    } else {
                        0.0
                    }
                })
                .sum();
            let pr = teleport + damping * sum + dangling_share;
            new_rank.insert(id, pr);

            let delta = (pr - rank[&id]).abs();
            if delta > max_delta {
                max_delta = delta;
            }
        }

        rank = new_rank;

        if max_delta < 1e-6 {
            break;
        }
    }

    entity_ids
        .iter()
        .map(|&id| {
            (
                id,
                CentralityScore {
                    entity_id: id,
                    in_degree: in_degree[&id],
                    out_degree: out_degree[&id],
                    pagerank: rank[&id],
                },
            )
        })
        .collect()
}

fn is_structural(kind: &RelationshipKind) -> bool {
    matches!(kind, RelationshipKind::InheritsFrom | RelationshipKind::Uses)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Entity, EntityKind};
    use std::path::PathBuf;

    fn make_entity(id: u64, name: &str) -> Entity {
        Entity {
            id,
            name: name.to_string(),
            kind: EntityKind::Function,
            concept_tags: vec![],
            semantic_role: String::new(),
            file: PathBuf::from("test.py"),
            line: 1,
            signature_idx: None,
            class_info_idx: None,
        }
    }

    fn make_rel(
        src: u64,
        tgt: u64,
        kind: RelationshipKind,
    ) -> Relationship {
        Relationship {
            source: src,
            target: tgt,
            kind,
            weight: 1.0,
        }
    }

    #[test]
    fn empty_graph() {
        let result =
            compute_centrality(&HashMap::new(), &[], 0.85, 100);
        assert!(result.is_empty());
    }

    #[test]
    fn single_entity() {
        let mut entities = HashMap::new();
        entities.insert(1, make_entity(1, "alone"));

        let result = compute_centrality(&entities, &[], 0.85, 100);
        assert_eq!(result.len(), 1);
        let score = &result[&1];
        assert_eq!(score.in_degree, 0);
        assert_eq!(score.out_degree, 0);
        // Single node: pagerank = 1/N = 1.0
        assert!((score.pagerank - 1.0).abs() < 0.01);
    }

    #[test]
    fn linear_chain_degrees() {
        // A -> B -> C
        let mut entities = HashMap::new();
        entities.insert(1, make_entity(1, "A"));
        entities.insert(2, make_entity(2, "B"));
        entities.insert(3, make_entity(3, "C"));

        let rels = vec![
            make_rel(1, 2, RelationshipKind::Uses),
            make_rel(2, 3, RelationshipKind::Uses),
        ];

        let result = compute_centrality(&entities, &rels, 0.85, 100);

        assert_eq!(result[&1].in_degree, 0);
        assert_eq!(result[&1].out_degree, 1);
        assert_eq!(result[&2].in_degree, 1);
        assert_eq!(result[&2].out_degree, 1);
        assert_eq!(result[&3].in_degree, 1);
        assert_eq!(result[&3].out_degree, 0);

        // C has highest in_degree (tied with B at 1, but C is a sink
        // so it gets rank from B). C should have highest pagerank.
        assert!(result[&3].pagerank > result[&1].pagerank);
    }

    #[test]
    fn star_pattern_highest_pagerank() {
        // A, B, C all point to D
        let mut entities = HashMap::new();
        for (id, name) in [(1, "A"), (2, "B"), (3, "C"), (4, "D")] {
            entities.insert(id, make_entity(id, name));
        }

        let rels = vec![
            make_rel(1, 4, RelationshipKind::Uses),
            make_rel(2, 4, RelationshipKind::Uses),
            make_rel(3, 4, RelationshipKind::Uses),
        ];

        let result = compute_centrality(&entities, &rels, 0.85, 100);

        // D should have the highest pagerank
        let d_rank = result[&4].pagerank;
        assert!(d_rank > result[&1].pagerank);
        assert!(d_rank > result[&2].pagerank);
        assert!(d_rank > result[&3].pagerank);
        assert_eq!(result[&4].in_degree, 3);
    }

    #[test]
    fn cycle_similar_pagerank() {
        // A -> B -> C -> A
        let mut entities = HashMap::new();
        entities.insert(1, make_entity(1, "A"));
        entities.insert(2, make_entity(2, "B"));
        entities.insert(3, make_entity(3, "C"));

        let rels = vec![
            make_rel(1, 2, RelationshipKind::InheritsFrom),
            make_rel(2, 3, RelationshipKind::InheritsFrom),
            make_rel(3, 1, RelationshipKind::InheritsFrom),
        ];

        let result = compute_centrality(&entities, &rels, 0.85, 100);

        // In a cycle, all nodes converge to equal pagerank.
        let a = result[&1].pagerank;
        let b = result[&2].pagerank;
        let c = result[&3].pagerank;
        assert!((a - b).abs() < 0.01);
        assert!((b - c).abs() < 0.01);
    }

    #[test]
    fn only_structural_edges_counted() {
        // A --Uses--> B (should count)
        // A --CoOccurs--> C (should NOT count)
        // A --SimilarTo--> C (should NOT count)
        let mut entities = HashMap::new();
        entities.insert(1, make_entity(1, "A"));
        entities.insert(2, make_entity(2, "B"));
        entities.insert(3, make_entity(3, "C"));

        let rels = vec![
            make_rel(1, 2, RelationshipKind::Uses),
            make_rel(1, 3, RelationshipKind::CoOccurs),
            make_rel(1, 3, RelationshipKind::SimilarTo),
        ];

        let result = compute_centrality(&entities, &rels, 0.85, 100);

        // B has in_degree 1 from the Uses edge
        assert_eq!(result[&2].in_degree, 1);
        // C has in_degree 0 — non-structural edges ignored
        assert_eq!(result[&3].in_degree, 0);
        // A has out_degree 1 (only the Uses edge)
        assert_eq!(result[&1].out_degree, 1);
    }

    #[test]
    fn disconnected_components() {
        // Component 1: A -> B
        // Component 2: C -> D
        let mut entities = HashMap::new();
        for (id, name) in [(1, "A"), (2, "B"), (3, "C"), (4, "D")] {
            entities.insert(id, make_entity(id, name));
        }

        let rels = vec![
            make_rel(1, 2, RelationshipKind::Uses),
            make_rel(3, 4, RelationshipKind::Uses),
        ];

        let result = compute_centrality(&entities, &rels, 0.85, 100);

        assert_eq!(result.len(), 4);
        // Sinks (B and D) should have higher pagerank than sources
        assert!(result[&2].pagerank > result[&1].pagerank);
        assert!(result[&4].pagerank > result[&3].pagerank);
        // Symmetric components: B and D should have similar pagerank
        assert!(
            (result[&2].pagerank - result[&4].pagerank).abs() < 0.01
        );
    }
}
