mod cluster;
mod diag;
mod file;
mod logic;
mod naming;
mod query;
mod trace;

use crate::embeddings::EmbeddingIndex;
use crate::logic::LogicIndex;
use crate::types::{
    AnalysisResult, CallSite, CentralityScore, ClassInfo, Convention,
    Entity, FileNestingTree, ImportStatement, LogicCluster, Relationship,
    RelationshipKind, Signature,
};
use anyhow::Result;
use std::collections::HashMap;

pub(super) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

pub struct ConceptGraph {
    pub concepts: HashMap<u64, crate::types::Concept>,
    pub relationships: Vec<Relationship>,
    pub conventions: Vec<Convention>,
    pub embeddings: EmbeddingIndex,
    pub signatures: Vec<Signature>,
    pub classes: Vec<ClassInfo>,
    pub call_sites: Vec<CallSite>,
    pub entities: HashMap<u64, Entity>,
    /// Mean embedding per cluster label, computed after clustering.
    pub cluster_centroids: HashMap<usize, Vec<f32>>,
    /// L4: Logic embedding index for behavioral similarity search.
    pub logic_index: LogicIndex,
    /// L4: Clusters of entities with similar behavioral patterns.
    pub logic_clusters: Vec<LogicCluster>,
    /// L4: PageRank centrality scores, keyed by entity ID.
    pub centrality: HashMap<u64, CentralityScore>,
    /// L4: Jaccard overlaps between logic clusters and concept clusters.
    /// Each tuple is (logic_cluster_id, concept_cluster_id, jaccard_score).
    pub logic_concept_overlaps: Vec<(usize, usize, f32)>,
    pub nesting_trees: Vec<FileNestingTree>,
    pub imports: Vec<ImportStatement>,
}

impl ConceptGraph {
    /// Create an empty graph (no concepts, relationships, or entities).
    /// Used as placeholder during deferred startup.
    pub fn empty() -> Self {
        Self {
            concepts: HashMap::new(),
            relationships: Vec::new(),
            conventions: Vec::new(),
            embeddings: EmbeddingIndex::empty(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
            entities: HashMap::new(),
            cluster_centroids: HashMap::new(),

            logic_index: LogicIndex::empty(),
            logic_clusters: Vec::new(),
            centrality: HashMap::new(),
            logic_concept_overlaps: Vec::new(),
            nesting_trees: Vec::new(),
            imports: Vec::new(),
        }
    }

    /// Build graph from analysis results + embeddings + entities.
    pub fn build(
        analysis: AnalysisResult,
        embeddings: EmbeddingIndex,
    ) -> Result<Self> {
        Self::build_with_entities(
            analysis, embeddings, Vec::new(), Vec::new(), Vec::new(),
        )
    }

    /// Build graph with pre-built entities, relationships, and import data.
    pub fn build_with_entities(
        analysis: AnalysisResult,
        embeddings: EmbeddingIndex,
        entities: Vec<Entity>,
        entity_relationships: Vec<Relationship>,
        imports: Vec<ImportStatement>,
    ) -> Result<Self> {
        let concepts: HashMap<u64, crate::types::Concept> = analysis
            .concepts
            .into_iter()
            .map(|c| (c.id, c))
            .collect();

        let mut relationships: Vec<Relationship> = analysis
            .co_occurrence_matrix
            .into_iter()
            .map(|((src, tgt), weight)| Relationship {
                source: src,
                target: tgt,
                kind: RelationshipKind::CoOccurs,
                weight,
            })
            .collect();

        relationships.extend(entity_relationships);

        let entity_map: HashMap<u64, Entity> =
            entities.into_iter().map(|e| (e.id, e)).collect();

        Ok(Self {
            concepts,
            relationships,
            conventions: analysis.conventions,
            embeddings,
            signatures: analysis.signatures,
            classes: analysis.classes,
            call_sites: analysis.call_sites,
            entities: entity_map,
            cluster_centroids: HashMap::new(),

            logic_index: LogicIndex::empty(),
            logic_clusters: Vec::new(),
            centrality: HashMap::new(),
            logic_concept_overlaps: Vec::new(),
            nesting_trees: analysis.nesting_trees,
            imports,
        })
    }
}
