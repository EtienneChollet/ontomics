use super::ConceptGraph;
use crate::types::{
    CentralityScore, EntitySummary, LogicClusterSummary, LogicDescription,
    SimilarLogicResult,
};

impl ConceptGraph {
    /// L4: Describe the behavioral logic of an entity.
    pub fn describe_logic(&self, name: &str) -> Option<LogicDescription> {
        let name_lower = name.to_lowercase();
        let entity = self.entities.values().find(|e| {
            e.name.to_lowercase() == name_lower
        })?;

        let body_text = self.signatures.iter()
            .find(|sig| sig.name == entity.name && sig.file == entity.file)
            .and_then(|sig| sig.body.as_ref())
            .map(|b| b.body_text.clone())
            .unwrap_or_default();

        let cluster_summary = self.logic_clusters.iter()
            .find(|lc| lc.entity_ids.contains(&entity.id))
            .map(|lc| {
                let members: Vec<EntitySummary> = lc.entity_ids.iter()
                    .filter_map(|id| self.entities.get(id))
                    .map(Self::entity_summary)
                    .collect();
                LogicClusterSummary {
                    id: lc.id,
                    size: lc.entity_ids.len(),
                    behavioral_label: lc.behavioral_label.clone(),
                    members,
                }
            });

        let centrality = self.centrality.get(&entity.id)
            .cloned()
            .unwrap_or(CentralityScore {
                entity_id: entity.id,
                in_degree: 0,
                out_degree: 0,
                pagerank: 0.0,
            });

        Some(LogicDescription {
            entity: Self::entity_summary(entity),
            body_text,
            logic_cluster: cluster_summary,
            centrality,
        })
    }

    /// L4: Find entities with similar behavioral patterns.
    pub fn find_similar_logic(
        &self,
        name: &str,
        top_k: usize,
    ) -> Option<SimilarLogicResult> {
        let name_lower = name.to_lowercase();
        let entity = self.entities.values().find(|e| {
            e.name.to_lowercase() == name_lower
        })?;

        let similar_ids = self.logic_index
            .find_similar_to_entity(entity.id, top_k);

        let similar: Vec<(EntitySummary, f32)> = similar_ids.iter()
            .filter_map(|(id, score)| {
                self.entities.get(id)
                    .map(|e| (Self::entity_summary(e), *score))
            })
            .collect();

        Some(SimilarLogicResult {
            query_entity: Self::entity_summary(entity),
            similar,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::ConceptGraph;
    use crate::types::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn make_entity(id: u64, name: &str) -> Entity {
        Entity {
            id,
            name: name.to_string(),
            kind: EntityKind::Function,
            concept_tags: Vec::new(),
            semantic_role: "utility".to_string(),
            file: PathBuf::from("test.py"),
            line: 1,
            signature_idx: None,
            class_info_idx: None,
        }
    }

    fn make_sig_with_body(name: &str, body_text: &str) -> Signature {
        Signature {
            name: name.to_string(),
            params: vec![],
            return_type: None,
            decorators: vec![],
            docstring_first_line: None,
            file: PathBuf::from("test.py"),
            line: 1,
            scope: None,
            body: Some(FunctionBody {
                entity_name: name.to_string(),
                scope: None,
                body_text: body_text.to_string(),
                language: "python".to_string(),
                file: PathBuf::from("test.py"),
                start_line: 2,
                end_line: 5,
                was_truncated: false,
            }),
        }
    }

    fn make_graph_with_l4(
        entities: Vec<Entity>,
        signatures: Vec<Signature>,
        centrality: HashMap<u64, CentralityScore>,
    ) -> ConceptGraph {
        let mut graph = ConceptGraph::empty();
        for entity in entities {
            graph.entities.insert(entity.id, entity);
        }
        graph.signatures = signatures;
        graph.centrality = centrality;
        graph
    }

    #[test]
    fn test_describe_logic_basic_flow() {
        let entity = make_entity(42, "process_data");
        let sig = make_sig_with_body("process_data", "result = validate(data)\nreturn result");
        let centrality_score = CentralityScore {
            entity_id: 42, in_degree: 3, out_degree: 1, pagerank: 0.42,
        };
        let mut centrality = HashMap::new();
        centrality.insert(42, centrality_score);

        let graph = make_graph_with_l4(vec![entity], vec![sig], centrality);
        let result = graph.describe_logic("process_data");
        assert!(result.is_some());
        let desc = result.unwrap();
        assert_eq!(desc.entity.name, "process_data");
        assert!(!desc.body_text.is_empty());
        assert!((desc.centrality.pagerank - 0.42).abs() < 1e-6);
        assert_eq!(desc.centrality.in_degree, 3);
        assert_eq!(desc.centrality.out_degree, 1);
    }

    #[test]
    fn test_describe_logic_missing_body_returns_empty_text() {
        let entity = make_entity(10, "my_func");
        let graph = make_graph_with_l4(vec![entity], vec![], HashMap::new());
        let result = graph.describe_logic("my_func");
        assert!(result.is_some());
        let desc = result.unwrap();
        assert_eq!(desc.body_text, "");
        assert_eq!(desc.centrality.in_degree, 0);
        assert_eq!(desc.centrality.pagerank, 0.0);
    }

    #[test]
    fn test_describe_logic_not_found_returns_none() {
        let graph = ConceptGraph::empty();
        assert!(graph.describe_logic("nonexistent_entity").is_none());
    }

    #[test]
    fn test_find_similar_logic_basic() {
        let entities = vec![
            make_entity(1, "entity_a"),
            make_entity(2, "entity_b"),
            make_entity(3, "entity_c"),
        ];
        let mut graph = make_graph_with_l4(entities, vec![], HashMap::new());
        graph.logic_index.insert_vector(1, vec![1.0, 0.0, 0.0]);
        graph.logic_index.insert_vector(2, vec![0.9, 0.1, 0.0]);
        graph.logic_index.insert_vector(3, vec![0.0, 1.0, 0.0]);

        let result = graph.find_similar_logic("entity_a", 2);
        assert!(result.is_some());
        let sim = result.unwrap();
        assert_eq!(sim.query_entity.name, "entity_a");
        assert_eq!(sim.similar.len(), 2);
        assert_eq!(sim.similar[0].0.name, "entity_b");
        assert!(sim.similar[0].1 > sim.similar[1].1);
    }

    #[test]
    fn test_find_similar_logic_no_vector_returns_empty_similar() {
        let entity = make_entity(5, "bare_entity");
        let graph = make_graph_with_l4(vec![entity], vec![], HashMap::new());
        let result = graph.find_similar_logic("bare_entity", 5);
        assert!(result.is_some());
        assert!(result.unwrap().similar.is_empty());
    }

    #[test]
    fn test_find_similar_logic_not_found_returns_none() {
        let graph = ConceptGraph::empty();
        assert!(graph.find_similar_logic("ghost", 3).is_none());
    }

    #[test]
    fn test_l4_tools_return_none_on_empty_graph() {
        let graph = ConceptGraph::empty();
        assert!(graph.describe_logic("anything").is_none());
        assert!(graph.find_similar_logic("anything", 5).is_none());
    }
}
