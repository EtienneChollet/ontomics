// L4 Logic Layer tests — TDD definitions.
//
// These tests define the expected behavior of the L4 logic layer extension.
// They reference types and functions that DO NOT EXIST yet. They will compile
// once the L4 types/modules are added, and will FAIL until the implementation
// is correct.
//
// The implementation must match these expectations, not vice versa.

// ============================================================================
// 1. Parser body extraction
// ============================================================================

mod body_extraction {
    use ontomics::parser::{self, parse_content_with};
    use std::path::Path;

    fn parse_py(source: &str) -> ontomics::types::ParseResult {
        let parser = parser::python_parser();
        let path = Path::new("test.py");
        parse_content_with(source, path, &parser).expect("parse failed")
    }

    #[test]
    fn python_function_body_populated() {
        let source = "\
def compute_loss(pred, target):
    diff = pred - target
    return diff.pow(2).mean()
";
        let result = parse_py(source);
        assert_eq!(result.signatures.len(), 1);
        let sig = &result.signatures[0];
        let body = sig.body.as_ref().expect("body should be present");
        assert!(
            body.body_text.contains("diff = pred - target"),
            "body should contain the assignment: {:?}",
            body.body_text,
        );
        assert!(
            body.body_text.contains("return diff.pow(2).mean()"),
            "body should contain the return statement: {:?}",
            body.body_text,
        );
        assert!(body.scope.is_none(), "top-level function has no scope");
    }

    #[test]
    fn python_method_body_has_class_scope() {
        let source = "\
class MyModel:
    def forward(self, x):
        out = self.encoder(x)
        return self.decoder(out)
";
        let result = parse_py(source);
        let forward_sig = result
            .signatures
            .iter()
            .find(|s| s.name == "forward")
            .expect("forward signature not found");
        let body = forward_sig.body.as_ref().expect("body should be present");
        assert_eq!(
            body.scope.as_deref(),
            Some("MyModel"),
            "method body scope should be the class name",
        );
        assert!(body.body_text.contains("self.encoder(x)"));
    }

    #[test]
    fn one_liner_function_body() {
        let source = "def identity(x): return x\n";
        let result = parse_py(source);
        assert_eq!(result.signatures.len(), 1);
        let body = result.signatures[0]
            .body
            .as_ref()
            .expect("one-liner body should be present");
        assert!(
            body.body_text.contains("return x"),
            "one-liner body should contain 'return x': {:?}",
            body.body_text,
        );
    }

    #[test]
    fn class_with_no_methods_has_no_bodies() {
        let source = "\
class Config:
    name = 'default'
    value = 42
";
        let result = parse_py(source);
        // No function signatures at all, so no bodies.
        assert!(
            result.signatures.is_empty(),
            "class with no methods should produce no signatures",
        );
    }

    #[test]
    fn decorated_function_body_starts_after_def() {
        let source = "\
@torch.no_grad()
def predict(model, x):
    return model(x)
";
        let result = parse_py(source);
        assert_eq!(result.signatures.len(), 1);
        let body = result.signatures[0]
            .body
            .as_ref()
            .expect("body should be present");
        assert!(
            !body.body_text.contains("@torch.no_grad"),
            "body should not contain the decorator: {:?}",
            body.body_text,
        );
        assert!(body.body_text.contains("return model(x)"));
    }

    #[test]
    fn body_start_end_lines_accurate() {
        let source = "\
def foo():
    a = 1
    b = 2
    return a + b
";
        let result = parse_py(source);
        let body = result.signatures[0]
            .body
            .as_ref()
            .expect("body should be present");
        // The def is line 1. Body starts at line 2 (a = 1) and ends at line 4
        // (return a + b). Lines are 1-indexed.
        assert_eq!(body.start_line, 2, "body should start at line 2");
        assert_eq!(body.end_line, 4, "body should end at line 4");
    }

    #[test]
    fn body_entity_name_matches_function_name() {
        let source = "\
def spatial_transform(src, trf):
    return grid_sample(src, trf)
";
        let result = parse_py(source);
        let body = result.signatures[0]
            .body
            .as_ref()
            .expect("body should be present");
        assert_eq!(body.entity_name, "spatial_transform");
    }

    #[test]
    fn body_file_path_correct() {
        let source = "def f(): return 1\n";
        let parser = parser::python_parser();
        let path = Path::new("my/module.py");
        let result = parse_content_with(source, path, &parser).expect("parse failed");
        let body = result.signatures[0]
            .body
            .as_ref()
            .expect("body should be present");
        assert_eq!(body.file, Path::new("my/module.py"));
    }

    #[test]
    fn multiline_method_body_complete() {
        let source = "\
class Loss:
    def forward(self, pred, target):
        if self.reduction == 'mean':
            diff = (pred - target).pow(2)
            return diff.mean()
        elif self.reduction == 'sum':
            diff = (pred - target).pow(2)
            return diff.sum()
        else:
            return (pred - target).pow(2)
";
        let result = parse_py(source);
        let forward_sig = result
            .signatures
            .iter()
            .find(|s| s.name == "forward")
            .expect("forward not found");
        let body = forward_sig.body.as_ref().expect("body should be present");
        // Body should contain the if/elif/else structure
        assert!(body.body_text.contains("if self.reduction"));
        assert!(body.body_text.contains("elif self.reduction"));
        assert!(body.body_text.contains("else:"));
    }
}

// ============================================================================
// 3. Logic embeddings
// ============================================================================
// (section 2 — pseudocode tests — deleted: pipeline removed in v0.3.0)

mod logic_embeddings {
    use ontomics::logic::LogicIndex;

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    #[test]
    fn empty_logic_index_returns_no_results() {
        let index = LogicIndex::empty();
        let results = index.find_similar(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    #[ignore] // requires model download
    fn embed_body_correct_dimensionality() {
        let mut index = LogicIndex::new(None).expect("failed to load model");
        let text = "\
if condition:
    loss = compute_loss(pred, target)
    return loss
else:
    return 0
";
        let vector = index
            .embed_body(1, text)
            .expect("embedding failed");
        assert_eq!(
            vector.len(),
            768,
            "CodeRankEmbed produces 768-dimensional vectors",
        );
    }

    #[test]
    #[ignore] // requires model download
    fn similar_bodies_high_similarity() {
        let mut index = LogicIndex::new(None).expect("failed to load model");

        let loss_a = "\
diff = pred - target
squared = diff ** 2
return squared.mean()
";
        let loss_b = "\
error = prediction - ground_truth
sq_error = error ** 2
return sq_error.mean()
";
        let vec_a = index.embed_body(1, loss_a).expect("embed a failed");
        let vec_b = index.embed_body(2, loss_b).expect("embed b failed");

        let sim = cosine_similarity(&vec_a, &vec_b);
        assert!(
            sim > 0.5,
            "similar loss bodies should have similarity > 0.5, got {sim}",
        );
    }

    #[test]
    #[ignore] // requires model download
    fn unrelated_bodies_low_similarity() {
        let mut index = LogicIndex::new(None).expect("failed to load model");

        let loss_code = "\
diff = pred - target
return (diff ** 2).mean()
";
        let loader_code = "\
with open(path) as f:
    for line in f:
        parsed = parse(line)
        data.append(parsed)
return data
";
        let vec_a = index.embed_body(1, loss_code).expect("embed a");
        let vec_b = index.embed_body(2, loader_code).expect("embed b");

        let sim = cosine_similarity(&vec_a, &vec_b);
        assert!(
            sim < 0.3,
            "unrelated bodies should have similarity < 0.3, got {sim}",
        );
    }

    #[test]
    #[ignore] // requires model download
    fn find_similar_ranked_by_descending_similarity() {
        let mut index = LogicIndex::new(None).expect("failed to load model");

        // Embed 3 code snippets with varying relatedness
        let loss_mse = "diff = pred - target\nreturn (diff ** 2).mean()";
        let loss_dice = "inter = (pred * target).sum()\nreturn 1 - inter / union";
        let loader = "with open(path) as f:\n    for item in data:\n        process(item)\nreturn results";

        index.embed_body(10, loss_mse).expect("embed mse");
        index.embed_body(20, loss_dice).expect("embed dice");
        index.embed_body(30, loader).expect("embed loader");

        // Query with the MSE loss vector
        let query = index
            .find_similar_to_entity(10, 3);

        // The query entity itself should be excluded
        assert!(
            query.iter().all(|(id, _)| *id != 10),
            "find_similar_to_entity should exclude the query entity",
        );

        // Results should be ranked by descending similarity
        for pair in query.windows(2) {
            assert!(
                pair[0].1 >= pair[1].1,
                "results should be ranked by descending similarity: {} >= {}",
                pair[0].1,
                pair[1].1,
            );
        }

        // The dice loss should be more similar to MSE loss than the loader
        if query.len() >= 2 {
            let dice_entry = query.iter().find(|(id, _)| *id == 20);
            let loader_entry = query.iter().find(|(id, _)| *id == 30);
            if let (Some(d), Some(l)) = (dice_entry, loader_entry) {
                assert!(
                    d.1 > l.1,
                    "dice loss should be more similar to MSE than loader: {} > {}",
                    d.1, l.1,
                );
            }
        }
    }

    #[test]
    #[ignore] // requires model download
    fn find_similar_to_entity_excludes_self() {
        let mut index = LogicIndex::new(None).expect("failed to load model");
        index
            .embed_body(1, "return compute(x)")
            .expect("embed");
        index
            .embed_body(2, "return transform(y)")
            .expect("embed");

        let results = index
            .find_similar_to_entity(1, 5);
        assert!(
            results.iter().all(|(id, _)| *id != 1),
            "query entity should not appear in its own results",
        );
    }
}

// ============================================================================
// 4. Logic clustering
// ============================================================================

mod logic_clustering {
    use ontomics::logic::{cluster_logic, label_clusters, LogicIndex};
    use ontomics::types::{CallSite, LogicCluster};
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[test]
    fn empty_input_no_clusters() {
        let index = LogicIndex::empty();
        let clusters = cluster_logic(&index, &[], 0.30);
        assert!(clusters.is_empty(), "empty input should produce no clusters");
    }

    #[test]
    fn single_entity_singleton_cluster() {
        let mut index = LogicIndex::empty();
        index.insert_vector(1, vec![1.0, 0.0, 0.0]);
        let clusters = cluster_logic(&index, &[1], 0.30);
        assert_eq!(clusters.len(), 1, "single entity should produce 1 cluster");
        assert_eq!(clusters[0].entity_ids.len(), 1);
        assert_eq!(clusters[0].entity_ids[0], 1);
    }

    #[test]
    fn identical_vectors_same_cluster() {
        let mut index = LogicIndex::empty();
        index.insert_vector(1, vec![1.0, 0.0, 0.0]);
        index.insert_vector(2, vec![1.0, 0.0, 0.0]);
        let clusters = cluster_logic(&index, &[1, 2], 0.30);

        assert_eq!(clusters.len(), 1, "identical vectors should form 1 cluster");
        assert_eq!(clusters[0].entity_ids.len(), 2);
    }

    #[test]
    fn unrelated_vectors_different_clusters() {
        let mut index = LogicIndex::empty();
        index.insert_vector(1, vec![1.0, 0.0, 0.0]);
        index.insert_vector(2, vec![0.0, 1.0, 0.0]);
        index.insert_vector(3, vec![0.0, 0.0, 1.0]);
        let clusters = cluster_logic(&index, &[1, 2, 3], 0.30);

        assert_eq!(
            clusters.len(),
            3,
            "orthogonal vectors should form 3 separate clusters",
        );
    }

    #[test]
    fn label_clusters_common_callee() {
        let mut clusters = vec![LogicCluster {
            id: 0,
            entity_ids: vec![1, 2],
            centroid: vec![1.0, 0.0, 0.0],
            behavioral_label: None,
        }];

        // Both entities call mse_loss
        let call_sites = [
            CallSite {
                caller_scope: Some("fn_a".into()),
                callee: "mse_loss".into(),
                file: PathBuf::from("a.py"),
                line: 10,
            },
            CallSite {
                caller_scope: Some("fn_b".into()),
                callee: "mse_loss".into(),
                file: PathBuf::from("b.py"),
                line: 20,
            },
        ];

        let mut entity_cs: HashMap<u64, Vec<&CallSite>> = HashMap::new();
        entity_cs.entry(1).or_default().push(&call_sites[0]);
        entity_cs.entry(2).or_default().push(&call_sites[1]);

        label_clusters(&mut clusters, &entity_cs);

        let label = clusters[0]
            .behavioral_label
            .as_ref()
            .expect("cluster should have a label");
        assert!(
            label.to_lowercase().contains("mse"),
            "label should contain 'mse' since both entities call mse_loss: {:?}",
            label,
        );
    }
}

// ============================================================================
// 5. Centrality
// ============================================================================

mod centrality {
    use ontomics::centrality::compute_centrality;
    use ontomics::types::{Entity, EntityKind, Relationship, RelationshipKind};
    use std::collections::HashMap;
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

    fn make_uses_edge(source: u64, target: u64) -> Relationship {
        Relationship {
            source,
            target,
            kind: RelationshipKind::Uses,
            weight: 1.0,
        }
    }

    fn make_inherits_edge(source: u64, target: u64) -> Relationship {
        Relationship {
            source,
            target,
            kind: RelationshipKind::InheritsFrom,
            weight: 1.0,
        }
    }

    #[test]
    fn empty_graph_empty_result() {
        let entities = HashMap::new();
        let relationships = vec![];
        let scores = compute_centrality(&entities, &relationships, 0.85, 50);
        assert!(scores.is_empty());
    }

    #[test]
    fn single_entity_pagerank_one() {
        let mut entities = HashMap::new();
        entities.insert(1, make_entity(1, "sole_function"));
        let scores = compute_centrality(&entities, &[], 0.85, 50);

        assert_eq!(scores.len(), 1);
        let score = &scores[&1];
        assert!(
            (score.pagerank - 1.0).abs() < 0.01,
            "single entity should have pagerank ~1.0, got {}",
            score.pagerank,
        );
        assert_eq!(score.in_degree, 0);
        assert_eq!(score.out_degree, 0);
    }

    #[test]
    fn linear_chain_degree_correctness() {
        // A -> B -> C
        let mut entities = HashMap::new();
        entities.insert(1, make_entity(1, "A"));
        entities.insert(2, make_entity(2, "B"));
        entities.insert(3, make_entity(3, "C"));

        let rels = vec![make_uses_edge(1, 2), make_uses_edge(2, 3)];

        let scores = compute_centrality(&entities, &rels, 0.85, 50);

        // A: out=1, in=0
        assert_eq!(scores[&1].out_degree, 1);
        assert_eq!(scores[&1].in_degree, 0);
        // B: out=1, in=1
        assert_eq!(scores[&2].out_degree, 1);
        assert_eq!(scores[&2].in_degree, 1);
        // C: out=0, in=1 -> highest in_degree
        assert_eq!(scores[&3].out_degree, 0);
        assert_eq!(scores[&3].in_degree, 1);
    }

    #[test]
    fn star_pattern_hub_highest_pagerank() {
        // A, B, C all -> D
        let mut entities = HashMap::new();
        entities.insert(1, make_entity(1, "A"));
        entities.insert(2, make_entity(2, "B"));
        entities.insert(3, make_entity(3, "C"));
        entities.insert(4, make_entity(4, "D"));

        let rels = vec![
            make_uses_edge(1, 4),
            make_uses_edge(2, 4),
            make_uses_edge(3, 4),
        ];

        let scores = compute_centrality(&entities, &rels, 0.85, 50);

        // D should have the highest pagerank
        let d_rank = scores[&4].pagerank;
        for (&id, score) in &scores {
            if id != 4 {
                assert!(
                    d_rank > score.pagerank,
                    "D (pagerank={d_rank}) should have higher pagerank than entity {id} (pagerank={})",
                    score.pagerank,
                );
            }
        }
    }

    #[test]
    fn disconnected_components_independent() {
        // Component 1: A -> B
        // Component 2: C -> D
        let mut entities = HashMap::new();
        entities.insert(1, make_entity(1, "A"));
        entities.insert(2, make_entity(2, "B"));
        entities.insert(3, make_entity(3, "C"));
        entities.insert(4, make_entity(4, "D"));

        let rels = vec![make_uses_edge(1, 2), make_uses_edge(3, 4)];

        let scores = compute_centrality(&entities, &rels, 0.85, 50);

        // All 4 entities should have scores
        assert_eq!(scores.len(), 4);

        // B and D should have similar pagerank (symmetric roles)
        let diff = (scores[&2].pagerank - scores[&4].pagerank).abs();
        assert!(
            diff < 0.01,
            "symmetric nodes in disconnected components should have similar pagerank: B={}, D={}",
            scores[&2].pagerank,
            scores[&4].pagerank,
        );
    }

    #[test]
    fn cycle_similar_pagerank() {
        // A -> B -> C -> A
        let mut entities = HashMap::new();
        entities.insert(1, make_entity(1, "A"));
        entities.insert(2, make_entity(2, "B"));
        entities.insert(3, make_entity(3, "C"));

        let rels = vec![
            make_uses_edge(1, 2),
            make_uses_edge(2, 3),
            make_uses_edge(3, 1),
        ];

        let scores = compute_centrality(&entities, &rels, 0.85, 50);

        // All 3 entities are symmetric in the cycle, so pagerank should be ~equal
        let ranks: Vec<f32> = scores.values().map(|s| s.pagerank).collect();
        let max = ranks.iter().cloned().fold(f32::MIN, f32::max);
        let min = ranks.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            (max - min) < 0.05,
            "cycle entities should have similar pagerank: min={min}, max={max}",
        );
    }

    #[test]
    fn inherits_from_edges_count() {
        // ChildA inherits from Parent, ChildB inherits from Parent
        let mut entities = HashMap::new();
        entities.insert(1, make_entity(1, "Parent"));
        entities.insert(2, make_entity(2, "ChildA"));
        entities.insert(3, make_entity(3, "ChildB"));

        let rels = vec![
            make_inherits_edge(2, 1),
            make_inherits_edge(3, 1),
        ];

        let scores = compute_centrality(&entities, &rels, 0.85, 50);

        assert_eq!(scores[&1].in_degree, 2, "Parent should have in_degree=2");
        assert!(
            scores[&1].pagerank > scores[&2].pagerank,
            "Parent should have higher pagerank than children",
        );
    }
}

// ============================================================================
// 6. Integration tests against real codebases
// ============================================================================

mod integration {
    use std::sync::Arc;

    /// Build a full L4 graph for a codebase. This reuses the testbed
    /// infrastructure for parsing + L1-L3, then runs the L4 pipeline
    /// (body extraction, logic embeddings, clustering, centrality) on top.
    ///
    /// Returns None if the codebase is not available locally.
    fn build_l4_graph(
        repo_path: &str,
    ) -> Option<Arc<ontomics::graph::ConceptGraph>> {
        use ontomics::analyzer;
        use ontomics::config::{Config, Language};
        use ontomics::embeddings::EmbeddingIndex;
        use ontomics::entity;
        use ontomics::graph::ConceptGraph;
        use ontomics::logic::LogicIndex;
        use ontomics::parser::{self, ParseOptions};
        use ontomics::centrality;
        use ontomics::types;
        use std::collections::HashMap;
        use std::path::Path;

        let repo = Path::new(repo_path);
        if !repo.exists() {
            eprintln!("SKIP: {repo_path} not found");
            return None;
        }

        let config = Config::load(repo).unwrap_or_default();
        let (language, _) = Language::detect(repo);

        let lang: Arc<dyn parser::LanguageParser> = match &language {
            Language::Python => Arc::new(parser::python_parser()),
            Language::TypeScript => Arc::new(parser::typescript_parser()),
            Language::JavaScript => Arc::new(parser::javascript_parser()),
            Language::Rust => Arc::new(parser::rust_parser()),
            Language::Auto => {
                let (detected, _) = Language::detect(repo);
                match detected {
                    Language::TypeScript => Arc::new(parser::typescript_parser()),
                    Language::JavaScript => Arc::new(parser::javascript_parser()),
                    Language::Rust => Arc::new(parser::rust_parser()),
                    _ => Arc::new(parser::python_parser()),
                }
            }
        };

        let language_name = language.name();

        let mut index_config = config.index.clone();
        index_config.resolve_for_language(&language);

        let parse_opts = ParseOptions {
            include: index_config.include.clone(),
            exclude: index_config.exclude.clone(),
            respect_gitignore: index_config.respect_gitignore,
        };
        let parse_results = parser::parse_directory_with(repo, &parse_opts, &*lang)
            .expect("parsing failed");

        let analysis_params = analyzer::AnalysisParams {
            min_frequency: config.index.min_frequency,
            tfidf_threshold: config.analysis.domain_specificity_threshold,
            convention_threshold: config.analysis.convention_threshold,
            language: language_name.to_string(),
        };
        let analysis = analyzer::analyze(&parse_results, &analysis_params)
            .expect("analysis failed");

        let concept_map: HashMap<u64, types::Concept> = analysis
            .concepts
            .iter()
            .map(|c| (c.id, c.clone()))
            .collect();
        let (built_entities, entity_rels) = entity::build_entities(
            &analysis.signatures,
            &analysis.classes,
            &analysis.call_sites,
            &concept_map,
        );

        let mut embedding_index = if config.embeddings.enabled {
            EmbeddingIndex::new(None).unwrap_or_else(|_| EmbeddingIndex::empty())
        } else {
            EmbeddingIndex::empty()
        };

        if config.embeddings.enabled {
            let _ = embedding_index.embed_concepts_batch(&analysis.concepts);
        }

        let classes_for_roles = analysis.classes.clone();
        let mut graph = ConceptGraph::build_with_entities(
            analysis,
            embedding_index,
            built_entities,
            entity_rels,
        )
        .expect("graph build failed");

        if config.embeddings.enabled {
            graph.cluster_and_add_similarity_edges(config.embeddings.similarity_threshold);
        }
        graph.add_abbreviation_edges();
        graph.add_contrastive_edges();
        graph.detect_subconcepts();

        let mut ents: Vec<types::Entity> = graph.entities.values().cloned().collect();
        entity::infer_semantic_roles(
            &mut ents,
            &classes_for_roles,
            &graph.conventions,
            &graph.concepts,
        );
        for ent in ents {
            graph.entities.insert(ent.id, ent);
        }

        // --- L4 pipeline ---

        // Logic embeddings: embed raw function bodies
        let mut logic_index = LogicIndex::new(None)
            .unwrap_or_else(|_| LogicIndex::empty());
        let embed_items: Vec<(u64, String)> = graph.signatures.iter()
            .filter_map(|sig| {
                let body = sig.body.as_ref()?;
                let entity = graph.entities.values().find(|e| {
                    e.name == sig.name && e.file == sig.file
                })?;
                Some((entity.id, body.body_text.clone()))
            })
            .collect();
        if !embed_items.is_empty() {
            let _ = logic_index.embed_batch(embed_items);
        }
        graph.logic_index = logic_index;

        // Logic clustering
        let entity_ids: Vec<u64> = graph
            .logic_index
            .vector_ids()
            .into_iter()
            .collect();
        let mut logic_clusters =
            ontomics::logic::cluster_logic(&graph.logic_index, &entity_ids, 0.30);

        // Build entity → call_sites map for label_clusters
        let mut entity_call_sites: HashMap<u64, Vec<&types::CallSite>> = HashMap::new();
        for cs in &graph.call_sites {
            if let Some(scope) = &cs.caller_scope {
                if let Some(entity) = graph.entities.values().find(|e| {
                    e.name == *scope && e.file == cs.file
                }) {
                    entity_call_sites.entry(entity.id).or_default().push(cs);
                }
            }
        }
        ontomics::logic::label_clusters(&mut logic_clusters, &entity_call_sites);
        graph.logic_clusters = logic_clusters;

        // Centrality
        let centrality_scores = centrality::compute_centrality(
            &graph.entities,
            &graph.relationships,
            0.85,
            50,
        );
        graph.centrality = centrality_scores;

        Some(Arc::new(graph))
    }

    // -- Voxelmorph integration tests --

    #[test]
    #[ignore] // requires voxelmorph codebase + model download
    fn voxelmorph_body_for_spatial_transformer() {
        let graph = match build_l4_graph("/home/eti/projects/voxelmorph") {
            Some(g) => g,
            None => return,
        };

        let desc = graph
            .describe_logic("SpatialTransformer")
            .expect("describe_logic should find SpatialTransformer");

        assert!(
            !desc.body_text.is_empty(),
            "SpatialTransformer should have non-empty body text",
        );
        // SpatialTransformer.forward() fundamentally does grid generation
        // and grid sampling. The body should reference these operations.
        let text_lower = desc.body_text.to_lowercase();
        assert!(
            text_lower.contains("grid") || text_lower.contains("sample"),
            "SpatialTransformer body should contain grid/sample operations: {:?}",
            desc.body_text,
        );
    }

    #[test]
    #[ignore] // requires voxelmorph codebase + model download
    fn voxelmorph_loss_functions_cluster_together() {
        let graph = match build_l4_graph("/home/eti/projects/voxelmorph") {
            Some(g) => g,
            None => return,
        };

        // Find the logic cluster IDs for MSE and Dice
        let mse_cluster = find_entity_logic_cluster(&graph, "MSE");
        let dice_cluster = find_entity_logic_cluster(&graph, "Dice");

        if let (Some(mse_c), Some(dice_c)) = (mse_cluster, dice_cluster) {
            assert_eq!(
                mse_c, dice_c,
                "MSE and Dice should be in the same logic cluster (both compute loss metrics)",
            );
        }
        // If one of them doesn't have a body (e.g., trivial), that's
        // acceptable -- the test verifies the clustering when both are present.
    }

    #[test]
    #[ignore] // requires voxelmorph codebase + model download
    fn voxelmorph_transform_vs_loss_different_clusters() {
        let graph = match build_l4_graph("/home/eti/projects/voxelmorph") {
            Some(g) => g,
            None => return,
        };

        let transform_cluster =
            find_entity_logic_cluster(&graph, "SpatialTransformer");
        let loss_cluster = find_entity_logic_cluster(&graph, "MSE");

        if let (Some(tc), Some(lc)) = (transform_cluster, loss_cluster) {
            assert_ne!(
                tc, lc,
                "Transform entities and loss entities should be in different logic clusters",
            );
        }
    }

    #[test]
    #[ignore] // requires voxelmorph codebase + model download
    fn voxelmorph_core_network_high_centrality() {
        let graph = match build_l4_graph("/home/eti/projects/voxelmorph") {
            Some(g) => g,
            None => return,
        };

        // VxmDense is the main registration network -- it wires together
        // encoder, decoder, spatial transformer, etc. It should have high
        // centrality.
        let mut ranked: Vec<_> = graph.centrality.iter().collect();
        ranked.sort_by(|a, b| {
            b.1.pagerank
                .partial_cmp(&a.1.pagerank)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find VxmDense in the top 10 by pagerank
        let top_10_ids: Vec<u64> = ranked.iter().take(10).map(|(&id, _)| id).collect();
        let vxm_entity = graph
            .entities
            .values()
            .find(|e| e.name == "VxmDense");

        if let Some(vxm) = vxm_entity {
            assert!(
                top_10_ids.contains(&vxm.id),
                "VxmDense should be in top 10 by pagerank. Top 10: {:?}",
                top_10_ids
                    .iter()
                    .filter_map(|id| graph.entities.get(id).map(|e| &e.name))
                    .collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    #[ignore] // requires voxelmorph codebase + model download
    fn voxelmorph_compact_context_under_budget() {
        let graph = match build_l4_graph("/home/eti/projects/voxelmorph") {
            Some(g) => g,
            None => return,
        };

        let ctx = graph
            .compact_context("transform", 500)
            .expect("compact_context should resolve 'transform'");

        assert!(
            ctx.token_estimate <= 500,
            "compact_context should respect the 500 token budget: got {}",
            ctx.token_estimate,
        );
        assert!(
            !ctx.text.is_empty(),
            "compact_context should produce non-empty text",
        );
        // Should contain structural info (entity names, kind)
        // and behavioral info (body text or logic cluster)
        let text_lower = ctx.text.to_lowercase();
        assert!(
            text_lower.contains("transform") || text_lower.contains("spatial"),
            "compact_context for 'transform' should mention transform/spatial: {:?}",
            ctx.text,
        );
    }

    #[test]
    #[ignore] // requires voxelmorph codebase + model download
    fn voxelmorph_describe_logic_returns_centrality() {
        let graph = match build_l4_graph("/home/eti/projects/voxelmorph") {
            Some(g) => g,
            None => return,
        };

        let desc = graph.describe_logic("SpatialTransformer");
        if let Some(d) = desc {
            // Centrality should be populated
            assert!(
                d.centrality.pagerank > 0.0,
                "SpatialTransformer should have positive pagerank",
            );
        }
    }

    #[test]
    #[ignore] // requires voxelmorph codebase + model download
    fn voxelmorph_find_similar_logic_returns_related() {
        let graph = match build_l4_graph("/home/eti/projects/voxelmorph") {
            Some(g) => g,
            None => return,
        };

        let result = graph.find_similar_logic("SpatialTransformer", 5);
        if let Some(r) = result {
            assert!(
                !r.similar.is_empty(),
                "SpatialTransformer should have at least one similar entity by logic",
            );
            // Similarity scores should be in (0, 1] range
            for (_, sim) in &r.similar {
                assert!(
                    *sim > 0.0 && *sim <= 1.0,
                    "similarity should be in (0, 1], got {}",
                    sim,
                );
            }
        }
    }

    // -- Neurite integration tests --

    #[test]
    #[ignore] // requires neurite codebase + model download
    fn neurite_functional_and_class_wrappers_similar_logic() {
        let graph = match build_l4_graph("/home/eti/projects/neurite") {
            Some(g) => g,
            None => return,
        };

        // neurite has functional implementations (e.g., `dice` function)
        // and class wrappers (e.g., `Dice` class with forward() calling
        // the functional). These should have similar logic embeddings.
        let dice_fn = graph
            .entities
            .values()
            .find(|e| e.name == "dice" && e.kind == ontomics::types::EntityKind::Function);
        let dice_cls = graph
            .entities
            .values()
            .find(|e| e.name == "Dice" && e.kind == ontomics::types::EntityKind::Class);

        if let (Some(func), Some(cls)) = (dice_fn, dice_cls) {
            // Check if both have logic embeddings
            let vec_fn = graph.logic_index.get_vector(func.id);
            let vec_cls = graph.logic_index.get_vector(cls.id);

            if let (Some(vf), Some(vc)) = (vec_fn, vec_cls) {
                let sim = cosine_sim(vf, vc);
                assert!(
                    sim > 0.3,
                    "functional `dice` and class `Dice` should have similar logic (sim > 0.3): got {}",
                    sim,
                );
            }
        }
    }

    #[test]
    #[ignore] // requires neurite codebase + model download
    fn neurite_utility_functions_high_centrality() {
        let graph = match build_l4_graph("/home/eti/projects/neurite") {
            Some(g) => g,
            None => return,
        };

        // neurite has utility functions used broadly across the codebase.
        // At least some entities should have in_degree > 0.
        let entities_with_incoming: Vec<_> = graph
            .centrality
            .values()
            .filter(|s| s.in_degree > 0)
            .collect();
        assert!(
            !entities_with_incoming.is_empty(),
            "neurite should have entities with positive in_degree (depended on by others)",
        );
    }

    // -- Helper functions --

    fn find_entity_logic_cluster(
        graph: &ontomics::graph::ConceptGraph,
        entity_name: &str,
    ) -> Option<usize> {
        let entity = graph.entities.values().find(|e| e.name == entity_name)?;
        graph
            .logic_clusters
            .iter()
            .find(|c| c.entity_ids.contains(&entity.id))
            .map(|c| c.id)
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }
}

// ============================================================================
// 7. MCP tool tests
// ============================================================================

mod mcp_tools {
    use ontomics::graph::ConceptGraph;
    use ontomics::types::{
        CentralityScore, Entity, EntityKind, FunctionBody, LogicCluster,
        Relationship, RelationshipKind, Signature,
    };
    use std::path::PathBuf;

    /// Build a minimal graph with L4 data for tool testing.
    fn build_test_graph() -> ConceptGraph {
        use ontomics::types::AnalysisResult;

        let sig_loss = Signature {
            name: "compute_loss".to_string(),
            params: vec![],
            return_type: None,
            decorators: vec![],
            docstring_first_line: None,
            file: PathBuf::from("losses.py"),
            line: 10,
            scope: None,
            body: Some(FunctionBody {
                entity_name: "compute_loss".to_string(),
                scope: None,
                body_text: "diff = subtract(pred, target)\nsquared = pow(diff, 2)\nreturn mean(squared)".to_string(),
                language: "python".to_string(),
                file: PathBuf::from("losses.py"),
                start_line: 11,
                end_line: 13,
                was_truncated: false,
            }),
        };

        let analysis = AnalysisResult {
            concepts: vec![],
            conventions: vec![],
            co_occurrence_matrix: vec![],
            signatures: vec![sig_loss],
            classes: vec![],
            call_sites: vec![],
            nesting_trees: vec![],
        };
        let embeddings = ontomics::embeddings::EmbeddingIndex::empty();

        let entity_a = Entity {
            id: 100,
            name: "compute_loss".to_string(),
            kind: EntityKind::Function,
            concept_tags: vec![],
            semantic_role: "loss computation".to_string(),
            file: PathBuf::from("losses.py"),
            line: 10,
            signature_idx: None,
            class_info_idx: None,
        };
        let entity_b = Entity {
            id: 200,
            name: "SpatialTransformer".to_string(),
            kind: EntityKind::Class,
            concept_tags: vec![],
            semantic_role: "spatial transformation".to_string(),
            file: PathBuf::from("networks.py"),
            line: 50,
            signature_idx: None,
            class_info_idx: None,
        };

        let rels = vec![Relationship {
            source: 100,
            target: 200,
            kind: RelationshipKind::Uses,
            weight: 1.0,
        }];

        let mut graph = ConceptGraph::build_with_entities(
            analysis,
            embeddings,
            vec![entity_a, entity_b],
            rels,
        )
        .expect("graph build failed");

        graph.centrality.insert(
            100,
            CentralityScore {
                entity_id: 100,
                in_degree: 0,
                out_degree: 1,
                pagerank: 0.15,
            },
        );
        graph.centrality.insert(
            200,
            CentralityScore {
                entity_id: 200,
                in_degree: 1,
                out_degree: 0,
                pagerank: 0.85,
            },
        );

        graph.logic_clusters = vec![LogicCluster {
            id: 0,
            entity_ids: vec![100],
            centroid: vec![1.0, 0.0, 0.0],
            behavioral_label: Some("loss computation".to_string()),
        }];

        graph
    }

    #[test]
    fn describe_logic_returns_body_and_centrality() {
        let graph = build_test_graph();
        let desc = graph
            .describe_logic("compute_loss")
            .expect("describe_logic should find compute_loss");

        assert!(
            !desc.body_text.is_empty(),
            "body_text should not be empty",
        );
        assert!(
            desc.body_text.contains("subtract")
                || desc.body_text.contains("diff"),
            "body should reference loss operations: {:?}",
            desc.body_text,
        );
        assert_eq!(desc.centrality.entity_id, 100);
        assert_eq!(desc.centrality.out_degree, 1);
    }

    #[test]
    fn describe_logic_includes_logic_cluster() {
        let graph = build_test_graph();
        let desc = graph
            .describe_logic("compute_loss")
            .expect("describe_logic should find compute_loss");

        let cluster = desc
            .logic_cluster
            .expect("compute_loss should have a logic cluster");
        assert_eq!(cluster.id, 0);
        assert!(
            cluster
                .behavioral_label
                .as_ref()
                .map(|l| l.contains("loss"))
                .unwrap_or(false),
            "cluster label should contain 'loss': {:?}",
            cluster.behavioral_label,
        );
    }

    #[test]
    fn describe_logic_unknown_entity_returns_none() {
        let graph = build_test_graph();
        let result = graph.describe_logic("nonexistent_function");
        assert!(
            result.is_none(),
            "describe_logic for unknown entity should return None",
        );
    }

    #[test]
    fn describe_logic_entity_without_body() {
        let graph = build_test_graph();
        // SpatialTransformer has centrality but no signature/body in our test graph
        let desc = graph.describe_logic("SpatialTransformer");
        if let Some(d) = desc {
            assert!(
                d.body_text.is_empty(),
                "entity without body should have empty body_text",
            );
            // But centrality should still be populated
            assert_eq!(d.centrality.entity_id, 200);
            assert_eq!(d.centrality.in_degree, 1);
        }
    }

    #[test]
    fn compact_context_returns_text_and_estimate() {
        let graph = build_test_graph();
        let ctx = graph
            .compact_context("compute_loss", 500)
            .expect("compact_context should resolve 'compute_loss'");

        assert!(
            !ctx.text.is_empty(),
            "compact_context text should not be empty",
        );
        assert!(
            ctx.token_estimate > 0,
            "token estimate should be positive",
        );
        assert!(
            ctx.token_estimate <= 500,
            "token estimate should respect budget: got {}",
            ctx.token_estimate,
        );
    }

    #[test]
    fn compact_context_unknown_scope_returns_none() {
        let graph = build_test_graph();
        let result = graph.compact_context("nonexistent_concept", 500);
        assert!(
            result.is_none(),
            "compact_context for unknown scope should return None",
        );
    }

    #[test]
    fn compact_context_contains_structure_and_behavior() {
        let graph = build_test_graph();
        let ctx = graph
            .compact_context("compute_loss", 1000)
            .expect("compact_context should resolve");

        let text_lower = ctx.text.to_lowercase();
        assert!(
            text_lower.contains("compute_loss"),
            "compact context should mention entity name: {:?}",
            ctx.text,
        );
        // Should contain behavioral info (body text or description)
        assert!(
            text_lower.contains("subtract")
                || text_lower.contains("diff")
                || text_lower.contains("loss"),
            "compact context should contain behavioral info: {:?}",
            ctx.text,
        );
    }

    #[test]
    fn find_similar_logic_unknown_entity_returns_none() {
        let graph = build_test_graph();
        let result = graph.find_similar_logic("nonexistent", 5);
        assert!(
            result.is_none(),
            "find_similar_logic for unknown entity should return None",
        );
    }
}

// ============================================================================
// 8. ConceptGraph L4 field initialization
// ============================================================================

mod graph_l4_fields {
    use ontomics::graph::ConceptGraph;

    #[test]
    fn empty_graph_has_l4_fields() {
        let graph = ConceptGraph::empty();
        assert!(graph.logic_clusters.is_empty());
        assert!(graph.centrality.is_empty());
        assert_eq!(graph.logic_index.nb_vectors(), 0);
    }
}
