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
    use ontomics::types::FunctionBody;
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
// 2. Pseudocode generation
// ============================================================================

mod pseudocode {
    use ontomics::pseudocode::{format_pseudocode, generate_pseudocode};
    use ontomics::parser;
    use ontomics::types::{FunctionBody, LoopKind, PseudocodeStep};
    use std::path::PathBuf;

    fn make_body(text: &str) -> FunctionBody {
        FunctionBody {
            entity_name: "test_func".to_string(),
            scope: None,
            body_text: text.to_string(),
            file: PathBuf::from("test.py"),
            start_line: 2,
            end_line: 2 + text.lines().count(),
        }
    }

    #[test]
    fn if_elif_else_produces_conditional() {
        let body = make_body(
            "\
    if x > 0:
        return x
    elif x == 0:
        return 0
    else:
        return -x
",
        );
        let py = parser::python_parser();
        let pseudo = generate_pseudocode(&body, &py, 30).expect("generation failed");

        // Should produce one Conditional step with 3 branches
        let conditionals: Vec<_> = pseudo
            .steps
            .iter()
            .filter(|s| matches!(s, PseudocodeStep::Conditional { .. }))
            .collect();
        assert_eq!(
            conditionals.len(),
            1,
            "expected one Conditional step, got {}",
            conditionals.len(),
        );
        if let PseudocodeStep::Conditional { branches } = &conditionals[0] {
            assert_eq!(branches.len(), 3, "expected 3 branches (if/elif/else)");
            // First branch has a condition
            assert!(
                branches[0].condition.is_some(),
                "if-branch must have a condition",
            );
            // Second branch (elif) has a condition
            assert!(
                branches[1].condition.is_some(),
                "elif-branch must have a condition",
            );
            // Third branch (else) has no condition
            assert!(
                branches[2].condition.is_none(),
                "else-branch must have condition=None",
            );
        }
    }

    #[test]
    fn for_loop_with_nested_call() {
        let body = make_body(
            "\
    for item in items:
        process(item)
",
        );
        let py = parser::python_parser();
        let pseudo = generate_pseudocode(&body, &py, 30).expect("generation failed");

        let loops: Vec<_> = pseudo
            .steps
            .iter()
            .filter(|s| matches!(s, PseudocodeStep::Loop { .. }))
            .collect();
        assert_eq!(loops.len(), 1, "expected one Loop step");
        if let PseudocodeStep::Loop { kind, body } = &loops[0] {
            assert!(
                matches!(kind, LoopKind::For { .. }),
                "expected For loop kind",
            );
            if let LoopKind::For { variable, iterable } = kind {
                assert_eq!(variable, "item");
                assert_eq!(iterable, "items");
            }
            // Body should contain a Call to process
            let has_call = body.iter().any(|s| {
                matches!(s, PseudocodeStep::Call { callee, .. } if callee == "process")
            });
            assert!(has_call, "for-loop body should contain Call to 'process'");
        }
    }

    #[test]
    fn while_loop() {
        let body = make_body(
            "\
    while not converged:
        step()
",
        );
        let py = parser::python_parser();
        let pseudo = generate_pseudocode(&body, &py, 30).expect("generation failed");

        let loops: Vec<_> = pseudo
            .steps
            .iter()
            .filter(|s| matches!(s, PseudocodeStep::Loop { .. }))
            .collect();
        assert_eq!(loops.len(), 1);
        if let PseudocodeStep::Loop { kind, .. } = &loops[0] {
            assert!(
                matches!(kind, LoopKind::While { .. }),
                "expected While loop kind",
            );
        }
    }

    #[test]
    fn return_statement() {
        let body = make_body("    return result\n");
        let py = parser::python_parser();
        let pseudo = generate_pseudocode(&body, &py, 30).expect("generation failed");

        let returns: Vec<_> = pseudo
            .steps
            .iter()
            .filter(|s| matches!(s, PseudocodeStep::Return { .. }))
            .collect();
        assert_eq!(returns.len(), 1);
        if let PseudocodeStep::Return { value } = &returns[0] {
            assert_eq!(value.as_deref(), Some("result"));
        }
    }

    #[test]
    fn assignment_from_function_call() {
        let body = make_body("    x = compute(a, b)\n");
        let py = parser::python_parser();
        let pseudo = generate_pseudocode(&body, &py, 30).expect("generation failed");

        let assigns: Vec<_> = pseudo
            .steps
            .iter()
            .filter(|s| matches!(s, PseudocodeStep::Assignment { .. }))
            .collect();
        assert_eq!(assigns.len(), 1);
        if let PseudocodeStep::Assignment { target, source } = &assigns[0] {
            assert_eq!(target, "x");
            assert!(
                source.contains("compute"),
                "assignment source should reference 'compute': {:?}",
                source,
            );
        }
    }

    #[test]
    fn nested_control_flow() {
        let body = make_body(
            "\
    for i in range(n):
        if i % 2 == 0:
            process(i)
",
        );
        let py = parser::python_parser();
        let pseudo = generate_pseudocode(&body, &py, 30).expect("generation failed");

        // Top level: one Loop
        assert_eq!(pseudo.steps.len(), 1);
        if let PseudocodeStep::Loop { body, .. } = &pseudo.steps[0] {
            // Inside the loop: one Conditional
            let has_conditional = body
                .iter()
                .any(|s| matches!(s, PseudocodeStep::Conditional { .. }));
            assert!(
                has_conditional,
                "loop body should contain a Conditional step",
            );
        } else {
            panic!("expected top-level Loop step");
        }
    }

    #[test]
    fn trivial_function_minimal_steps() {
        let body = make_body("    return self.x\n");
        let py = parser::python_parser();
        let pseudo = generate_pseudocode(&body, &py, 30).expect("generation failed");

        assert_eq!(
            pseudo.steps.len(),
            1,
            "trivial function should produce exactly 1 step",
        );
        assert!(matches!(&pseudo.steps[0], PseudocodeStep::Return { .. }));
    }

    #[test]
    fn format_pseudocode_readable() {
        let body = make_body(
            "\
    if x > 0:
        y = compute(x)
        return y
    else:
        return 0
",
        );
        let py = parser::python_parser();
        let pseudo = generate_pseudocode(&body, &py, 30).expect("generation failed");
        let text = format_pseudocode(&pseudo);

        assert!(!text.is_empty(), "formatted pseudocode should not be empty");
        // Should be human-readable with indentation
        assert!(
            text.contains("  "),
            "formatted pseudocode should use indentation",
        );
        // Should reference the key operations
        assert!(
            text.to_lowercase().contains("if") || text.to_lowercase().contains("conditional"),
            "formatted text should mention conditional: {:?}",
            text,
        );
    }

    #[test]
    fn truncation_at_max_lines() {
        // Build a body with many statements to trigger truncation
        let mut lines = Vec::new();
        for i in 0..50 {
            lines.push(format!("    x_{i} = compute_{i}()"));
        }
        let body_text = lines.join("\n") + "\n";
        let body = make_body(&body_text);
        let py = parser::python_parser();
        // max_lines = 10 should trigger truncation
        let pseudo = generate_pseudocode(&body, &py, 10).expect("generation failed");

        assert!(
            pseudo.steps.len() <= 10,
            "truncation should limit steps to max_lines={}: got {}",
            10,
            pseudo.steps.len(),
        );
    }

    #[test]
    fn expression_simplification_literals_replaced() {
        let body = make_body(
            "\
    x = 'hello world'
    y = 42
    z = compute(x, y)
",
        );
        let py = parser::python_parser();
        let pseudo = generate_pseudocode(&body, &py, 30).expect("generation failed");

        // String/number literals in assignments should be simplified
        let text = format_pseudocode(&pseudo);
        assert!(
            !text.contains("hello world"),
            "literal strings should be simplified away: {:?}",
            text,
        );
        // But function names should be kept
        assert!(
            text.contains("compute"),
            "function names should be preserved: {:?}",
            text,
        );
    }

    #[test]
    fn empty_function_body_produces_empty_steps() {
        // Python pass-only function
        let body = make_body("    pass\n");
        let py = parser::python_parser();
        let pseudo = generate_pseudocode(&body, &py, 30).expect("generation failed");

        assert!(
            pseudo.steps.is_empty(),
            "pass-only function should produce 0 steps: got {}",
            pseudo.steps.len(),
        );
    }

    #[test]
    fn body_hash_changes_with_content() {
        let body_a = make_body("    return 1\n");
        let body_b = make_body("    return 2\n");
        let py = parser::python_parser();
        let pseudo_a = generate_pseudocode(&body_a, &py, 30).expect("gen a failed");
        let pseudo_b = generate_pseudocode(&body_b, &py, 30).expect("gen b failed");

        assert_ne!(
            pseudo_a.body_hash, pseudo_b.body_hash,
            "different bodies should produce different hashes",
        );
    }
}

// ============================================================================
// 3. Logic embeddings
// ============================================================================

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
    fn embed_pseudocode_correct_dimensionality() {
        let mut index = LogicIndex::new(None).expect("failed to load model");
        let text = "\
IF condition:
  CALL compute_loss(pred, target)
  RETURN loss
ELSE:
  RETURN 0
";
        let vector = index
            .embed_pseudocode(1, text)
            .expect("embedding failed");
        assert_eq!(
            vector.len(),
            384,
            "bge-small-en-v1.5 produces 384-dimensional vectors",
        );
    }

    #[test]
    #[ignore] // requires model download
    fn similar_pseudocode_high_similarity() {
        let mut index = LogicIndex::new(None).expect("failed to load model");

        let loss_a = "\
ASSIGN diff = CALL subtract(pred, target)
ASSIGN squared = CALL pow(diff, 2)
RETURN CALL mean(squared)
";
        let loss_b = "\
ASSIGN error = CALL subtract(prediction, ground_truth)
ASSIGN sq_error = CALL square(error)
RETURN CALL reduce_mean(sq_error)
";
        let vec_a = index.embed_pseudocode(1, loss_a).expect("embed a failed");
        let vec_b = index.embed_pseudocode(2, loss_b).expect("embed b failed");

        let sim = cosine_similarity(&vec_a, &vec_b);
        assert!(
            sim > 0.5,
            "similar loss pseudocode should have similarity > 0.5, got {sim}",
        );
    }

    #[test]
    #[ignore] // requires model download
    fn unrelated_pseudocode_low_similarity() {
        let mut index = LogicIndex::new(None).expect("failed to load model");

        let loss_code = "\
ASSIGN diff = CALL subtract(pred, target)
RETURN CALL mean(CALL pow(diff, 2))
";
        let loader_code = "\
CALL open_file(path)
FOR line IN CALL read_lines(file):
  CALL parse(line)
  CALL append(data, parsed)
RETURN data
";
        let vec_a = index.embed_pseudocode(1, loss_code).expect("embed a");
        let vec_b = index.embed_pseudocode(2, loader_code).expect("embed b");

        let sim = cosine_similarity(&vec_a, &vec_b);
        assert!(
            sim < 0.3,
            "unrelated pseudocode should have similarity < 0.3, got {sim}",
        );
    }

    #[test]
    #[ignore] // requires model download
    fn find_similar_ranked_by_descending_similarity() {
        let mut index = LogicIndex::new(None).expect("failed to load model");

        // Embed 3 pseudocode snippets with varying relatedness
        let loss_mse = "ASSIGN diff = CALL subtract(pred, target)\nRETURN CALL mean(CALL pow(diff, 2))";
        let loss_dice = "ASSIGN inter = CALL sum(CALL multiply(pred, target))\nRETURN CALL subtract(1, CALL divide(inter, union))";
        let loader = "CALL open(path)\nFOR item IN data:\n  CALL process(item)\nRETURN results";

        index.embed_pseudocode(10, loss_mse).expect("embed mse");
        index.embed_pseudocode(20, loss_dice).expect("embed dice");
        index.embed_pseudocode(30, loader).expect("embed loader");

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
            .embed_pseudocode(1, "RETURN CALL compute(x)")
            .expect("embed");
        index
            .embed_pseudocode(2, "RETURN CALL transform(y)")
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
    use ontomics::types::{LogicCluster, Pseudocode, PseudocodeStep};
    use std::collections::HashMap;

    #[test]
    fn empty_input_no_clusters() {
        let index = LogicIndex::empty();
        let clusters = cluster_logic(&index, &[], 0.30);
        assert!(clusters.is_empty(), "empty input should produce no clusters");
    }

    #[test]
    fn single_entity_singleton_cluster() {
        let mut index = LogicIndex::empty();
        // Insert a synthetic vector directly
        index.insert_vector(1, vec![1.0, 0.0, 0.0]);
        let clusters = cluster_logic(&index, &[1], 0.30);
        assert_eq!(clusters.len(), 1, "single entity should produce 1 cluster");
        assert_eq!(clusters[0].entity_ids.len(), 1);
        assert_eq!(clusters[0].entity_ids[0], 1);
    }

    #[test]
    fn identical_vectors_same_cluster() {
        let mut index = LogicIndex::empty();
        // Two entities with identical vectors should cluster together
        index.insert_vector(1, vec![1.0, 0.0, 0.0]);
        index.insert_vector(2, vec![1.0, 0.0, 0.0]);
        let clusters = cluster_logic(&index, &[1, 2], 0.30);

        // Both should be in the same cluster
        assert_eq!(clusters.len(), 1, "identical vectors should form 1 cluster");
        assert_eq!(clusters[0].entity_ids.len(), 2);
    }

    #[test]
    fn unrelated_vectors_different_clusters() {
        let mut index = LogicIndex::empty();
        // Orthogonal vectors should not cluster
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
        let mut pseudocode = HashMap::new();
        pseudocode.insert(
            1,
            Pseudocode {
                entity_id: 1,
                steps: vec![
                    PseudocodeStep::Call {
                        callee: "mse_loss".to_string(),
                        args: vec!["pred".to_string(), "target".to_string()],
                    },
                    PseudocodeStep::Return {
                        value: Some("loss".to_string()),
                    },
                ],
                body_hash: 111,
                omitted_count: 0,
            },
        );
        pseudocode.insert(
            2,
            Pseudocode {
                entity_id: 2,
                steps: vec![
                    PseudocodeStep::Call {
                        callee: "mse_loss".to_string(),
                        args: vec!["output".to_string(), "label".to_string()],
                    },
                    PseudocodeStep::Return {
                        value: Some("result".to_string()),
                    },
                ],
                body_hash: 222,
                omitted_count: 0,
            },
        );

        label_clusters(&mut clusters, &pseudocode);

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
    /// (body extraction, pseudocode, logic embeddings, clustering,
    /// centrality) on top.
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
        use ontomics::pseudocode;
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

        // Generate pseudocode for all entities with bodies
        let parser_refs: Vec<(&dyn parser::LanguageParser, &str)> =
            vec![(&*lang as &dyn parser::LanguageParser, language_name)];
        let all_pseudocode = pseudocode::generate_all_pseudocode(
            &graph.entities,
            &graph.signatures,
            &parser_refs,
            30, // max_pseudocode_lines
        );
        graph.pseudocode = all_pseudocode;

        // Logic embeddings
        let mut logic_index = LogicIndex::new(None)
            .unwrap_or_else(|_| LogicIndex::empty());
        let embed_items: Vec<(u64, String)> = graph
            .pseudocode
            .iter()
            .filter(|(_, p)| p.steps.len() >= 3)
            .map(|(&id, p)| (id, pseudocode::format_pseudocode(p)))
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
        ontomics::logic::label_clusters(&mut logic_clusters, &graph.pseudocode);
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
    fn voxelmorph_pseudocode_for_spatial_transformer() {
        let graph = match build_l4_graph("/home/eti/projects/voxelmorph") {
            Some(g) => g,
            None => return,
        };

        let desc = graph
            .describe_logic("SpatialTransformer")
            .expect("describe_logic should find SpatialTransformer");

        assert!(
            !desc.pseudocode_text.is_empty(),
            "SpatialTransformer should have non-empty pseudocode",
        );
        // SpatialTransformer.forward() fundamentally does grid generation
        // and grid sampling. The pseudocode should reference these operations.
        let text_lower = desc.pseudocode_text.to_lowercase();
        assert!(
            text_lower.contains("grid") || text_lower.contains("sample"),
            "SpatialTransformer pseudocode should contain grid/sample operations: {:?}",
            desc.pseudocode_text,
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
        // If one of them doesn't have pseudocode (e.g., trivial body), that's
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
        // and behavioral info (pseudocode or logic cluster)
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
    use ontomics::centrality;
    use ontomics::graph::ConceptGraph;
    use ontomics::logic::LogicIndex;
    use ontomics::pseudocode;
    use ontomics::types::{
        CentralityScore, Entity, EntityKind, LogicCluster, Pseudocode,
        PseudocodeStep, Relationship, RelationshipKind,
    };
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// Build a minimal graph with L4 data for tool testing.
    fn build_test_graph() -> ConceptGraph {
        use ontomics::types::{AnalysisResult, Concept, Convention, Signature};

        // Minimal L1-L3 data
        let analysis = AnalysisResult {
            concepts: vec![],
            conventions: vec![],
            co_occurrence_matrix: vec![],
            signatures: vec![],
            classes: vec![],
            call_sites: vec![],
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

        // Populate L4 data manually
        graph.pseudocode.insert(
            100,
            Pseudocode {
                entity_id: 100,
                steps: vec![
                    PseudocodeStep::Assignment {
                        target: "diff".to_string(),
                        source: "subtract(pred, target)".to_string(),
                    },
                    PseudocodeStep::Assignment {
                        target: "squared".to_string(),
                        source: "pow(diff, 2)".to_string(),
                    },
                    PseudocodeStep::Return {
                        value: Some("mean(squared)".to_string()),
                    },
                ],
                body_hash: 12345,
                omitted_count: 0,
            },
        );

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
    fn describe_logic_returns_pseudocode_and_centrality() {
        let graph = build_test_graph();
        let desc = graph
            .describe_logic("compute_loss")
            .expect("describe_logic should find compute_loss");

        assert!(
            !desc.pseudocode_text.is_empty(),
            "pseudocode_text should not be empty",
        );
        assert!(
            desc.pseudocode_text.contains("subtract")
                || desc.pseudocode_text.contains("diff"),
            "pseudocode should reference loss operations: {:?}",
            desc.pseudocode_text,
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
    fn describe_logic_entity_without_pseudocode() {
        let graph = build_test_graph();
        // SpatialTransformer has centrality but no pseudocode in our test graph
        let desc = graph.describe_logic("SpatialTransformer");
        if let Some(d) = desc {
            assert!(
                d.pseudocode_text.is_empty(),
                "entity without pseudocode should have empty pseudocode_text",
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
        // Should contain the entity name
        assert!(
            text_lower.contains("compute_loss"),
            "compact context should mention entity name: {:?}",
            ctx.text,
        );
        // Should contain behavioral info (pseudocode or description)
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
        assert!(graph.pseudocode.is_empty());
        assert!(graph.logic_clusters.is_empty());
        assert!(graph.centrality.is_empty());
    }
}
