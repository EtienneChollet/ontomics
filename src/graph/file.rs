use super::ConceptGraph;
use crate::types::{
    DescribeFileResult, FileNestingTree, FileSymbol, SymbolKind,
};
use std::collections::HashSet;
use std::path::PathBuf;

impl ConceptGraph {
    /// Return a concept-annotated structural overview of every file whose
    /// path ends with `path`. Classes include nested method symbols.
    pub fn describe_file(&self, path: &str) -> Vec<DescribeFileResult> {
        let mut all_files: HashSet<PathBuf> = HashSet::new();
        for sig in &self.signatures {
            all_files.insert(sig.file.clone());
        }
        for cls in &self.classes {
            all_files.insert(cls.file.clone());
        }

        let matching_files: Vec<PathBuf> = all_files
            .into_iter()
            .filter(|f| f.display().to_string().ends_with(path))
            .collect();

        let mut results = Vec::new();

        for file in &matching_files {
            let file_classes: Vec<&crate::types::ClassInfo> = self
                .classes
                .iter()
                .filter(|c| &c.file == file)
                .collect();

            let class_names: HashSet<&str> =
                file_classes.iter().map(|c| c.name.as_str()).collect();

            let mut symbols: Vec<FileSymbol> = Vec::new();

            for cls in &file_classes {
                let entity = self.entities.values().find(|e| {
                    e.name == cls.name && e.file == cls.file
                });

                let concepts = self.resolve_concept_tags(entity);
                let role = entity.map(|e| e.semantic_role.clone());

                let methods: Vec<FileSymbol> = cls
                    .methods
                    .iter()
                    .map(|method_name| {
                        let method_sig =
                            self.signatures.iter().find(|s| {
                                s.file == cls.file
                                    && s.name == *method_name
                                    && s.scope.as_ref().is_some_and(
                                        |sc| sc.starts_with(&cls.name),
                                    )
                            });

                        let method_entity =
                            self.entities.values().find(|e| {
                                e.name == *method_name
                                    && e.file == cls.file
                                    && e.kind
                                        == crate::types::EntityKind::Method
                            });

                        let m_concepts =
                            self.resolve_concept_tags(method_entity);
                        let m_role =
                            method_entity.map(|e| e.semantic_role.clone());

                        let (params, return_type, line) =
                            match method_sig {
                                Some(sig) => (
                                    Self::format_params(&sig.params),
                                    sig.return_type.clone(),
                                    sig.line,
                                ),
                                None => (Vec::new(), None, cls.line),
                            };

                        FileSymbol {
                            name: method_name.clone(),
                            kind: SymbolKind::Method,
                            line,
                            concepts: m_concepts,
                            role: m_role,
                            params,
                            return_type,
                            bases: Vec::new(),
                            methods: Vec::new(),
                        }
                    })
                    .collect();

                symbols.push(FileSymbol {
                    name: cls.name.clone(),
                    kind: SymbolKind::Class,
                    line: cls.line,
                    concepts,
                    role,
                    params: Vec::new(),
                    return_type: None,
                    bases: cls.bases.clone(),
                    methods,
                });
            }

            for sig in &self.signatures {
                if sig.file != *file {
                    continue;
                }
                let is_standalone =
                    sig.scope.as_ref().is_none_or(|s| {
                        !class_names.iter().any(|cn| s.starts_with(cn))
                    });
                if !is_standalone {
                    continue;
                }

                let entity = self.entities.values().find(|e| {
                    e.name == sig.name && e.file == sig.file
                });

                let concepts = self.resolve_concept_tags(entity);
                let role = entity.map(|e| e.semantic_role.clone());

                symbols.push(FileSymbol {
                    name: sig.name.clone(),
                    kind: SymbolKind::Function,
                    line: sig.line,
                    concepts,
                    role,
                    params: Self::format_params(&sig.params),
                    return_type: sig.return_type.clone(),
                    bases: Vec::new(),
                    methods: Vec::new(),
                });
            }

            symbols.sort_by_key(|s| s.line);

            results.push(DescribeFileResult {
                file: file.clone(),
                symbols,
            });
        }

        results.sort_by(|a, b| a.file.cmp(&b.file));
        results
    }

    /// Look up the nesting tree for a file whose path ends with `path`.
    pub fn nesting_tree(&self, path: &str) -> Option<&FileNestingTree> {
        self.nesting_trees
            .iter()
            .find(|st| st.file.to_string_lossy().ends_with(path))
    }
}

#[cfg(test)]
mod tests {
    use crate::embeddings::EmbeddingIndex;
    use crate::graph::ConceptGraph;
    use crate::types::*;
    use std::collections::HashSet;
    use std::path::PathBuf;

    fn make_concept(
        id: u64, canonical: &str, identifiers: &[&str],
    ) -> Concept {
        Concept {
            id, canonical: canonical.to_string(),
            subtokens: vec![canonical.to_string()],
            occurrences: identifiers.iter().map(|name| Occurrence {
                file: PathBuf::from("test.py"), line: 1,
                identifier: name.to_string(), entity_type: EntityType::Function,
            }).collect(),
            entity_types: HashSet::from([EntityType::Function]),
            embedding: None, cluster_id: None,
            subconcepts: Vec::new(), doc_context: Vec::new(),
        }
    }

    fn make_test_graph() -> ConceptGraph {
        let analysis = AnalysisResult {
            concepts: vec![
                make_concept(1, "transform", &["spatial_transform", "apply_transform", "transform"]),
                make_concept(2, "spatial", &["spatial_transform"]),
                make_concept(3, "ndim", &["ndim", "ndim", "ndim"]),
                make_concept(4, "nb", &["nb_features", "nb_bins", "nb_steps", "nb_dims"]),
                make_concept(5, "features", &["nb_features", "features"]),
            ],
            conventions: vec![Convention {
                pattern: PatternKind::Prefix("nb_".to_string()),
                entity_type: EntityType::Parameter,
                semantic_role: "count".to_string(),
                examples: vec!["nb_features".into(), "nb_bins".into(), "nb_steps".into(), "nb_dims".into()],
                frequency: 4,
            }],
            co_occurrence_matrix: vec![((1, 2), 1.0)],
            signatures: Vec::new(), classes: Vec::new(),
            call_sites: Vec::new(), nesting_trees: Vec::new(), doc_texts: Vec::new(),
        };
        ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap()
    }

    fn make_describe_file_graph() -> ConceptGraph {
        let concepts = vec![
            make_concept(1, "transform", &["spatial_transform"]),
            make_concept(2, "spatial", &["spatial_transform"]),
            make_concept(3, "loss", &["dice_loss"]),
            make_concept(4, "network", &["VoxNet"]),
        ];
        let signatures = vec![
            Signature {
                name: "forward".to_string(),
                params: vec![Param { name: "x".to_string(),
                    type_annotation: Some("Tensor".to_string()), default: None }],
                return_type: Some("Tensor".to_string()), decorators: Vec::new(),
                docstring_first_line: None, file: PathBuf::from("src/networks.py"),
                line: 15, scope: Some("VoxNet".to_string()), body: None,
            },
            Signature {
                name: "init_weights".to_string(), params: vec![],
                return_type: None, decorators: Vec::new(),
                docstring_first_line: None, file: PathBuf::from("src/networks.py"),
                line: 25, scope: Some("VoxNet".to_string()), body: None,
            },
            Signature {
                name: "spatial_transform".to_string(),
                params: vec![
                    Param { name: "vol".to_string(),
                        type_annotation: Some("Tensor".to_string()), default: None },
                    Param { name: "trf".to_string(),
                        type_annotation: None, default: None },
                ],
                return_type: Some("Tensor".to_string()), decorators: Vec::new(),
                docstring_first_line: None, file: PathBuf::from("src/networks.py"),
                line: 40, scope: None, body: None,
            },
            Signature {
                name: "dice_loss".to_string(),
                params: vec![Param { name: "pred".to_string(),
                    type_annotation: Some("Tensor".to_string()), default: None }],
                return_type: Some("float".to_string()), decorators: Vec::new(),
                docstring_first_line: None, file: PathBuf::from("src/losses.py"),
                line: 5, scope: None, body: None,
            },
        ];
        let classes = vec![ClassInfo {
            name: "VoxNet".to_string(),
            bases: vec!["nn.Module".to_string()],
            methods: vec!["forward".to_string(), "init_weights".to_string()],
            attributes: Vec::new(),
            docstring_first_line: Some("Voxel network.".to_string()),
            file: PathBuf::from("src/networks.py"), line: 10,
        }];
        let entities = vec![
            Entity { id: 100, name: "VoxNet".to_string(), kind: EntityKind::Class,
                concept_tags: vec![4], semantic_role: "network".to_string(),
                file: PathBuf::from("src/networks.py"), line: 10,
                signature_idx: None, class_info_idx: None },
            Entity { id: 101, name: "forward".to_string(), kind: EntityKind::Method,
                concept_tags: vec![], semantic_role: "entry point".to_string(),
                file: PathBuf::from("src/networks.py"), line: 15,
                signature_idx: None, class_info_idx: None },
            Entity { id: 102, name: "spatial_transform".to_string(), kind: EntityKind::Function,
                concept_tags: vec![1, 2], semantic_role: "transform".to_string(),
                file: PathBuf::from("src/networks.py"), line: 40,
                signature_idx: None, class_info_idx: None },
            Entity { id: 103, name: "dice_loss".to_string(), kind: EntityKind::Function,
                concept_tags: vec![3], semantic_role: "loss".to_string(),
                file: PathBuf::from("src/losses.py"), line: 5,
                signature_idx: None, class_info_idx: None },
        ];
        let analysis = AnalysisResult {
            concepts, conventions: Vec::new(), co_occurrence_matrix: Vec::new(),
            signatures, classes, call_sites: Vec::new(),
            nesting_trees: Vec::new(), doc_texts: Vec::new(),
        };
        ConceptGraph::build_with_entities(
            analysis, EmbeddingIndex::empty(), entities, Vec::new(), Vec::new(),
        ).unwrap()
    }

    #[test]
    fn test_describe_file_exact_match() {
        let graph = make_describe_file_graph();
        let results = graph.describe_file("src/losses.py");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file, PathBuf::from("src/losses.py"));
        assert_eq!(results[0].symbols.len(), 1);
        let sym = &results[0].symbols[0];
        assert_eq!(sym.name, "dice_loss");
        assert!(matches!(sym.kind, SymbolKind::Function));
        assert_eq!(sym.concepts, vec!["loss".to_string()]);
        assert_eq!(sym.params, vec!["pred: Tensor".to_string()]);
        assert_eq!(sym.return_type.as_deref(), Some("float"));
    }

    #[test]
    fn test_describe_file_partial_match() {
        let graph = make_describe_file_graph();
        let results = graph.describe_file("networks.py");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file, PathBuf::from("src/networks.py"));
        assert_eq!(results[0].symbols.len(), 2);
    }

    #[test]
    fn test_describe_file_no_match() {
        let graph = make_describe_file_graph();
        assert!(graph.describe_file("nonexistent.py").is_empty());
    }

    #[test]
    fn test_describe_file_classes_with_methods() {
        let graph = make_describe_file_graph();
        let results = graph.describe_file("src/networks.py");
        assert_eq!(results.len(), 1);

        let symbols = &results[0].symbols;
        let class_sym = symbols.iter().find(|s| s.name == "VoxNet")
            .expect("VoxNet must be present");
        assert!(matches!(class_sym.kind, SymbolKind::Class));
        assert_eq!(class_sym.line, 10);
        assert_eq!(class_sym.concepts, vec!["network".to_string()]);
        assert_eq!(class_sym.bases, vec!["nn.Module".to_string()]);
        assert_eq!(class_sym.role.as_deref(), Some("network"));

        assert_eq!(class_sym.methods.len(), 2);
        let forward = class_sym.methods.iter().find(|m| m.name == "forward")
            .expect("forward must be a method");
        assert!(matches!(forward.kind, SymbolKind::Method));
        assert_eq!(forward.params, vec!["x: Tensor".to_string()]);
        assert_eq!(forward.return_type.as_deref(), Some("Tensor"));
        assert_eq!(forward.role.as_deref(), Some("entry point"));

        let st = symbols.iter().find(|s| s.name == "spatial_transform")
            .expect("spatial_transform must be present");
        assert!(matches!(st.kind, SymbolKind::Function));
        assert_eq!(st.line, 40);
        assert!(st.concepts.contains(&"transform".to_string()));
        assert!(st.concepts.contains(&"spatial".to_string()));
        assert_eq!(st.params, vec!["vol: Tensor".to_string(), "trf".to_string()]);
    }

    #[test]
    fn test_nesting_tree_lookup() {
        let mut graph = make_test_graph();
        graph.nesting_trees = vec![FileNestingTree {
            file: PathBuf::from("src/model.py"),
            root: NestingNode {
                name: "model.py".to_string(), kind: NestingKind::Module,
                line: 0,
                children: vec![
                    NestingNode {
                        name: "train".to_string(), kind: NestingKind::Function,
                        line: 1, children: Vec::new(),
                    },
                    NestingNode {
                        name: "MyModel".to_string(), kind: NestingKind::Class,
                        line: 10,
                        children: vec![NestingNode {
                            name: "forward".to_string(), kind: NestingKind::Method,
                            line: 15, children: Vec::new(),
                        }],
                    },
                ],
            },
        }];

        let tree = graph.nesting_tree("src/model.py");
        assert!(tree.is_some());
        let tree = tree.unwrap();
        assert_eq!(tree.root.kind, NestingKind::Module);
        assert_eq!(tree.root.children.len(), 2);
        assert!(graph.nesting_tree("model.py").is_some());
        assert!(graph.nesting_tree("nonexistent.py").is_none());
    }

    #[test]
    fn test_nesting_tree_class_nesting() {
        let mut graph = make_test_graph();
        graph.nesting_trees = vec![FileNestingTree {
            file: PathBuf::from("layers.py"),
            root: NestingNode {
                name: "layers.py".to_string(), kind: NestingKind::Module, line: 0,
                children: vec![NestingNode {
                    name: "ConvBlock".to_string(), kind: NestingKind::Class, line: 5,
                    children: vec![
                        NestingNode {
                            name: "__init__".to_string(), kind: NestingKind::Method,
                            line: 6, children: Vec::new(),
                        },
                        NestingNode {
                            name: "forward".to_string(), kind: NestingKind::Method,
                            line: 20, children: Vec::new(),
                        },
                    ],
                }],
            },
        }];

        let tree = graph.nesting_tree("layers.py").unwrap();
        let class_node = &tree.root.children[0];
        assert_eq!(class_node.name, "ConvBlock");
        assert_eq!(class_node.kind, NestingKind::Class);
        assert_eq!(class_node.children.len(), 2);
        assert_eq!(class_node.children[0].kind, NestingKind::Method);
        assert_eq!(class_node.children[1].kind, NestingKind::Method);
    }
}
