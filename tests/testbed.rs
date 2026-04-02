mod testbed {
    pub mod codebases;
    pub mod expectations;
    pub mod universal;

    use expectations::TestbedExpectations;
    use ontomics::analyzer;
    use ontomics::config::{Config, Language};
    use ontomics::embeddings::EmbeddingIndex;
    use ontomics::entity;
    use ontomics::graph::ConceptGraph;
    use ontomics::parser::{self, ParseOptions};
    use ontomics::types;
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::{Arc, Mutex, OnceLock};

    /// Thread-safe cache of built ConceptGraphs, keyed by repo path.
    fn graph_cache() -> &'static Mutex<HashMap<String, Arc<ConceptGraph>>> {
        static CACHE: OnceLock<Mutex<HashMap<String, Arc<ConceptGraph>>>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Build or retrieve a cached ConceptGraph for the given repo.
    /// Returns None if the repo path doesn't exist (test should skip).
    pub fn get_or_build_graph(exp: &TestbedExpectations) -> Option<Arc<ConceptGraph>> {
        let repo = Path::new(exp.repo_path);
        if !repo.exists() {
            eprintln!("SKIP: {} not found at {}", exp.name, exp.repo_path);
            return None;
        }

        let mut cache = graph_cache().lock().unwrap();
        if let Some(graph) = cache.get(exp.repo_path) {
            return Some(Arc::clone(graph));
        }

        eprintln!("Building graph for {} at {}...", exp.name, exp.repo_path);

        let config = Config::load(repo).unwrap_or_default();
        let (language, _file_count) = Language::detect(repo);

        let lang: Box<dyn parser::LanguageParser> = match &language {
            Language::Python => Box::new(parser::python_parser()),
            Language::TypeScript => Box::new(parser::typescript_parser()),
            Language::JavaScript => Box::new(parser::javascript_parser()),
            Language::Rust => Box::new(parser::rust_parser()),
            Language::Auto => {
                let (detected, _) = Language::detect(repo);
                match detected {
                    Language::TypeScript => Box::new(parser::typescript_parser()),
                    Language::JavaScript => Box::new(parser::javascript_parser()),
                    Language::Rust => Box::new(parser::rust_parser()),
                    _ => Box::new(parser::python_parser()),
                }
            }
        };

        let mut index_config = config.index.clone();
        index_config.resolve_for_language(&language);

        let parse_opts = ParseOptions {
            include: index_config.include.clone(),
            exclude: index_config.exclude.clone(),
            respect_gitignore: index_config.respect_gitignore,
        };
        let parse_results = parser::parse_directory_with(repo, &parse_opts, &*lang)
            .expect("parsing failed");
        eprintln!("  Parsed {} files", parse_results.len());

        let analysis_params = analyzer::AnalysisParams {
            min_frequency: config.index.min_frequency,
            tfidf_threshold: config.analysis.domain_specificity_threshold,
            convention_threshold: config.analysis.convention_threshold,
            domain_specificity_threshold: config
                .analysis
                .domain_specificity_threshold,
        };
        let analysis =
            analyzer::analyze(&parse_results, &analysis_params, language.name())
                .expect("analysis failed");
        eprintln!(
            "  Found {} concepts, {} conventions",
            analysis.concepts.len(),
            analysis.conventions.len()
        );

        // Build entities
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
        eprintln!("  Built {} entities", built_entities.len());

        // Build embedding index
        let mut embedding_index = if config.embeddings.enabled {
            match EmbeddingIndex::new(None) {
                Ok(idx) => {
                    eprintln!("  Loaded embedding model");
                    idx
                }
                Err(e) => {
                    eprintln!("  Embedding model unavailable: {e}");
                    EmbeddingIndex::empty()
                }
            }
        } else {
            EmbeddingIndex::empty()
        };

        if config.embeddings.enabled {
            if let Err(e) = embedding_index.embed_concepts_batch(&analysis.concepts) {
                eprintln!("  Warning: failed to embed concepts: {e}");
            }
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

        // Infer semantic roles
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

        eprintln!(
            "  Graph ready: {} concepts, {} entities, {} conventions",
            graph.concepts.len(),
            graph.entities.len(),
            graph.conventions.len()
        );

        let arc = Arc::new(graph);
        cache.insert(exp.repo_path.to_string(), Arc::clone(&arc));
        Some(arc)
    }

    /// Macro to skip test gracefully if codebase not found.
    macro_rules! skip_if_missing {
        ($exp:expr) => {
            match $crate::testbed::get_or_build_graph($exp) {
                Some(g) => g,
                None => {
                    eprintln!("SKIP: {} not available", $exp.name);
                    return;
                }
            }
        };
    }
    pub(crate) use skip_if_missing;
}
