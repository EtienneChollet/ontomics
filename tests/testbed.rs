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
    use std::fmt::Write as _;
    use std::io::Write as _;
    use std::path::Path;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Instant;

    /// Recorded timing for one codebase build.
    struct BuildTiming {
        name: String,
        seconds: f64,
        files: usize,
        concepts: usize,
        entities: usize,
        skipped: bool,
        skip_path: String,
    }

    /// Thread-safe cache of build timings, keyed by codebase name.
    fn timings_cache() -> &'static Mutex<Vec<BuildTiming>> {
        static CACHE: OnceLock<Mutex<Vec<BuildTiming>>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// Write all recorded timings to `testbed_timings.txt` in the project root.
    fn write_timings_file() {
        let timings = timings_cache().lock().unwrap();
        if timings.is_empty() {
            return;
        }

        let mut buf = String::new();
        for t in timings.iter() {
            if t.skipped {
                let _ = writeln!(buf, "{:<20} SKIP    (not found at {})", t.name, t.skip_path);
            } else {
                let _ = writeln!(
                    buf,
                    "{:<20} {:>5.1}s {:>5} files {:>5} concepts {:>5} entities",
                    t.name, t.seconds, t.files, t.concepts, t.entities,
                );
            }
        }

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = Path::new(manifest_dir).join("testbed_timings.txt");
        if let Ok(mut f) = std::fs::File::create(&path) {
            let _ = f.write_all(buf.as_bytes());
        }
    }

    /// A built graph plus the parser and language name that produced it.
    #[derive(Clone)]
    pub struct BuiltGraph {
        pub graph: Arc<ConceptGraph>,
        pub parser: Arc<dyn parser::LanguageParser>,
        pub language_name: &'static str,
    }

    /// Thread-safe cache of built graphs, keyed by repo path.
    fn graph_cache() -> &'static Mutex<HashMap<String, BuiltGraph>> {
        static CACHE: OnceLock<Mutex<HashMap<String, BuiltGraph>>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Cap the rayon global thread pool to one-third of CPUs (matching
    /// the production default in ResourcesConfig).
    fn init_thread_pool() {
        static INIT: OnceLock<()> = OnceLock::new();
        INIT.get_or_init(|| {
            let cpus = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            let threads = (cpus / 3).max(1);
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .ok();
            eprintln!("Rayon global pool: {threads} threads ({cpus} CPUs)");
        });
    }

    /// Build entity-to-call-sites map for logic cluster labeling.
    fn build_entity_call_sites(
        graph: &ConceptGraph,
    ) -> HashMap<u64, Vec<&types::CallSite>> {
        let mut map: HashMap<u64, Vec<&types::CallSite>> = HashMap::new();
        for cs in &graph.call_sites {
            if let Some(scope) = &cs.caller_scope {
                if let Some(entity) = graph.entities.values().find(|e| {
                    e.name == *scope && e.file == cs.file
                }) {
                    map.entry(entity.id).or_default().push(cs);
                }
            }
        }
        map
    }

    /// Build or retrieve a cached BuiltGraph for the given repo.
    /// Returns None if the repo path doesn't exist (test should skip).
    pub fn get_or_build_graph(exp: &TestbedExpectations) -> Option<BuiltGraph> {
        init_thread_pool();
        let repo = Path::new(exp.repo_path);
        if !repo.exists() {
            eprintln!("SKIP: {} not found at {}", exp.name, exp.repo_path);
            // Record skip in timings (only once per codebase)
            {
                let mut timings = timings_cache().lock().unwrap();
                if !timings.iter().any(|t| t.name == exp.name) {
                    timings.push(BuildTiming {
                        name: exp.name.to_string(),
                        seconds: 0.0,
                        files: 0,
                        concepts: 0,
                        entities: 0,
                        skipped: true,
                        skip_path: exp.repo_path.to_string(),
                    });
                }
            }
            write_timings_file();
            return None;
        }

        let mut cache = graph_cache().lock().unwrap();
        if let Some(bg) = cache.get(exp.repo_path) {
            return Some(bg.clone());
        }

        eprintln!("Building graph for {} at {}...", exp.name, exp.repo_path);
        let build_start = Instant::now();

        let config = Config::load(repo).unwrap_or_default();
        let (language, _file_count) = Language::detect(repo);

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
        eprintln!("  Parsed {} files", parse_results.len());

        let analysis_params = analyzer::AnalysisParams {
            min_frequency: config.index.min_frequency,
            tfidf_threshold: config.analysis.domain_specificity_threshold,
            convention_threshold: config.analysis.convention_threshold,
            language: language_name.to_string(),
        };
        let analysis =
            analyzer::analyze(&parse_results, &analysis_params)
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
            &[],
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
            Vec::new(),
        )
        .expect("graph build failed");

        if config.embeddings.enabled {
            graph.cluster_and_add_similarity_edges(config.embeddings.similarity_threshold);

            // Release the concept embedding model to free GPU VRAM.
            // Only the vectors are needed from this point forward.
            let mut slim_emb = EmbeddingIndex::empty();
            for id in graph.embeddings.vector_ids() {
                if let Some(v) = graph.embeddings.get_vector(id) {
                    slim_emb.insert_vector(id, v.clone());
                }
            }
            graph.embeddings = slim_emb;
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

        // L4: logic embeddings (CodeRankEmbed for function body text)
        if config.embeddings.enabled {
            match ontomics::logic::LogicIndex::new(None) {
                Ok(mut logic_idx) => {
                    let items: Vec<(u64, String)> = graph
                        .signatures
                        .iter()
                        .filter_map(|sig| {
                            let body = sig.body.as_ref()?;
                            let entity = graph.entities.values().find(|e| {
                                e.name == sig.name && e.file == sig.file
                            })?;
                            Some((entity.id, body.body_text.clone()))
                        })
                        .collect();
                    if let Err(e) = logic_idx.embed_batch(items) {
                        eprintln!("  Warning: logic embedding failed: {e}");
                    }
                    let entity_ids: Vec<u64> =
                        graph.entities.keys().copied().collect();
                    graph.logic_clusters = ontomics::logic::cluster_logic(
                        &logic_idx,
                        &entity_ids,
                        1.0 - config.logic.similarity_threshold,
                    );
                    // Label clusters by most common callee
                    let mut clusters = std::mem::take(&mut graph.logic_clusters);
                    let entity_cs = build_entity_call_sites(&graph);
                    ontomics::logic::label_clusters(&mut clusters, &entity_cs);
                    graph.logic_clusters = clusters;
                    eprintln!(
                        "  Logic: {} embeddings, {} clusters",
                        logic_idx.nb_vectors(),
                        graph.logic_clusters.len(),
                    );
                    // Transfer vectors to a model-free index to
                    // release the GPU model and avoid VRAM exhaustion
                    // when multiple codebases are tested sequentially.
                    let mut slim_idx = ontomics::logic::LogicIndex::empty();
                    for id in logic_idx.vector_ids() {
                        if let Some(v) = logic_idx.get_vector(id) {
                            slim_idx.insert_vector(id, v.clone());
                        }
                    }
                    graph.logic_index = slim_idx;
                }
                Err(e) => eprintln!("  Warning: logic model unavailable: {e}"),
            }
        }

        // L4: centrality (PageRank on entity graph)
        if config.centrality.enabled {
            let centrality = ontomics::centrality::compute_centrality(
                &graph.entities,
                &graph.relationships,
                config.centrality.damping,
                config.centrality.iterations,
            );
            eprintln!("  Computed centrality for {} entities", centrality.len());
            graph.centrality = centrality;
        }

        // L4: cross-reference logic clusters with concept clusters
        if !graph.logic_clusters.is_empty() {
            let concept_assignments: HashMap<u64, usize> = graph
                .entities
                .values()
                .filter_map(|ent| {
                    ent.concept_tags
                        .iter()
                        .find_map(|&cid| graph.concepts.get(&cid)?.cluster_id)
                        .map(|cid| (ent.id, cid))
                })
                .collect();
            graph.logic_concept_overlaps = ontomics::logic::cross_reference(
                &graph.logic_clusters,
                &concept_assignments,
                0.1,
            );
        }

        let build_elapsed = build_start.elapsed();
        let build_secs = build_elapsed.as_secs_f64();

        eprintln!(
            "  Graph ready: {} concepts, {} entities, {} conventions ({:.1}s)",
            graph.concepts.len(),
            graph.entities.len(),
            graph.conventions.len(),
            build_secs,
        );

        // Record timing
        {
            let mut timings = timings_cache().lock().unwrap();
            timings.push(BuildTiming {
                name: exp.name.to_string(),
                seconds: build_secs,
                files: parse_results.len(),
                concepts: graph.concepts.len(),
                entities: graph.entities.len(),
                skipped: false,
                skip_path: String::new(),
            });
        }
        write_timings_file();

        // Performance ceiling: only assert when ONTOMICS_ASSERT_PERF is set,
        // since debug builds are much slower than release.
        if std::env::var("ONTOMICS_ASSERT_PERF").is_ok() {
            assert!(
                build_secs <= exp.max_build_seconds as f64,
                "{}: build took {:.1}s, exceeds ceiling of {}s",
                exp.name, build_secs, exp.max_build_seconds,
            );
        }

        let bg = BuiltGraph {
            graph: Arc::new(graph),
            parser: Arc::clone(&lang),
            language_name,
        };
        cache.insert(exp.repo_path.to_string(), bg.clone());
        Some(bg)
    }

    /// Macro to skip test gracefully if codebase not found.
    macro_rules! skip_if_missing {
        ($exp:expr) => {
            match $crate::testbed::get_or_build_graph($exp) {
                Some(bg) => bg,
                None => {
                    eprintln!("SKIP: {} not available", $exp.name);
                    return;
                }
            }
        };
    }
    pub(crate) use skip_if_missing;
}
