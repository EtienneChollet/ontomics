mod analyzer;
mod cache;
mod config;
mod diff;
mod domain_pack;
mod embeddings;
mod enrichment;
mod entity;
mod graph;
#[cfg(feature = "lsp")]
mod lsp;
mod parser;
mod tokenizer;
mod tools;
mod types;

use clap::{Parser as ClapParser, Subcommand};
use rmcp::ServiceExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Spawn a watchdog thread that exits the process if the parent dies.
///
/// Checks both PID reparenting (Linux/macOS: parent becomes PID 1 or changes)
/// and `kill(pid, 0)` (works across platforms, including containers with
/// subreaper processes). Polls every 5 seconds.
fn spawn_parent_watchdog() {
    let parent_pid = std::os::unix::process::parent_id();
    if parent_pid <= 1 {
        return;
    }
    std::thread::spawn(move || {
        let parent = nix::unistd::Pid::from_raw(parent_pid as i32);
        loop {
            std::thread::sleep(Duration::from_secs(5));
            let reparented =
                std::os::unix::process::parent_id() != parent_pid;
            let dead =
                nix::sys::signal::kill(parent, None).is_err();
            if reparented || dead {
                eprintln!(
                    "Parent (PID {parent_pid}) gone, exiting."
                );
                std::process::exit(0);
            }
        }
    });
}

#[derive(ClapParser)]
#[command(name = "semex", about = "Domain ontology extraction for Python/TypeScript/JavaScript codebases")]
struct Cli {
    /// Path to the repository to analyze (defaults to current directory)
    #[arg(long, default_value = ".")]
    repo: PathBuf,

    /// Language to analyze: auto, python, typescript, javascript
    #[arg(long, default_value = "auto")]
    language: String,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Start MCP server on stdio (default when no subcommand)
    Serve,
    /// Look up a domain concept by name
    Query {
        /// Concept term to look up (e.g. "transform")
        term: String,
    },
    /// Check an identifier against project naming conventions
    Check {
        /// Identifier to check (e.g. "n_dims")
        identifier: String,
    },
    /// Suggest an identifier name from a description
    Suggest {
        /// Natural language description (e.g. "count of features")
        description: String,
    },
    /// Compare domain ontology between git refs
    Diff {
        /// Git ref to diff from (default: HEAD~5)
        #[arg(default_value = "HEAD~5")]
        since: String,
    },
    /// List all detected domain concepts
    Concepts {
        /// Maximum number of concepts to return
        #[arg(long)]
        top_k: Option<usize>,
    },
    /// List all detected naming conventions
    Conventions,
    /// Describe a function or class (structural info)
    Describe {
        /// Function or class name
        name: String,
    },
    /// Find the best entry points for working with a concept
    Locate {
        /// Concept to locate (e.g. "transform")
        term: String,
    },
    /// Print session briefing (conventions, abbreviations, warnings)
    Briefing,
    /// Export domain knowledge as a portable YAML pack
    Export {
        /// Output file path (stdout if omitted)
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// List entities (classes, functions) with optional filtering
    Entities {
        /// Filter by concept
        #[arg(long)]
        concept: Option<String>,
        /// Filter by semantic role
        #[arg(long)]
        role: Option<String>,
        /// Maximum results
        #[arg(long, default_value = "20")]
        top_k: usize,
    },
}

/// Result of building the initial graph. If embeddings were deferred,
/// `needs_embedding` contains what the background thread needs.
struct BuildResult {
    graph: graph::ConceptGraph,
    needs_embedding: Option<EmbeddingWork>,
    loaded_packs: Vec<types::DomainPack>,
}

/// Work deferred to a background thread for embedding enrichment.
/// Holds only the concept list (not full analysis) so it can be
/// reconstructed from a cached graph with partial embeddings.
struct EmbeddingWork {
    concepts: Vec<types::Concept>,
    model_cache_dir: Option<PathBuf>,
    similarity_threshold: f32,
    batch_size: usize,
    domain_packs: Vec<types::DomainPack>,
}

/// Build or load the concept graph from cache.
/// When `defer_embeddings` is true, skips the slow embedding step and
/// returns `EmbeddingWork` so it can be done in the background.
fn build_graph(
    repo: &Path,
    config: &config::Config,
    defer_embeddings: bool,
    lang: &dyn parser::LanguageParser,
) -> anyhow::Result<BuildResult> {
    let cache = cache::IndexCache::open(repo)?;

    let model_cache_dir = resolve_cache_dir(&config.embeddings.model_cache_dir);

    // Load domain packs early so they're available for all paths
    // (fresh build, cache hit, watcher, background enrichment).
    let loaded_packs = load_domain_packs(repo, config);

    if let Some(mut cached) = cache.load()? {
        let nb_concepts = cached.concepts.len();
        let nb_embedded = cached.embeddings.nb_vectors();
        eprintln!(
            "Loaded graph from cache ({nb_embedded}/{nb_concepts} embedded)"
        );

        if config.embeddings.enabled {
            if let Some(ref dir) = model_cache_dir {
                cached.embeddings.set_cache_dir(dir.clone());
            }

            // If embeddings are incomplete and we're in serve mode,
            // schedule background completion instead of blocking.
            if defer_embeddings && nb_embedded < nb_concepts {
                let concepts: Vec<types::Concept> =
                    cached.concepts.values().cloned().collect();
                return Ok(BuildResult {
                    graph: cached,
                    needs_embedding: Some(EmbeddingWork {
                        concepts,
                        model_cache_dir: model_cache_dir.clone(),
                        similarity_threshold: config
                            .embeddings
                            .similarity_threshold,
                        batch_size: config
                            .resources
                            .embedding_batch_size,
                        domain_packs: loaded_packs.clone(),
                    }),
                    loaded_packs,
                });
            }

            if let Err(e) = cached.embeddings.load_model() {
                eprintln!(
                    "Warning: failed to load embedding model: {e}"
                );
            }
        }
        return Ok(BuildResult {
            graph: cached,
            needs_embedding: None,
            loaded_packs,
        });
    }

    eprintln!("Indexing {}...", repo.display());
    let parse_opts = parser::ParseOptions {
        include: config.index.include.clone(),
        exclude: config.index.exclude.clone(),
        respect_gitignore: config.index.respect_gitignore,
    };
    let parse_results = parser::parse_directory_with(repo, &parse_opts, lang)?;
    eprintln!("Parsed {} files", parse_results.len());

    let analysis_params = analyzer::AnalysisParams {
        min_frequency: config.index.min_frequency,
        tfidf_threshold: config.analysis.domain_specificity_threshold,
        convention_threshold: config.analysis.convention_threshold,
    };
    let mut analysis =
        analyzer::analyze(&parse_results, &analysis_params)?;

    // Merge bootstrap conventions from config (if any)
    for conv_cfg in &config.conventions {
        let pattern = match conv_cfg.pattern.as_str() {
            "prefix" => types::PatternKind::Prefix(conv_cfg.value.clone()),
            "suffix" => types::PatternKind::Suffix(conv_cfg.value.clone()),
            "compound" => {
                types::PatternKind::Compound(conv_cfg.value.clone())
            }
            "conversion" => {
                types::PatternKind::Conversion(conv_cfg.value.clone())
            }
            other => {
                eprintln!(
                    "Warning: unknown convention pattern '{other}', skipping"
                );
                continue;
            }
        };
        let entity_type = conv_cfg
            .entity_types
            .first()
            .and_then(|s| match s.as_str() {
                "Function" => Some(types::EntityType::Function),
                "Parameter" => Some(types::EntityType::Parameter),
                "Variable" => Some(types::EntityType::Variable),
                "Class" => Some(types::EntityType::Class),
                _ => None,
            })
            .unwrap_or(types::EntityType::Variable);

        // Only add if not already discovered
        let already_exists = analysis.conventions.iter().any(|c| {
            std::mem::discriminant(&c.pattern)
                == std::mem::discriminant(&pattern)
                && match (&c.pattern, &pattern) {
                    (
                        types::PatternKind::Prefix(a),
                        types::PatternKind::Prefix(b),
                    ) => a == b,
                    (
                        types::PatternKind::Suffix(a),
                        types::PatternKind::Suffix(b),
                    ) => a == b,
                    (
                        types::PatternKind::Conversion(a),
                        types::PatternKind::Conversion(b),
                    ) => a == b,
                    (
                        types::PatternKind::Compound(a),
                        types::PatternKind::Compound(b),
                    ) => a == b,
                    _ => false,
                }
        });
        if !already_exists {
            analysis.conventions.push(types::Convention {
                pattern,
                entity_type,
                semantic_role: conv_cfg.role.clone(),
                examples: Vec::new(),
                frequency: 0,
            });
        }
    }

    // Merge domain packs into analysis (after bootstrap conventions, before embedding)
    for pack in &loaded_packs {
        domain_pack::merge_pack_into_analysis(pack, &mut analysis);
    }

    eprintln!(
        "Found {} concepts, {} conventions",
        analysis.concepts.len(),
        analysis.conventions.len()
    );

    // Build entities from L2 + L1 data (runs before graph construction)
    let concept_map: std::collections::HashMap<u64, types::Concept> = analysis
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
    eprintln!("Built {} entities", built_entities.len());

    // When deferring embeddings, build a lightweight graph immediately
    // and save to cache so parse/analysis work survives a restart.
    if defer_embeddings && config.embeddings.enabled {
        let concepts = analysis.concepts.clone();
        let classes_for_roles = analysis.classes.clone();
        let mut graph = graph::ConceptGraph::build_with_entities(
            analysis,
            embeddings::EmbeddingIndex::empty(),
            built_entities,
            entity_rels,
        )?;
        graph.add_abbreviation_edges();
        graph.add_contrastive_edges();
        for pack in &loaded_packs {
            domain_pack::merge_pack_associations(pack, &mut graph);
        }

        // Infer semantic roles
        let mut ents: Vec<types::Entity> =
            graph.entities.values().cloned().collect();
        entity::infer_semantic_roles(
            &mut ents,
            &classes_for_roles,
            &graph.conventions,
            &graph.concepts,
        );
        for ent in ents {
            graph.entities.insert(ent.id, ent);
        }

        if let Err(e) = cache.save(&graph) {
            eprintln!("Warning: failed to cache initial index: {e}");
        }

        eprintln!("Graph ready (embeddings deferred to background)");
        return Ok(BuildResult {
            graph,
            needs_embedding: Some(EmbeddingWork {
                concepts,
                model_cache_dir: model_cache_dir.clone(),
                similarity_threshold: config.embeddings.similarity_threshold,
                batch_size: config.resources.embedding_batch_size,
                domain_packs: loaded_packs.clone(),
            }),
            loaded_packs,
        });
    }

    let mut embedding_index = if config.embeddings.enabled {
        match embeddings::EmbeddingIndex::new(model_cache_dir) {
            Ok(idx) => {
                eprintln!("Loaded embedding model");
                idx
            }
            Err(e) => {
                eprintln!("Warning: failed to load embeddings: {e}");
                embeddings::EmbeddingIndex::empty()
            }
        }
    } else {
        embeddings::EmbeddingIndex::empty()
    };

    if config.embeddings.enabled {
        if let Err(e) = embedding_index.embed_concepts_batch(&analysis.concepts) {
            eprintln!("Warning: failed to embed concepts: {e}");
        }
    }

    let mut graph = graph::ConceptGraph::build_with_entities(
        analysis,
        embedding_index,
        built_entities,
        entity_rels,
    )?;

    if config.embeddings.enabled {
        graph
            .add_similarity_edges(config.embeddings.similarity_threshold);
    }

    graph.add_abbreviation_edges();
    graph.add_contrastive_edges();
    graph.detect_subconcepts();
    for pack in &loaded_packs {
        domain_pack::merge_pack_associations(pack, &mut graph);
    }

    // Infer semantic roles after graph enrichment (needs conventions)
    let mut ents: Vec<types::Entity> =
        graph.entities.values().cloned().collect();
    entity::infer_semantic_roles(
        &mut ents,
        &graph.classes,
        &graph.conventions,
        &graph.concepts,
    );
    for ent in ents {
        graph.entities.insert(ent.id, ent);
    }

    // Optional LSP enrichment
    #[cfg(feature = "lsp")]
    if config.lsp.enabled {
        eprintln!("Running pyright LSP enrichment...");
        match lsp::enrich_with_pyright(
            repo,
            &config.lsp.pyright_path,
            config.lsp.timeout_secs,
        ) {
            Ok(enrichment) => {
                let ents: Vec<types::Entity> =
                    graph.entities.values().cloned().collect();
                lsp::apply_enrichment(
                    &ents,
                    &enrichment,
                    &mut graph.relationships,
                );
                eprintln!("LSP enrichment applied");
            }
            Err(e) => {
                eprintln!("Warning: LSP enrichment failed: {e}");
            }
        }
    }

    if let Err(e) = cache.save(&graph) {
        eprintln!("Warning: failed to cache index: {e}");
    } else {
        eprintln!("Cached index to .semex/index.db");
    }

    Ok(BuildResult {
        graph,
        needs_embedding: None,
        loaded_packs,
    })
}

/// Run embedding enrichment incrementally: only embed concepts that
/// don't already have vectors, checkpoint to cache after each batch.
/// Safe to interrupt — next startup resumes from the last checkpoint.
/// Acquires an embedding lock to prevent duplicate work across instances.
/// Retries lock acquisition if the graph generation changes while waiting
/// (indicates a previous holder will release soon).
fn enrich_graph_with_embeddings(
    work: EmbeddingWork,
    graph_handle: std::sync::Arc<std::sync::RwLock<graph::ConceptGraph>>,
    generation: Arc<std::sync::atomic::AtomicU64>,
    repo: PathBuf,
) {
    use std::sync::atomic::Ordering;

    let my_gen = generation.load(Ordering::SeqCst);

    let cache = match cache::IndexCache::open(&repo) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Background: cache open failed: {e}");
            return;
        }
    };

    // Retry lock acquisition: if another thread in our process holds
    // the lock, it will release once it sees the generation change.
    let mut attempts = 0;
    while !cache.try_acquire_embedding_lock() {
        attempts += 1;
        if attempts > 30 {
            eprintln!(
                "Background: could not acquire embedding lock after 30s, giving up"
            );
            return;
        }
        // Check if we've been superseded while waiting
        if enrichment::is_generation_stale(&generation, my_gen) {
            eprintln!(
                "Background: generation changed while waiting for lock, stopping"
            );
            return;
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    // From here on, we hold the embedding lock — release on all exit paths.
    let result = enrich_graph_with_embeddings_inner(
        work,
        graph_handle,
        generation,
        my_gen,
        &cache,
    );

    cache.release_embedding_lock();

    if let Err(e) = result {
        eprintln!("Background: embedding failed: {e}");
    }
}

fn enrich_graph_with_embeddings_inner(
    work: EmbeddingWork,
    graph_handle: std::sync::Arc<std::sync::RwLock<graph::ConceptGraph>>,
    generation: Arc<std::sync::atomic::AtomicU64>,
    my_gen: u64,
    cache: &cache::IndexCache,
) -> anyhow::Result<()> {
    // Check generation before expensive model load
    if enrichment::is_generation_stale(&generation, my_gen) {
        eprintln!("Background: graph already replaced, skipping model load");
        return Ok(());
    }

    eprintln!("Background: loading embedding model...");
    let mut embedding_index =
        embeddings::EmbeddingIndex::new(work.model_cache_dir)?;

    // Determine which concepts still need embedding
    let embedded_ids = graph_handle
        .read()
        .map(|g| g.embeddings.vector_ids())
        .unwrap_or_default();

    let to_embed = enrichment::compute_embedding_plan(
        work.concepts,
        &embedded_ids,
    );

    let total = to_embed.len();
    if total == 0 {
        eprintln!("Background: all concepts already embedded");
    } else {
        eprintln!(
            "Background: embedding {total} concepts \
             (batch size {})...",
            work.batch_size
        );
        let mut done = 0;

        for chunk in to_embed.chunks(work.batch_size) {
            // Check if graph was replaced by watcher
            if enrichment::is_generation_stale(&generation, my_gen) {
                eprintln!(
                    "Background: graph replaced, stopping stale embedding"
                );
                break;
            }

            if let Err(e) =
                embedding_index.embed_concepts_batch(chunk)
            {
                eprintln!(
                    "Background: embedding batch failed: {e}"
                );
                break; // save whatever we have so far
            }

            // Merge new vectors into the live graph
            if let Ok(mut g) = graph_handle.write() {
                let ids: Vec<u64> = chunk.iter().map(|c| c.id).collect();
                enrichment::merge_vectors_into_graph(
                    &mut g,
                    &embedding_index,
                    &ids,
                );
            }

            done += chunk.len();
            eprintln!("Background: embedded {done}/{total}");

            // Checkpoint to cache
            if let Ok(g) = graph_handle.read() {
                if let Err(e) = cache.save(&g) {
                    eprintln!(
                        "Background: checkpoint failed: {e}"
                    );
                }
            }
        }
    }

    // Don't enrich a stale graph
    if enrichment::is_generation_stale(&generation, my_gen) {
        eprintln!("Background: graph replaced, skipping enrichment");
        return Ok(());
    }

    // Add similarity edges now that embeddings are available
    if let Ok(mut g) = graph_handle.write() {
        g.add_similarity_edges(work.similarity_threshold);
        g.add_abbreviation_edges();
        g.add_contrastive_edges();
        g.detect_subconcepts();
        for pack in &work.domain_packs {
            domain_pack::merge_pack_associations(pack, &mut g);
        }

        // Re-infer semantic roles with enriched conventions
        let mut ents: Vec<types::Entity> =
            g.entities.values().cloned().collect();
        entity::infer_semantic_roles(
            &mut ents,
            &g.classes,
            &g.conventions,
            &g.concepts,
        );
        for ent in ents {
            g.entities.insert(ent.id, ent);
        }

        eprintln!("Background: graph enriched with embeddings");
    }

    // Final cache save
    if let Ok(g) = graph_handle.read() {
        if let Err(e) = cache.save(&g) {
            eprintln!("Background: cache save failed: {e}");
        } else {
            eprintln!(
                "Background: cached index to .semex/index.db"
            );
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let mut config = config::Config::load(&cli.repo)?;

    // Resolve language: CLI flag > config file > auto-detection
    let language = match cli.language.as_str() {
        "auto" => config.language.resolve(&cli.repo),
        "python" | "py" => config::Language::Python,
        "typescript" | "ts" => config::Language::TypeScript,
        "javascript" | "js" => config::Language::JavaScript,
        other => anyhow::bail!(
            "Unknown language: {other}. Use: auto, python, typescript, javascript"
        ),
    };
    config.index.resolve_for_language(&language);

    let active_parser: Box<dyn parser::LanguageParser> = match &language {
        config::Language::Python => Box::new(parser::python_parser()),
        config::Language::TypeScript => Box::new(parser::typescript_parser()),
        config::Language::JavaScript => Box::new(parser::javascript_parser()),
        config::Language::Auto => Box::new(parser::python_parser()),
    };
    eprintln!("Language: {language:?}");

    rayon::ThreadPoolBuilder::new()
        .num_threads(config.resources.max_threads)
        .build_global()
        .ok(); // ignore if already set (e.g. in tests)
    eprintln!(
        "Resource limits: {} threads, batch size {}",
        config.resources.max_threads,
        config.resources.embedding_batch_size,
    );

    let is_serve = matches!(
        cli.command.as_ref().unwrap_or(&Command::Serve),
        Command::Serve
    );
    let result = build_graph(&cli.repo, &config, is_serve, &*active_parser)?;

    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => {
            run_server(
                result.graph,
                result.needs_embedding,
                result.loaded_packs,
                &cli.repo,
                &config,
                active_parser,
            )
            .await
        }
        Command::Query { term } => cmd_query(&result.graph, &term),
        Command::Check { identifier } => {
            cmd_check(&result.graph, &identifier)
        }
        Command::Suggest { description } => {
            cmd_suggest(&result.graph, &description)
        }
        Command::Diff { since } => {
            cmd_diff(&cli.repo, &result.graph, &since, &*active_parser)
        }
        Command::Concepts { top_k } => {
            cmd_concepts(&result.graph, top_k)
        }
        Command::Conventions => cmd_conventions(&result.graph),
        Command::Describe { name } => {
            cmd_describe(&result.graph, &name)
        }
        Command::Locate { term } => cmd_locate(&result.graph, &term),
        Command::Briefing => cmd_briefing(&result.graph),
        Command::Export { output } => cmd_export(&result.graph, output.as_deref()),
        Command::Entities {
            concept,
            role,
            top_k,
        } => cmd_entities(&result.graph, concept.as_deref(), role.as_deref(), top_k),
    }
}

// --- CLI subcommand handlers ---

fn cmd_query(
    graph: &graph::ConceptGraph,
    term: &str,
) -> anyhow::Result<()> {
    match graph.query_concept(term, &types::QueryConceptParams::default()) {
        Some(result) => print_json(&result),
        None => {
            eprintln!("No concept matching '{term}'");
            std::process::exit(1);
        }
    }
    Ok(())
}

fn cmd_check(
    graph: &graph::ConceptGraph,
    identifier: &str,
) -> anyhow::Result<()> {
    let result = graph.check_naming(identifier);
    print_json(&result);
    Ok(())
}

fn cmd_suggest(
    graph: &graph::ConceptGraph,
    description: &str,
) -> anyhow::Result<()> {
    let result = graph.suggest_name(description);
    print_json(&result);
    Ok(())
}

fn cmd_diff(
    repo: &Path,
    graph: &graph::ConceptGraph,
    since: &str,
    lang: &dyn parser::LanguageParser,
) -> anyhow::Result<()> {
    let result = diff::ontology_diff(repo, since, &graph.concepts, lang)?;
    print_json(&result);
    Ok(())
}

fn cmd_concepts(
    graph: &graph::ConceptGraph,
    top_k: Option<usize>,
) -> anyhow::Result<()> {
    let mut concepts = graph.list_concepts();
    if let Some(k) = top_k {
        concepts.truncate(k);
    }
    let summary: Vec<serde_json::Value> = concepts
        .iter()
        .map(|c| {
            let entity_count = graph
                .entities
                .values()
                .filter(|e| e.concept_tags.contains(&c.id))
                .count();
            serde_json::json!({
                "canonical": c.canonical,
                "occurrences": c.occurrences.len(),
                "entity_types": c.entity_types,
                "entity_count": entity_count,
            })
        })
        .collect();
    print_json(&summary);
    Ok(())
}

fn cmd_describe(
    graph: &graph::ConceptGraph,
    name: &str,
) -> anyhow::Result<()> {
    match graph.describe_symbol(name) {
        Some(result) => print_json(&result),
        None => {
            eprintln!("No symbol matching '{name}'");
            std::process::exit(1);
        }
    }
    Ok(())
}

fn cmd_locate(
    graph: &graph::ConceptGraph,
    term: &str,
) -> anyhow::Result<()> {
    match graph.locate_concept(term) {
        Some(result) => print_json(&result),
        None => {
            eprintln!("No concept matching '{term}'");
            std::process::exit(1);
        }
    }
    Ok(())
}

fn cmd_briefing(graph: &graph::ConceptGraph) -> anyhow::Result<()> {
    let briefing = graph.session_briefing();
    print_json(&briefing);
    Ok(())
}

fn cmd_export(
    graph: &graph::ConceptGraph,
    output: Option<&Path>,
) -> anyhow::Result<()> {
    let pack = domain_pack::export_domain_pack(graph);
    let yaml = serde_yaml::to_string(&pack)?;
    if let Some(path) = output {
        std::fs::write(path, &yaml)?;
        eprintln!("Wrote domain pack to {}", path.display());
    } else {
        print!("{yaml}");
    }
    Ok(())
}

fn cmd_conventions(graph: &graph::ConceptGraph) -> anyhow::Result<()> {
    print_json(&graph.list_conventions());
    Ok(())
}

fn cmd_entities(
    graph: &graph::ConceptGraph,
    concept: Option<&str>,
    role: Option<&str>,
    top_k: usize,
) -> anyhow::Result<()> {
    let results = graph.list_entities(concept, role, None, top_k);
    print_json(&results);
    Ok(())
}

/// Load domain packs from paths in config, resolving relative to repo root.
fn load_domain_packs(
    repo: &Path,
    config: &config::Config,
) -> Vec<types::DomainPack> {
    let mut packs = Vec::new();
    for pack_path_str in &config.domain_packs {
        let pack_path = if std::path::Path::new(pack_path_str).is_absolute() {
            PathBuf::from(pack_path_str)
        } else {
            repo.join(pack_path_str)
        };
        match domain_pack::load_domain_pack(&pack_path) {
            Ok(pack) => {
                eprintln!("Loaded domain pack: {}", pack_path.display());
                packs.push(pack);
            }
            Err(e) => {
                eprintln!(
                    "Warning: failed to load domain pack '{}': {e}",
                    pack_path.display()
                );
            }
        }
    }
    packs
}

/// Expand `~` prefix and return Some(path) if non-empty, None otherwise.
fn resolve_cache_dir(raw: &str) -> Option<PathBuf> {
    if raw.is_empty() {
        return None;
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return Some(PathBuf::from(home).join(rest));
        }
    }
    Some(PathBuf::from(raw))
}

fn print_json(value: &impl serde::Serialize) {
    println!(
        "{}",
        serde_json::to_string_pretty(value)
            .expect("serialization failed")
    );
}

// --- MCP server ---

async fn run_server(
    graph: graph::ConceptGraph,
    embedding_work: Option<EmbeddingWork>,
    loaded_packs: Vec<types::DomainPack>,
    repo: &Path,
    config: &config::Config,
    active_parser: Box<dyn parser::LanguageParser>,
) -> anyhow::Result<()> {
    spawn_parent_watchdog();

    let active_parser: Arc<dyn parser::LanguageParser> = Arc::from(active_parser);
    let server = tools::SemexServer::new(
        graph,
        repo.to_path_buf(),
        Arc::clone(&active_parser),
    );

    let graph_handle = server.graph_handle();
    let repo_root = repo.to_path_buf();
    let generation = Arc::new(std::sync::atomic::AtomicU64::new(0));

    // Spawn background embedding enrichment if needed
    if let Some(work) = embedding_work {
        let bg_handle = Arc::clone(&graph_handle);
        let bg_gen = Arc::clone(&generation);
        let bg_repo = repo_root.clone();
        std::thread::spawn(move || {
            enrich_graph_with_embeddings(
                work, bg_handle, bg_gen, bg_repo,
            );
        });
    }
    let watcher_config = config.clone();
    let watcher_packs = loaded_packs;
    let watcher_extensions: Vec<String> = active_parser
        .extensions()
        .iter()
        .map(|e| e.to_string())
        .collect();
    tokio::task::spawn_blocking(move || {
        let watcher_cache = match cache::IndexCache::open(&repo_root) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: file watcher cache open failed: {e}");
                return;
            }
        };
        let ext_refs: Vec<&str> = watcher_extensions.iter().map(|s| s.as_str()).collect();
        let (_watcher, rx) = match watcher_cache.watch(&repo_root, &ext_refs) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("Warning: file watcher failed to start: {e}");
                return;
            }
        };
        eprintln!("File watcher started");
        loop {
            match rx.recv_timeout(Duration::from_secs(2)) {
                Ok(changed) => {
                    let mut all_changed = changed;
                    while let Ok(more) = rx.try_recv() {
                        all_changed.extend(more);
                    }
                    all_changed.sort();
                    all_changed.dedup();

                    eprintln!(
                        "Re-indexing {} changed files...",
                        all_changed.len()
                    );

                    let watcher_parse_opts = parser::ParseOptions {
                        include: watcher_config.index.include.clone(),
                        exclude: watcher_config.index.exclude.clone(),
                        respect_gitignore: watcher_config
                            .index
                            .respect_gitignore,
                    };
                    let parse_results =
                        match parser::parse_directory_with(
                            &repo_root,
                            &watcher_parse_opts,
                            &*active_parser,
                        ) {
                            Ok(r) => r,
                            Err(e) => {
                                eprintln!(
                                    "Warning: re-parse failed: {e}"
                                );
                                continue;
                            }
                        };

                    let analysis_params = analyzer::AnalysisParams {
                        min_frequency: watcher_config
                            .index
                            .min_frequency,
                        tfidf_threshold: watcher_config
                            .analysis
                            .domain_specificity_threshold,
                        convention_threshold: watcher_config
                            .analysis
                            .convention_threshold,
                    };
                    let mut analysis = match analyzer::analyze(
                        &parse_results,
                        &analysis_params,
                    ) {
                        Ok(a) => a,
                        Err(e) => {
                            eprintln!(
                                "Warning: re-analysis failed: {e}"
                            );
                            continue;
                        }
                    };

                    // Re-merge domain packs into fresh analysis
                    for pack in &watcher_packs {
                        domain_pack::merge_pack_into_analysis(
                            pack, &mut analysis,
                        );
                    }

                    // Build entities
                    let watcher_concept_map: std::collections::HashMap<u64, types::Concept> =
                        analysis.concepts.iter().map(|c| (c.id, c.clone())).collect();
                    let (mut watcher_entities, watcher_entity_rels) = entity::build_entities(
                        &analysis.signatures,
                        &analysis.classes,
                        &analysis.call_sites,
                        &watcher_concept_map,
                    );

                    let mut new_graph = match graph::ConceptGraph::build_with_entities(
                        analysis.clone(),
                        embeddings::EmbeddingIndex::empty(),
                        watcher_entities.clone(),
                        watcher_entity_rels,
                    ) {
                        Ok(g) => g,
                        Err(e) => {
                            eprintln!(
                                "Warning: graph rebuild failed: {e}"
                            );
                            continue;
                        }
                    };
                    new_graph.add_abbreviation_edges();
                    new_graph.add_contrastive_edges();
                    for pack in &watcher_packs {
                        domain_pack::merge_pack_associations(
                            pack, &mut new_graph,
                        );
                    }

                    // Infer roles
                    entity::infer_semantic_roles(
                        &mut watcher_entities,
                        &new_graph.classes,
                        &new_graph.conventions,
                        &new_graph.concepts,
                    );
                    for ent in watcher_entities {
                        new_graph.entities.insert(ent.id, ent);
                    }

                    // Increment generation BEFORE replacing graph —
                    // stale background threads will see the change and stop.
                    generation.fetch_add(
                        1,
                        std::sync::atomic::Ordering::SeqCst,
                    );

                    match graph_handle.write() {
                        Ok(mut g) => {
                            *g = new_graph;
                            eprintln!("Graph updated (re-embedding in background)");
                        }
                        Err(e) => {
                            eprintln!(
                                "Warning: failed to update graph: {e}"
                            );
                            continue;
                        }
                    }

                    // Re-embed in the background
                    if watcher_config.embeddings.enabled {
                        let work = EmbeddingWork {
                            concepts: analysis.concepts,
                            model_cache_dir: resolve_cache_dir(
                                &watcher_config.embeddings.model_cache_dir,
                            ),
                            similarity_threshold: watcher_config
                                .embeddings
                                .similarity_threshold,
                            batch_size: watcher_config
                                .resources
                                .embedding_batch_size,
                            domain_packs: watcher_packs.clone(),
                        };
                        let bg_handle = Arc::clone(&graph_handle);
                        let bg_gen = Arc::clone(&generation);
                        let bg_repo = repo_root.clone();
                        std::thread::spawn(move || {
                            enrich_graph_with_embeddings(
                                work, bg_handle, bg_gen, bg_repo,
                            );
                        });
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    eprintln!("File watcher channel closed");
                    break;
                }
            }
        }
    });

    eprintln!("Starting MCP server on stdio...");
    let running = server
        .serve(rmcp::transport::io::stdio())
        .await
        .map_err(|e: std::io::Error| anyhow::anyhow!(e))?;

    running
        .waiting()
        .await
        .map_err(|e| anyhow::anyhow!("server task failed: {e}"))?;

    Ok(())
}
