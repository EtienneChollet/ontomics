mod analyzer;
mod cache;
mod config;
mod diff;
mod embeddings;
mod graph;
mod parser;
mod tokenizer;
mod tools;
mod types;

use clap::{Parser as ClapParser, Subcommand};
use rmcp::ServiceExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(ClapParser)]
#[command(name = "semex", about = "Domain ontology extraction for Python codebases")]
struct Cli {
    /// Path to the Python repository to analyze
    #[arg(long)]
    repo: PathBuf,

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
}

/// Build or load the concept graph from cache.
fn build_graph(
    repo: &Path,
    config: &config::Config,
) -> anyhow::Result<graph::ConceptGraph> {
    let cache = cache::IndexCache::open(repo)?;

    let model_cache_dir = resolve_cache_dir(&config.embeddings.model_cache_dir);

    if let Some(mut cached) = cache.load()? {
        eprintln!("Loaded graph from cache");
        if config.embeddings.enabled {
            if let Some(ref dir) = model_cache_dir {
                cached.embeddings.set_cache_dir(dir.clone());
            }
            if let Err(e) = cached.embeddings.load_model() {
                eprintln!("Warning: failed to load embedding model: {e}");
            }
        }
        return Ok(cached);
    }

    eprintln!("Indexing {}...", repo.display());
    let parse_opts = parser::ParseOptions {
        include: config.index.include.clone(),
        exclude: config.index.exclude.clone(),
        respect_gitignore: config.index.respect_gitignore,
    };
    let parse_results = parser::parse_directory(repo, &parse_opts)?;
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

    eprintln!(
        "Found {} concepts, {} conventions",
        analysis.concepts.len(),
        analysis.conventions.len()
    );

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
        for concept in &analysis.concepts {
            if let Err(e) = embedding_index.embed_concept(concept) {
                eprintln!(
                    "Warning: failed to embed '{}': {e}",
                    concept.canonical
                );
            }
        }
    }

    let mut graph =
        graph::ConceptGraph::build(analysis, embedding_index)?;

    if config.embeddings.enabled {
        graph
            .add_similarity_edges(config.embeddings.similarity_threshold);
    }

    graph.add_contrastive_edges();
    graph.detect_subconcepts();

    if let Err(e) = cache.save(&graph) {
        eprintln!("Warning: failed to cache index: {e}");
    } else {
        eprintln!("Cached index to .semex/index.db");
    }

    Ok(graph)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let config = config::Config::load(&cli.repo)?;
    let graph = build_graph(&cli.repo, &config)?;

    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => run_server(graph, &cli.repo, &config).await,
        Command::Query { term } => cmd_query(&graph, &term),
        Command::Check { identifier } => cmd_check(&graph, &identifier),
        Command::Suggest { description } => {
            cmd_suggest(&graph, &description)
        }
        Command::Diff { since } => cmd_diff(&cli.repo, &graph, &since),
        Command::Concepts { top_k } => cmd_concepts(&graph, top_k),
        Command::Conventions => cmd_conventions(&graph),
        Command::Describe { name } => cmd_describe(&graph, &name),
        Command::Locate { term } => cmd_locate(&graph, &term),
        Command::Briefing => cmd_briefing(&graph),
    }
}

// --- CLI subcommand handlers ---

fn cmd_query(
    graph: &graph::ConceptGraph,
    term: &str,
) -> anyhow::Result<()> {
    match graph.query_concept(term) {
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
) -> anyhow::Result<()> {
    let result = diff::ontology_diff(repo, since, &graph.concepts)?;
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
            serde_json::json!({
                "canonical": c.canonical,
                "occurrences": c.occurrences.len(),
                "entity_types": c.entity_types,
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

fn cmd_conventions(graph: &graph::ConceptGraph) -> anyhow::Result<()> {
    print_json(&graph.list_conventions());
    Ok(())
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
    repo: &Path,
    config: &config::Config,
) -> anyhow::Result<()> {
    let server = tools::SemexServer::new(graph, repo.to_path_buf());

    let graph_handle = server.graph_handle();
    let repo_root = repo.to_path_buf();
    let watcher_config = config.clone();
    tokio::task::spawn_blocking(move || {
        let watcher_cache = match cache::IndexCache::open(&repo_root) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: file watcher cache open failed: {e}");
                return;
            }
        };
        let (_watcher, rx) = match watcher_cache.watch(&repo_root) {
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
                        match parser::parse_directory(
                            &repo_root,
                            &watcher_parse_opts,
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
                    let analysis = match analyzer::analyze(
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

                    let new_graph = match graph::ConceptGraph::build(
                        analysis,
                        embeddings::EmbeddingIndex::empty(),
                    ) {
                        Ok(g) => g,
                        Err(e) => {
                            eprintln!(
                                "Warning: graph rebuild failed: {e}"
                            );
                            continue;
                        }
                    };

                    match graph_handle.write() {
                        Ok(mut g) => {
                            *g = new_graph;
                            eprintln!("Graph updated");
                        }
                        Err(e) => {
                            eprintln!(
                                "Warning: failed to update graph: {e}"
                            );
                        }
                    }

                    if let Ok(g) = graph_handle.read() {
                        if let Err(e) = watcher_cache.save(&g) {
                            eprintln!(
                                "Warning: cache save failed: {e}"
                            );
                        }
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
