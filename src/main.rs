mod analyzer;
mod cache;
mod config;
mod embeddings;
mod graph;
mod parser;
mod tokenizer;
mod tools;
mod types;

use clap::Parser as ClapParser;
use rmcp::ServiceExt;
use std::path::PathBuf;
use std::time::Duration;

#[derive(ClapParser)]
#[command(name = "semex", about = "Domain ontology extraction for Python codebases")]
struct Cli {
    /// Path to the Python repository to analyze
    #[arg(long)]
    repo: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // stderr-only tracing so stdout stays clean for JSON-RPC
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let config = config::Config::load(&cli.repo)?;

    let cache = cache::IndexCache::open(&cli.repo)?;

    let graph = if let Some(cached) = cache.load()? {
        eprintln!("Loaded graph from cache");
        cached
    } else {
        eprintln!("Indexing {}...", cli.repo.display());
        let parse_results = parser::parse_directory(&cli.repo)?;
        eprintln!("Parsed {} files", parse_results.len());

        let analysis_params = analyzer::AnalysisParams {
            min_frequency: config.index.min_frequency,
            tfidf_threshold: config.analysis.domain_specificity_threshold,
            convention_threshold: config.analysis.convention_threshold,
        };
        let analysis = analyzer::analyze(&parse_results, &analysis_params)?;
        eprintln!(
            "Found {} concepts, {} conventions",
            analysis.concepts.len(),
            analysis.conventions.len()
        );

        let mut embedding_index = if config.embeddings.enabled {
            match embeddings::EmbeddingIndex::new() {
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

        // Embed each concept
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

        // Add similarity edges between concepts with close embeddings
        if config.embeddings.enabled {
            graph.add_similarity_edges(
                config.embeddings.similarity_threshold,
            );
        }

        if let Err(e) = cache.save(&graph) {
            eprintln!("Warning: failed to cache index: {e}");
        } else {
            eprintln!("Cached index to .semex/index.db");
        }

        graph
    };

    let server = tools::SemexServer::new(graph);

    // Start file watcher for incremental re-indexing
    let graph_handle = server.graph_handle();
    let repo_root = cli.repo.clone();
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
            // Batch changes over a 2-second window
            match rx.recv_timeout(Duration::from_secs(2)) {
                Ok(changed) => {
                    // Drain any additional pending changes
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

                    // Full re-index (v1: simpler than true incremental)
                    let parse_results =
                        match parser::parse_directory(&repo_root) {
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

                    // Swap graph under write lock
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

                    // Persist updated cache
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
