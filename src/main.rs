#[allow(dead_code)]
mod analyzer;
mod cache;
#[allow(dead_code)]
mod embeddings;
mod graph;
mod parser;
#[allow(dead_code)]
mod tokenizer;
mod tools;
#[allow(dead_code)]
mod types;

use clap::Parser as ClapParser;
use rmcp::ServiceExt;
use std::path::PathBuf;

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

    let cache = cache::IndexCache::open(&cli.repo)?;

    let graph = if let Some(cached) = cache.load()? {
        eprintln!("Loaded graph from cache");
        cached
    } else {
        eprintln!("Indexing {}...", cli.repo.display());
        let parse_results = parser::parse_directory(&cli.repo)?;
        eprintln!("Parsed {} files", parse_results.len());

        let analysis = analyzer::analyze(&parse_results)?;
        eprintln!(
            "Found {} concepts, {} conventions",
            analysis.concepts.len(),
            analysis.conventions.len()
        );

        let embeddings = embeddings::EmbeddingIndex::empty();
        let graph =
            graph::ConceptGraph::build(analysis, embeddings)?;

        if let Err(e) = cache.save(&graph) {
            eprintln!("Warning: failed to cache index: {e}");
        } else {
            eprintln!("Cached index to .semex/index.db");
        }

        graph
    };

    let server = tools::SemexServer::new(graph);

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
