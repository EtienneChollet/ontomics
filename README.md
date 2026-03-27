# semex

A Rust MCP server that extracts domain ontologies from Python codebases.

## What it does

semex indexes **domain semantics** — the naming conventions and vocabulary of a specific codebase — not code structure. It parses all identifiers, docstrings, and comments via tree-sitter, then builds a queryable concept graph from statistical analysis of the results.

## How it's different from search tools

Grep and tree-sitter are one-shot and syntactic. They answer "where does this term appear?" — but only if you already know what to search for.

semex answers different questions:

- **Concept clustering**: `spatial_transform`, `apply_transform`, `TransformLayer`, and `trf` are all the same underlying domain concept. semex discovers this automatically via embedding similarity and edit distance, so querying `transform` returns all variants including the abbreviation.

- **Project-specific convention detection**: semex derives conventions empirically from corpus statistics, not from a style guide. If `nb_features`, `nb_bins`, and `nb_steps` appear 40+ times, semex concludes that `nb_` means "count of" in this project. `check_naming("n_dims")` returns `Inconsistent` with suggestion `ndim` because it knows the vocabulary of the actual codebase.

- **Vocabulary-aware name suggestion**: `suggest_name("count of features")` returns `nb_features` because semex knows the project's conventions. No search tool can do this.

For structural navigation — finding all uses of a function, call graphs, import analysis — use an LSP or a code-index server. semex is specifically for the question: *what are the naming conventions and domain vocabulary of this codebase, derived from the code itself?*

## MCP tools

| Tool | Input | What it does |
|------|-------|--------------|
| `query_concept` | `{ "term": "transform" }` | Returns all variants, related concepts, conventions, and occurrences |
| `check_naming` | `{ "identifier": "n_dims" }` | Checks against project conventions; suggests canonical form if inconsistent |
| `suggest_name` | `{ "description": "count of features" }` | Suggests an identifier name based on project vocabulary |
| `list_conventions` | `{}` | Lists all detected naming patterns (prefixes, suffixes, conversions) |
| `list_concepts` | `{ "top_k": 50 }` | Lists concepts ordered by frequency |
| `ontology_diff` | `{ "since": "HEAD~5" }` | Surfaces new, changed, or removed domain concepts since a git ref |

## Usage

```bash
# Build
cargo build --release

# Run against a Python codebase
./target/release/semex --repo /path/to/python/project
```

semex writes its index to `<repo>/.semex/index.db` and serves MCP tools over stdio. On first run it indexes the full repo; subsequent startups load from cache and watch for file changes.

Configure via `.semex/config.toml` in the repo root — all fields have sensible defaults, zero config required to get started. See `SPEC.md` for the full design contract.
