# ontomics

A Rust MCP server that extracts domain ontologies from codebases. It indexes **domain semantics** — the naming conventions, vocabulary, and structural interfaces of a specific codebase — not code structure.

## Supported languages

| Language | Extensions | Auto-detected |
|----------|-----------|---------------|
| Python | `.py` | yes |
| TypeScript | `.ts`, `.tsx` | yes |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` | yes |
| Rust | `.rs` | yes |

Language is auto-detected from file extensions in the repo root, or set explicitly in `.ontomics/config.toml`:
```toml
language = "python"  # or "typescript" / "ts", "javascript" / "js", "rust" / "rs"
```

## How it's different from search tools

Grep and tree-sitter are one-shot and syntactic. They answer "where does this term appear?" — but only if you already know what to search for.

ontomics answers different questions:

- **Concept clustering**: `spatial_transform`, `apply_transform`, `TransformLayer`, and `trf` are all the same underlying domain concept. ontomics discovers this automatically via embedding similarity and edit distance, so querying `transform` returns all variants including the abbreviation.

- **Project-specific convention detection**: ontomics derives conventions empirically from corpus statistics, not from a style guide. If `nb_features`, `nb_bins`, and `nb_steps` appear 40+ times, ontomics concludes that `nb_` means "count of" in this project. `check_naming("n_dims")` returns `Inconsistent` with suggestion `ndim` because it knows the vocabulary of the actual codebase.

- **Vocabulary-aware name suggestion**: `suggest_name("count of features")` returns `nb_features` because ontomics knows the project's conventions. No search tool can do this.

- **Entity understanding**: `describe_symbol("SpatialTransformer")` returns the class signature, docstring, concept tags, and relationships — without reading any files.

For structural navigation — finding all uses of a function, call graphs, import analysis — use an LSP or a code-index server. ontomics is specifically for the question: *what are the naming conventions and domain vocabulary of this codebase, derived from the code itself?*

## MCP tools

| Tool | Input | What it does |
|------|-------|--------------|
| `query_concept` | `{ "term": "transform" }` | Returns all variants, related concepts, conventions, and occurrences |
| `check_naming` | `{ "identifier": "n_dims" }` | Checks against project conventions; suggests canonical form if inconsistent |
| `suggest_name` | `{ "description": "count of features" }` | Suggests an identifier name based on project vocabulary |
| `list_concepts` | `{ "top_k": 50 }` | Lists concepts ordered by frequency |
| `list_conventions` | `{}` | Lists all detected naming patterns (prefixes, suffixes, conversions) |
| `locate_concept` | `{ "term": "transform" }` | Ranked signatures, classes, and files — where to start reading about a concept |
| `describe_symbol` | `{ "name": "SpatialTransformer" }` | Signature, docstring, concept tags, and relationships for a function or class |
| `list_entities` | `{ "kind": "class" }` | List code entities (classes, functions) that instantiate domain concepts |
| `ontology_diff` | `{ "since": "HEAD~5" }` | Surfaces new, changed, or removed domain concepts since a git ref |
| `export_domain_pack` | `{}` | Export domain knowledge as portable YAML for bootstrapping conventions in other repos |

## Install

```bash
git clone https://github.com/EtienneChollet/ontomics.git
cd ontomics
cargo build --release
claude mcp add -s user ontomics ./target/release/ontomics
```

That's it. ontomics is now available in every project you open with Claude Code. It auto-detects the repo from its working directory — no `--repo` flag needed.

## Usage

ontomics writes its index to `<repo>/.ontomics/index.db` and serves MCP tools over stdio. On first run it indexes the full repo; subsequent startups load from cache and watch for file changes.

Configure via `.ontomics/config.toml` in the repo root — all fields have sensible defaults, zero config required to get started. See `SPEC.md` for the full design contract.
