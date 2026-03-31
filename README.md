# ontomics

Every codebase develops its own vocabulary — abbreviations, naming patterns, domain terms that make sense to longtime contributors but are opaque to newcomers. `trf` means "transform". `nb_` means "count of". `ndim` not `n_dims`. You learn this by reading thousands of lines of code, or by asking someone who already has.

ontomics learns it automatically. It reads your codebase, discovers the domain vocabulary and naming conventions, and makes that knowledge available to AI assistants (or any tool that speaks [MCP](https://modelcontextprotocol.io/)).

## What can it do?

**Understand your project's vocabulary:**
> "What does `trf` mean in this codebase?"
>
> `trf` is an abbreviation for `transform` — related symbols include `spatial_transform`, `apply_transform`, and `TransformLayer`.

**Check names against project conventions:**
> "Is `n_dims` the right name?"
>
> Inconsistent. This project uses `ndim` (42 occurrences across 15 files). `n_dims` appears 0 times.

**Suggest names that fit the project's style:**
> "What should I call a variable that counts features?"
>
> `nb_features` — this project uses the `nb_` prefix for counts (`nb_bins`, `nb_steps`, `nb_features`).

**Find where concepts live:**
> "Where should I start reading about transforms?"
>
> `SpatialTransformer` in `layers.py:45`, `spatial_transform()` in `utils.py:112`, `apply_transform()` in `utils.py:203`.

## How is this different from search?

Grep answers "where does this exact string appear?" — but only if you already know the string.

ontomics answers "what does this project call things?" It discovers that `spatial_transform`, `apply_transform`, `TransformLayer`, and `trf` are all the same underlying concept. It detects that `nb_` is a project-wide prefix meaning "count of" by observing `nb_features`, `nb_bins`, and `nb_steps` appearing 40+ times. No search tool can derive conventions from usage patterns — ontomics does this through statistical analysis of the codebase.

For structural code navigation (call graphs, find-all-references, go-to-definition), use an LSP. ontomics is specifically for **domain knowledge**: the naming conventions and vocabulary of a codebase, derived from the code itself.

## Install

ontomics is an [MCP](https://modelcontextprotocol.io/) server — it gives AI coding assistants access to your project's domain knowledge. Install it once and it's available in every project.

**npm (recommended):**
```bash
# Claude Code
claude mcp add -s user ontomics -- npx -y @ontomics/ontomics

# Codex
codex mcp add ontomics -- npx -y @ontomics/ontomics
```

**From source:**
```bash
git clone https://github.com/EtienneChollet/ontomics.git
cd ontomics
cargo build --release
claude mcp add -s user ontomics ./target/release/ontomics
```

No configuration needed. ontomics auto-detects the repo from its working directory and indexes it on first run.

## Supported languages

| Language | Extensions |
|----------|-----------|
| Python | `.py` |
| TypeScript | `.ts`, `.tsx` |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` |
| Rust | `.rs` |

Language is auto-detected from file extensions. To override, create `.ontomics/config.toml` in your repo:
```toml
language = "python"
```

## Full tool reference

| Tool | What it does |
|------|--------------|
| `query_concept` | Find all variants, related concepts, and occurrences of a term |
| `check_naming` | Check an identifier against project conventions; suggests the canonical form |
| `suggest_name` | Generate an identifier name that fits the project's vocabulary |
| `list_concepts` | List the top domain concepts by frequency |
| `list_conventions` | List all detected naming patterns (prefixes, suffixes, conversions) |
| `locate_concept` | Find the key signatures, classes, and files for a concept |
| `describe_symbol` | Get the signature, docstring, and relationships for a function or class |
| `list_entities` | List code entities (classes, functions) of a given kind |
| `ontology_diff` | Show new, changed, or removed domain concepts since a git ref |
| `export_domain_pack` | Export domain knowledge as portable YAML for use in other repos |

## How it works

ontomics indexes your codebase into `<repo>/.ontomics/index.db`. On first run it parses every file, extracts identifiers and docstrings, clusters related terms using embedding similarity, and detects naming conventions through frequency analysis. Subsequent startups load from cache and watch for file changes.

All configuration lives in `.ontomics/config.toml` — every field has sensible defaults. See `SPEC.md` for the full design contract.
