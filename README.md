# ontomics

ontomics extracts the domain knowledge embedded in your codebase — concepts, naming conventions, vocabulary, and relationships — and makes it queryable. It serves this knowledge to AI coding assistants and developers through [MCP](https://modelcontextprotocol.io/).

## Examples

**"What are the main concepts in this codebase?"**
> Top concepts: `transform` (spatial_transform, apply_transform, TransformLayer, trf), `segmentation` (seg, labels, one_hot), `displacement` (disp, flow, vel). 847 symbols across 12 concept clusters.

**"What does `trf` mean in this codebase?"**
> `trf` is an abbreviation for `transform` — related symbols include `spatial_transform`, `apply_transform`, and `TransformLayer`.

**"What does `SpatialTransformer` do?"**
> Class in `layers.py:45`. Applies a spatial transformation to an image tensor. Related concepts: `transform`, `interpolation`. Called by `VxmDense`, `VxmAffine`.

**"Is `n_dims` the right name?"**
> Inconsistent. This project uses `ndim` (42 occurrences across 15 files). `n_dims` appears 0 times.

**"What changed in the domain since last week?"**
> 2 new concepts: `prompt`, `click_map`. 1 renamed: `mask` cluster absorbed `binary_mask`. Convention `nb_` prefix extended to `nb_prompts`.

## What it does that search can't

Search tells you where a string appears. An LSP tells you where a symbol is defined and referenced. Neither answers: what are the domain concepts in this codebase? How do they relate? What naming conventions emerged? What changed in the domain vocabulary since last release?

ontomics builds a semantic index of your project's domain — clustering related symbols into concepts, detecting naming conventions from usage frequency, resolving abbreviations, and tracking how the vocabulary evolves over time. That index can be exported as a portable artifact to bootstrap conventions in other repos.

## Install

Install once, available in every project.

```bash
npm install -g @ontomics/ontomics
chmod +x $(npm root -g)/@ontomics/ontomics/node_modules/.bin_real/ontomics

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

No configuration needed. ontomics auto-detects the repo and indexes it on first run.

## Supported languages

Python, TypeScript, JavaScript, Rust. Auto-detected from file extensions.

## Tools

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

ontomics parses every file in your repo, extracts identifiers and docstrings, clusters related terms by embedding similarity, and detects naming conventions through frequency analysis. The index lives at `<repo>/.ontomics/index.db` — subsequent startups load from cache and watch for changes.

Configuration via `.ontomics/config.toml` in the repo root. All fields have sensible defaults. See `SPEC.md` for the full design contract.
