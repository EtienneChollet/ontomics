# ontomics

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Rust](https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)
![platform](https://img.shields.io/badge/macOS_·_Linux-grey)
![MCP](https://img.shields.io/badge/MCP_Server-green)
![Claude Code](https://img.shields.io/badge/Claude_Code-cc785c)
![Codex](https://img.shields.io/badge/Codex-black)

ontomics extracts domain knowledge from codebases to reduce LLM token consumption by 20x and time in agentic search by 10x. It gathers the concepts, naming conventions, and vocabulary embedded in your code and makes them queryable via [MCP](https://modelcontextprotocol.io/).

## Benchmark

Tested with Claude Sonnet — same question, with and without ontomics.

"What does 'transform' mean in this codebase?" on [voxelmorph](https://github.com/voxelmorph/voxelmorph) ([full transcript](doc/benchmarks/transform-voxelmorph.md)):

|                | With ontomics | Without    |
|----------------|---------------|------------|
| Tool calls     | 1             | 19         |
| Tokens         | ~3.7k         | ~76k       |
| Time           | 5s            | 1m 15s     |
| Answer quality | Complete      | Complete   |

"What are the main domain concepts in this codebase?" on [ScribblePrompt](https://github.com/halleewong/ScribblePrompt) ([full transcript](doc/benchmarks/concepts-scribbleprompt.md)):

|                | With ontomics | Without    |
|----------------|---------------|------------|
| Tool calls     | 1             | 26         |
| Tokens         | ~3.7k         | ~61.6k     |
| Time           | ~5s           | 56s        |
| Answer quality | Complete      | Complete   |

Both conditions produced complete, correct answers. ontomics got there in one call.

## What it does that search can't

Search tells you where a string appears. An LSP tells you where a symbol is defined and referenced. Neither answers: what are the domain concepts in this codebase? How do they relate? What naming conventions emerged? What changed in the domain vocabulary since last release?

ontomics builds a semantic index of your project's domain — clustering related symbols into concepts, detecting naming conventions from usage frequency, resolving abbreviations, and tracking how the vocabulary evolves over time. That index can be exported as a portable artifact to bootstrap conventions in other repos.

## Install

Install once, available in every project.

**macOS (Homebrew):**
```bash
brew install EtienneChollet/tap/ontomics
```

**macOS/Linux:**
```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/EtienneChollet/ontomics/releases/latest/download/ontomics-installer.sh | sh
```

**npm:**
```bash
npm install -g @ontomics/ontomics
```

**From source:**
```bash
git clone https://github.com/EtienneChollet/ontomics.git
cd ontomics
cargo build --release
```

Then register with your MCP client:
```bash
claude mcp add -s user ontomics -- ontomics
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
