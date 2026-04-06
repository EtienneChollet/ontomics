# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ABSOLUTE RULE — No AI attribution

**NEVER mention Claude Code, Anthropic, ChatGPT, Codex, OpenAI, or any AI tool in commits, PRs, issues, comments, or any git artifact.** No "Co-Authored-By", no "Generated with", no AI attribution of any kind. This applies to all text that enters the git history or appears on GitHub.

## ABSOLUTE RULE — Worktree workflow

**ALL source code and test changes MUST go through a worktree branch and PR.** Never commit implementation work directly to main. **NEVER edit a single file before the worktree exists.**

**The worktree MUST be created BEFORE any files are edited. This is the very first action for any implementation task. No exceptions. Do not edit on main and stash/move retroactively.**

Worktree location: `./worktrees/per-<issue-number>-<issue-name>` (e.g., `worktrees/per-42-testbed-thread-cap`). **NEVER place worktrees outside the project root (e.g., `../`).**

Workflow:
1. Create a feature branch: `epc28/<description>` (associate with a Linear issue when one exists)
2. Create the worktree: `git worktree add -b epc28/<name> worktrees/per-<issue>-<name> main`
3. Work ONLY inside that worktree — all edits, tests, and commits happen there
4. Create a PR from the feature branch to main
5. Merge to main only after review/validation

Direct commits to main are only acceptable for version bumps and release automation (cargo-release).

## ABSOLUTE RULE — Embeddings must stay enabled

**NEVER disable embeddings** (`embeddings.enabled = false`) in any `.ontomics/config.toml` for any codebase without explicit permission from the user. Embeddings are a core part of the pipeline — disabling them silently degrades concept discovery, clustering, and suggest_name quality.

## ABSOLUTE RULE — Tests are sacred

**NEVER modify, delete, weaken, skip, or rewrite any test without explicit permission from the user.** This is the single most important rule in this project. No exceptions.

- If a test fails, the implementation is wrong — fix the implementation.
- Do not change assertions, expected values, or test structure to make a failing test pass.
- Do not add `#[ignore]`, comment out, or remove failing tests.
- Do not refactor tests "for clarity" or "consistency" unless explicitly asked.
- The only time a test may be modified is when the user explicitly says to modify that specific test.
- Do not widen assertion windows (e.g., top_n thresholds) to absorb implementation regressions.
- Do not remove expected entities, concepts, or conventions from test expectations.
- Do not replace specific assertions with weaker structural checks.
- Do not "update expectations to match reality" — reality must match expectations.
- Do not relax naming checks, convention checks, or entity lists because the tool produces wrong output.
- Do not move must-contain items to should-contain or drop them entirely.
- Failing testbed tests are bugs in `src/`, not bugs in `tests/`.
- The testbed expectations are the definition of done — they define what ontomics MUST produce.
- Any change to analysis logic (TF-IDF, stop words, convention detection) that breaks testbed expectations must be fixed in the analysis code until the tests pass, not by weakening the tests.
- When in doubt: the test is right, the implementation is wrong.

## Terminology

- **"pi"** — Refers to the [pi agent harness](https://github.com/badlogic/pi-mono), NOT Raspberry Pi or the number. When the user mentions "pi", they are talking about this agent harness UI.

## What this is

ontomics is a Rust MCP server that extracts domain ontologies from codebases (Python, TypeScript, JavaScript, Rust). It parses source files with tree-sitter, clusters related identifiers by embedding similarity, detects naming conventions, clusters functions by behavioral similarity using code embeddings, and exposes the results as MCP tools. Runs locally with no API keys.

## Build and test

```bash
cargo build --release          # production binary
cargo build                    # debug build
cargo test                     # all tests (inline in each module)
cargo test --lib config        # tests in a single module
cargo clippy                   # lint (must pass with zero warnings)
```

Optional feature flag: `cargo build --features lsp` enables pyright-based LSP enrichment for inheritance chains.

## Releases

**All four version sources MUST be bumped in lockstep on every release:**

1. `Cargo.toml` — `version` field
2. `Cargo.lock` — auto-updated via `cargo update -p ontomics`
3. `server.json` — both top-level `version` and `packages[0].version`

**Release procedure:**
1. Bump versions in `Cargo.toml` and `server.json` (all three locations)
2. Run `cargo update -p ontomics` to sync `Cargo.lock`
3. Commit: `chore: Release ontomics version X.Y.Z`
4. Tag: `git tag vX.Y.Z`
5. Push commit and tag: `git push && git push origin vX.Y.Z`

CI (`release.yml`) handles everything from there: builds binaries, publishes to GitHub Releases, npm, Homebrew, and MCP Registry. All four channels must show the same version after a release.

Distribution targets: macOS (aarch64, x86_64), Linux (x86_64, glibc 2.28+). See `dist-workspace.toml` for config.

## Architecture

The pipeline runs in this order. Each module is a single file under `src/`.

1. **`parser.rs`** — `LanguageParser` trait with implementations for Python, TypeScript, JavaScript, Rust. Uses tree-sitter to walk ASTs and extract `RawIdentifier`, `Signature`, `ClassInfo`, `CallSite`. Parallel file parsing via rayon.

2. **`tokenizer.rs`** — Splits identifiers into subtokens (`spatial_transform` → `["spatial", "transform"]`), handles snake_case, camelCase, ALLCAPS boundaries. Also detects abbreviations.

3. **`analyzer.rs`** — TF-IDF scoring on subtokens to find domain-specific concepts. Builds `Concept` nodes, `Convention` patterns, and `Relationship` edges (co-occurrence, abbreviation). Takes `AnalysisParams` (min_frequency, tfidf_threshold, convention_threshold).

4. **`embeddings.rs`** — `EmbeddingModel` trait with pluggable backends (BGE-small, CodeRankEmbed, Jina Code, GTE-ModernBERT). `EmbeddingIndex` stores concept ID → vector mapping using BGE-small (`BAAI/bge-small-en-v1.5`, 384-dim). Candle + safetensors inference.

5. **`cluster.rs`** — Agglomerative clustering with average linkage (priority queue, O(N² log N)). Groups concepts by embedding similarity. Assigns `cluster_id` on each `Concept`.

6. **`entity.rs`** — Promotes functions/classes into `Entity` nodes with concept tags and semantic roles. Builds entity-level relationships (Instantiates, InheritsFrom, Uses).

7. **`logic.rs`** — L4 behavioral layer. `LogicIndex` embeds raw function body text via CodeRankEmbed (`nomic-ai/CodeRankEmbed`, 768-dim) and clusters functions by behavioral similarity. Produces `LogicCluster` groups with behavioral labels. Also contains `cross_reference()` for Jaccard overlap between logic and concept clusters (called from `main.rs`).

8. **`centrality.rs`** — PageRank centrality scoring for entities using structural edges (`InheritsFrom`, `Uses`).

9. **`graph.rs`** — `ConceptGraph` is the central data structure. Holds concepts, relationships, conventions, embeddings, entities, signatures, classes, call_sites, logic_clusters, centrality scores. All query methods (query_concept, check_naming, suggest_name, locate_concept, describe_symbol, describe_logic, etc.) live here.

10. **`tools.rs`** — `OntomicsServer` implements `rmcp::ServerHandler`. Maps MCP tool calls to `ConceptGraph` query methods. Supports deferred startup (graph populated in background).

11. **`cache.rs`** — SQLite persistence at `<repo>/.ontomics/index.db`. Auto-invalidation via `ONTOMICS_CACHE_VERSION` (hash of indexing source files, computed in `build.rs`). File watcher (notify) for incremental re-indexing.

12. **`diff.rs`** — `ontology_diff`: parses a base git ref's source tree via git2, re-runs analysis, and diffs concept sets against HEAD.

13. **`domain_pack.rs`** — Export/import of portable YAML domain knowledge (abbreviations, conventions, terms, associations).

14. **`enrichment.rs`** — Background embedding planning and merging. Computes embedding plans (which concepts and function bodies still need vectors), merges computed vectors into the graph, checks generation staleness. Model loading and inference happen in `main.rs`.

15. **`config.rs`** — `Config` loaded from `.ontomics/config.toml`. Sections: language, index, analysis, embeddings, cache, resources, conventions, domain_packs.

16. **`types.rs`** — All shared types: `Concept`, `Entity`, `Relationship`, `Convention`, `Occurrence`, `Signature`, `ClassInfo`, `CallSite`, `FunctionBody`, `LogicCluster`, `CentralityScore`, `DomainPack`, query param/result structs.

17. **`lsp.rs`** — Behind `lsp` feature flag. Shells out to pyright for cross-file inheritance resolution.

18. **`main.rs`** — CLI (clap) with subcommands (serve, query, check, suggest, diff, concepts, conventions, describe, file, locate, trace, briefing, export, update, health, map, type-flows, trace-type, entities, benchmark-embeddings). Default is `serve` (MCP stdio). Orchestrates the full pipeline: config → parse → analyze → embed → cluster → entity build → logic embed → logic cluster → centrality → graph. Supports deferred embedding (background thread) for fast MCP startup. Parent process watchdog for clean exit.

## Key design decisions

- **`build.rs`** hashes `INDEX_FILES` (entity, analyzer, parser, cache, tokenizer, types, logic, centrality) into `ONTOMICS_CACHE_VERSION`. Any change to these files auto-invalidates all caches.
- **Dual embedding models**: Concept embeddings use BGE-small (384-dim); behavioral embeddings use CodeRankEmbed (768-dim). Both loaded via the `EmbeddingModel` trait.
- **Deferred startup**: In serve mode, graph is built without embeddings first, MCP is available immediately, embeddings complete in background thread.
- **No `unwrap()` outside tests, no `panic!` in library code.** Error handling uses `anyhow::Result` throughout.
- **Modules map 1:1 to files** — no `mod.rs` nesting.
- **Entity IDs use a distinct hash namespace** from concept IDs to prevent collisions.
- **`serde(default)`** on cached types ensures old caches deserialize gracefully when new fields are added.

## Configuration

Optional `.ontomics/config.toml` in repo root. All fields have sensible defaults. Key knobs:
- `language`: auto (default), python, typescript, javascript, rust
- `index.min_frequency`: minimum subtoken frequency (default: 2)
- `analysis.domain_specificity_threshold`: TF-IDF cutoff (default: 0.3)
- `embeddings.similarity_threshold`: clustering merge threshold (default: 0.75)
- `resources.max_threads`: rayon thread cap (default: half CPUs)
