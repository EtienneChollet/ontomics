# CLAUDE.md

## What is ontomics?

A Rust MCP server that extracts domain ontologies from Python codebases. It indexes **domain semantics** (concepts, naming conventions, vocabulary) — not code structure. See `SPEC.md` for the full design contract.

## Source of truth

`SPEC.md` defines all API contracts, data types, logic flows, and configuration. Every implementation decision should trace back to the spec. If the spec is wrong, update the spec first, then the code.

## Paths

- **Project**: `/home/eti/projects/ontomics`
- **Test corpus (primary)**: `/home/eti/projects/voxelmorph` — medical image registration, rich domain vocabulary
- **Test corpus (secondary)**: `/home/eti/projects/neurite` — neural network utilities, overlapping domain

Use these real codebases for integration testing and validation. When testing `check_naming`, `query_concept`, etc., run against voxelmorph to verify real-world behavior.

## Build and test

```bash
cargo build                  # dev build
cargo build --release        # release build
cargo test                   # all tests
cargo test -- --nocapture    # tests with stdout
cargo clippy                 # lint (must pass with zero warnings)
```

## Rust conventions

- **Errors**: `Result<T, E>` everywhere, `thiserror` for error types. No `unwrap()` outside tests. No `panic!` in library code.
- **Modules**: one file per module, no `mod.rs` nesting beyond one level.
- **Builders**: use builder pattern when constructors take >3 parameters.
- **Serialization**: derive `Serialize`/`Deserialize` on all public types that cross the MCP boundary.
- **Parallelism**: use `rayon` for file parsing (embarrassingly parallel). Keep the MCP event loop single-threaded (`tokio`).
- **Tests**: `#[cfg(test)] mod tests` at the bottom of each module. Integration tests in `tests/` directory.

## Test strategy

**Unit tests** use small synthetic fixtures — hardcoded Python snippets or tiny `.py` files in `tests/fixtures/`. These must be deterministic and self-contained.

**Integration tests** run the full pipeline against voxelmorph/neurite. These validate real-world behavior but may be slower. Gate them behind `#[ignore]` or a feature flag if they require the test corpora to be present.

**Key acceptance tests** (from the spec):
- `check_naming("n_dims")` on voxelmorph → `Inconsistent`, suggests `ndim`
- `query_concept("transform")` on voxelmorph → finds `spatial_transform`, `apply_transform`, `trf`
- `list_conventions()` on voxelmorph → detects `nb_` prefix, `is_` booleans, `_to_` conversions

## Implementation order

Build bottom-up, each layer testable independently:

1. **`types.rs`** — data model structs and enums
2. **`tokenizer.rs`** — identifier splitting (pure functions, easy to test)
3. **`parser.rs`** — tree-sitter extraction (test against fixture `.py` files)
4. **`analyzer.rs`** — TF-IDF, convention detection, co-occurrence (test with synthetic corpora)
5. **`embeddings.rs`** — fastembed integration (test similarity on known word pairs)
6. **`graph.rs`** — concept graph construction and query methods
7. **`cache.rs`** — SQLite persistence + file watcher
8. **`tools.rs`** — MCP tool handlers (thin wrappers around graph methods)
9. **`main.rs`** — wire it all together, MCP server startup

Each step should compile and pass tests before moving to the next.

## Key design decisions

- **No ML for convention detection** — frequency analysis and pattern matching only. Embeddings are only used for concept similarity clustering, not for detecting naming rules.
- **TF-IDF over subtokens** where "documents" = files. This naturally separates domain terms (`displacement`, `segmentation`) from generic terms (`value`, `data`, `self`).
- **Convention detection threshold**: minimum 3 examples before declaring a pattern (e.g., need `nb_features`, `nb_bins`, `nb_steps` before declaring `nb_` is a convention).
- **MCP transport**: stdio only. No HTTP, no WebSocket.
- **Cache location**: `<target_repo>/.ontomics/index.db` — lives inside the repo being analyzed, not in ontomics's own directory.

