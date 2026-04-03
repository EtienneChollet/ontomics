# Contributing

## Build

```bash
cargo build --release          # production binary
cargo build                    # debug build
cargo clippy                   # lint (must pass with zero warnings)
```

## Test

### Unit tests

```bash
cargo test --lib               # all unit tests (~246, fast)
cargo test --lib config        # tests in a single module
```

### Testbed (integration tests)

The testbed builds full ontomics graphs for real codebases and validates tool outputs against expectations. Use the cargo alias:

```bash
cargo testbed
```

This automatically enables GPU acceleration and limits thread parallelism. The alias is defined in `.cargo/config.toml` (not committed — machine-specific).

#### GPU-accelerated embeddings

Embedding inference uses candle. GPU backends are opt-in at compile time via feature flags:

| Feature | Backend | Requirement |
|---------|---------|-------------|
| `cuda`  | NVIDIA CUDA | CUDA toolkit installed |
| `metal` | Apple Metal | macOS with Apple Silicon |
| (none)  | CPU only | Works everywhere |

The production binary ships CPU-only. GPU features are for local development only.

#### Setting up `.cargo/config.toml`

Create `.cargo/config.toml` in the project root with the right feature for your machine:

```toml
# Linux with NVIDIA GPU
[alias]
testbed = "test --features cuda --test testbed -- --test-threads=1"

# macOS with Apple Silicon
[alias]
testbed = "test --features metal --test testbed -- --test-threads=1"

# CPU only
[alias]
testbed = "test --test testbed -- --test-threads=1"
```

## Local MCP development

To test ontomics as an MCP server in Claude Code, point your config at the local release binary.

**Project-level** (`.mcp.json` in the repo root, analyzes a fixed repo):
```json
{
  "mcpServers": {
    "ontomics": {
      "command": "/path/to/ontomics/target/release/ontomics",
      "args": ["--repo", "/path/to/target/codebase"]
    }
  }
}
```

**Global** (`~/.claude/.mcp.json`, analyzes whichever project you're in):
```json
{
  "mcpServers": {
    "ontomics": {
      "command": "/path/to/ontomics/target/release/ontomics",
      "args": ["--repo", "."]
    }
  }
}
```

After changing code, rebuild and restart your Claude Code session:
```bash
cargo build --release
```

The MCP config points directly at the binary, so rebuilding is all that's needed — no reinstall step.

## CLI subcommands

You can test individual tools without an MCP client:

```bash
cargo run -- concepts /path/to/codebase
cargo run -- query /path/to/codebase "transform"
cargo run -- check /path/to/codebase "myFunctionName"
cargo run -- suggest /path/to/codebase "count of features"
cargo run -- conventions /path/to/codebase
cargo run -- entities /path/to/codebase
cargo run -- locate /path/to/codebase "transform"
cargo run -- describe /path/to/codebase "SomeClass"
cargo run -- diff /path/to/codebase HEAD~5
```

## Releases

Releases use `cargo-dist`. CI builds on tag push via `.github/workflows/release.yml`. To cut a release, see the `cargo-release` workflow.
