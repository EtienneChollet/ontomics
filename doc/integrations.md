# Integrations

ontomics is a stdio MCP server. Any tool that speaks MCP over stdio can use it with zero code changes — only configuration differs per host.

## Claude Code

```bash
claude mcp add --transport stdio ontomics -- npx -y @ontomics/ontomics --repo .
```

Or from a local build:

```bash
claude mcp add --transport stdio ontomics -- /path/to/ontomics --repo .
```

## Codex CLI

Add to `~/.codex/config.toml` (global) or `.codex/config.toml` (per-project):

```toml
[mcp_servers.ontomics]
command = "npx"
args = ["-y", "@ontomics/ontomics", "--repo", "."]
startup_timeout_sec = 30
tool_timeout_sec = 120
```

Or from a local build:

```toml
[mcp_servers.ontomics]
command = "/path/to/ontomics"
args = ["--repo", "."]
startup_timeout_sec = 30
tool_timeout_sec = 120
```

Set `tool_timeout_sec = 120` — the first indexing pass on large codebases can exceed the 60s default.

## pi

pi has no built-in MCP support. Install the community adapter first:

```bash
pi install npm:pi-mcp-adapter
```

Then add to `~/.pi/agent/mcp.json` (global) or `.pi/mcp.json` (per-project):

```json
{
  "mcpServers": {
    "ontomics": {
      "command": "npx",
      "args": ["-y", "@ontomics/ontomics", "--repo", "."],
      "lifecycle": "eager",
      "directTools": true
    }
  }
}
```

Or from a local build:

```json
{
  "mcpServers": {
    "ontomics": {
      "command": "/path/to/ontomics",
      "args": ["--repo", "."],
      "lifecycle": "eager",
      "directTools": true
    }
  }
}
```

- `"lifecycle": "eager"` — starts ontomics at agent startup so the indexing pass doesn't block the first tool call.
- `"directTools": true` — exposes all ontomics tools as first-class tools instead of routing through a single proxy.

## Agent instructions

Each host has its own mechanism for telling the agent *when* to use ontomics tools:

| Host | File | Notes |
|------|------|-------|
| Claude Code | `CLAUDE.md` | Checked into repo root |
| Codex CLI | `codex-instructions.md` | Repo root or `~/.codex/instructions.md` |
| pi | `.pi/agent/instructions.md` | Repo root or `~/.pi/agent/instructions.md` |

A minimal instruction block that works across all three:

```
ontomics extracts domain ontologies from codebases (Python, TypeScript, JavaScript).
Use it BEFORE reading files when exploring unfamiliar code.

- "what is X" / "what does X mean" -> query_concept
- "how does X work" / "where is X" -> locate_concept + describe_symbol
- "what are the main concepts" -> list_concepts
- "what naming conventions" -> list_conventions
- "is this name correct" -> check_naming
- "what should I call this" -> suggest_name
```
