#!/bin/bash
# Install ontomics MCP server for a Python project.
# Usage: ./install.sh /path/to/python/project
#
# Creates .mcp.json in the target project so Claude Code
# automatically starts ontomics when opened in that directory.

set -e

ONTOMICS_BIN="/home/eti/projects/ontomics/target/release/ontomics"

if [ ! -f "$ONTOMICS_BIN" ]; then
    echo "Building ontomics..."
    cargo build --release --manifest-path /home/eti/projects/ontomics/Cargo.toml
fi

TARGET="${1:-.}"
TARGET=$(cd "$TARGET" && pwd)

cat > "$TARGET/.mcp.json" << EOF
{
  "mcpServers": {
    "ontomics": {
      "command": "$ONTOMICS_BIN",
      "args": ["--repo", "$TARGET"]
    }
  }
}
EOF

echo "Installed ontomics for: $TARGET"
echo "Restart Claude Code in that directory to activate."
