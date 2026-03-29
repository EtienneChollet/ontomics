#!/bin/bash
# Install semex MCP server for a Python project.
# Usage: ./install.sh /path/to/python/project
#
# Creates .mcp.json in the target project so Claude Code
# automatically starts semex when opened in that directory.

set -e

SEMEX_BIN="/home/eti/projects/semex/target/release/semex"

if [ ! -f "$SEMEX_BIN" ]; then
    echo "Building semex..."
    cargo build --release --manifest-path /home/eti/projects/semex/Cargo.toml
fi

TARGET="${1:-.}"
TARGET=$(cd "$TARGET" && pwd)

cat > "$TARGET/.mcp.json" << EOF
{
  "mcpServers": {
    "semex": {
      "command": "$SEMEX_BIN",
      "args": ["--repo", "$TARGET"]
    }
  }
}
EOF

echo "Installed semex for: $TARGET"
echo "Restart Claude Code in that directory to activate."
