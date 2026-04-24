#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${AIOPS_MCP_VENV:-"$ROOT_DIR/.aiops/mcp-venv"}"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install awslabs.aws-api-mcp-server awslabs.eks-mcp-server

cat <<EOF
MCP Python environment is ready:
  $VENV_DIR/bin/python

The app will use this interpreter automatically for the default AWS API and EKS MCP servers.
To force a different interpreter, start the app with:
  AIOPS_MCP_PYTHON=/path/to/python npm run dev
EOF
