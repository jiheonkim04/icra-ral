#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
echo "TCA-Map preflight (shell)"
echo "Repo root: $REPO_ROOT"
"$PYTHON_BIN" -m tca_map.launch.preflight
