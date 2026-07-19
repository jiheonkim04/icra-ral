#!/usr/bin/env bash
set -euo pipefail

CAP_GIB="$1"
WSLCONFIG_SHA256="$2"
OUTPUT_JSON="$3"
OUTPUT_MD="$4"
REPO_ROOT="/mnt/c/Users/jiheo/tca_map"
PYTHON="/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python"

cd "$REPO_ROOT"
exec "$PYTHON" scripts/run_a2c2_resource_smoke.py \
  --resource-cap-gib "$CAP_GIB" \
  --wslconfig-sha256 "$WSLCONFIG_SHA256" \
  --resource-smoke-output "$OUTPUT_JSON" \
  --resource-smoke-md "$OUTPUT_MD"
