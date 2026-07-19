#!/usr/bin/env bash
set -euo pipefail

CAP_GIB="$1"
WSLCONFIG_SHA256="$2"
REPO_ROOT="/mnt/c/Users/jiheo/tca_map"
PYTHON="/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python"

cd "$REPO_ROOT"
exec "$PYTHON" scripts/run_a2c2_resource_smoke.py \
  --resource-cap-gib "$CAP_GIB" \
  --wslconfig-sha256 "$WSLCONFIG_SHA256" \
  --resource-smoke-output "reports/a2c2_prior/resource_smoke_cap_${CAP_GIB}gb_internal.json" \
  --resource-smoke-md "reports/a2c2_prior/resource_smoke_cap_${CAP_GIB}gb_internal.md"
