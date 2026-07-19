#!/usr/bin/env bash
set -euo pipefail

MODE="$1"
RUN_ID="$2"
REPO_ROOT="/mnt/c/Users/jiheo/tca_map"
PYTHON="/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python"
RUN_ROOT="runs/a2c2_prior/clean_host_panel/${RUN_ID}"
REPORT_ROOT="reports/a2c2_prior"

case "$MODE" in
  rollout-base|rollout-prior|adjudicate) ;;
  *) echo "Unsupported clean-host panel mode: $MODE" >&2; exit 64 ;;
esac

cd "$REPO_ROOT"
exec "$PYTHON" scripts/run_a2c2_problem_verification.py \
  --mode "$MODE" \
  --base-rollout-partial "${RUN_ROOT}/base_rollout_partial.json" \
  --base-rollout-output "${REPORT_ROOT}/${RUN_ID}_base_closed_loop_result.json" \
  --base-rollout-md "${REPORT_ROOT}/${RUN_ID}_base_closed_loop_result.md" \
  --prior-rollout-partial "${RUN_ROOT}/prior_rollout_partial.json" \
  --prior-rollout-output "${REPORT_ROOT}/${RUN_ID}_prior_closed_loop_result.json" \
  --prior-rollout-md "${REPORT_ROOT}/${RUN_ID}_prior_closed_loop_result.md" \
  --adjudication-output "${REPORT_ROOT}/${RUN_ID}_frozen_adjudication_result.json" \
  --adjudication-md "${REPORT_ROOT}/${RUN_ID}_frozen_adjudication_result.md"
