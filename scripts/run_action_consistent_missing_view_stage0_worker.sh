#!/usr/bin/env bash
set -uo pipefail

RUN_DIR="$1"
PYTHON_BIN="/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python"
SPEC="configs/action_consistent_missing_view_distillation_xvla_frozen_spec.json"
CONTRACT="configs/action_consistent_missing_view_distillation_xvla_stage0_execution_contract.json"
THRESHOLDS="reports/action_consistent_missing_view_distillation_numerical_threshold_freeze_result.json"
MICROBATCH="reports/action_consistent_missing_view_distillation_microbatch_preflight_result.json"

mkdir -p "$RUN_DIR"
printf '%s\n' "$$" > "$RUN_DIR/wsl_shell_pid.txt"
printf '%s\n' "$(date --iso-8601=seconds)" > "$RUN_DIR/worker_started_at.txt"
printf '%q ' "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1" "$PYTHON_BIN" \
  scripts/run_action_consistent_missing_view_stage0.py \
  --run-dir "$RUN_DIR" --spec "$SPEC" --contract "$CONTRACT" \
  --threshold-report "$THRESHOLDS" --microbatch-report "$MICROBATCH" \
  > "$RUN_DIR/exact_command.txt"
printf '\n' >> "$RUN_DIR/exact_command.txt"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl \
  "$PYTHON_BIN" scripts/run_action_consistent_missing_view_stage0.py \
  --run-dir "$RUN_DIR" --spec "$SPEC" --contract "$CONTRACT" \
  --threshold-report "$THRESHOLDS" --microbatch-report "$MICROBATCH"
CODE=$?

printf '%s\n' "$CODE" > "$RUN_DIR/shell_exit_code.txt"
printf '%s\n' "$(date --iso-8601=seconds)" > "$RUN_DIR/worker_finished_at.txt"
exit "$CODE"
