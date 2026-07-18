#!/usr/bin/env bash
set -uo pipefail

MODE="$1"
RUN_DIR="$2"
NOISE_REPORT="${3:-}"
PYTHON_BIN="/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python"
SPEC="configs/action_consistent_missing_view_distillation_xvla_frozen_spec.json"

mkdir -p "$RUN_DIR"
printf '%s\n' "$$" > "$RUN_DIR/wsl_shell_pid.txt"
printf '%s\n' "$(date --iso-8601=seconds)" > "$RUN_DIR/worker_started_at.txt"

ARGS=(--mode "$MODE" --run-dir "$RUN_DIR" --spec "$SPEC")
if [[ "$MODE" == "microbatch" ]]; then
  ARGS+=(--noise-report "$NOISE_REPORT")
fi
printf '%q ' "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1" "$PYTHON_BIN" scripts/run_action_consistent_missing_view_preflight.py "${ARGS[@]}" > "$RUN_DIR/exact_command.txt"
printf '\n' >> "$RUN_DIR/exact_command.txt"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl \
  "$PYTHON_BIN" scripts/run_action_consistent_missing_view_preflight.py "${ARGS[@]}"
CODE=$?

printf '%s\n' "$CODE" > "$RUN_DIR/shell_exit_code.txt"
printf '%s\n' "$(date --iso-8601=seconds)" > "$RUN_DIR/worker_finished_at.txt"
exit "$CODE"
