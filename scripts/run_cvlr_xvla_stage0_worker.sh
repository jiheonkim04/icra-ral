#!/usr/bin/env bash
set -uo pipefail

RUN_DIR="$1"
PYTHON_BIN="/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python"
CONTRACT="configs/cvlr_xvla_stage0_frozen_contract.json"

mkdir -p "$RUN_DIR"
printf '%s\n' "$$" > "$RUN_DIR/wsl_shell_pid.txt"
printf '%s\n' "$(date --iso-8601=seconds)" > "$RUN_DIR/worker_started_at.txt"
printf '%s\n' "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PYTHON_BIN scripts/run_cvlr_xvla_stage0.py --run-dir $RUN_DIR --contract $CONTRACT" > "$RUN_DIR/exact_command.txt"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl \
  "$PYTHON_BIN" scripts/run_cvlr_xvla_stage0.py --run-dir "$RUN_DIR" --contract "$CONTRACT"
CODE=$?

printf '%s\n' "$CODE" > "$RUN_DIR/shell_exit_code.txt"
printf '%s\n' "$(date --iso-8601=seconds)" > "$RUN_DIR/worker_finished_at.txt"
exit "$CODE"
