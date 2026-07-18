#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-runs/xvla_prior/wrist_dropout_generality_$(date +%Y%m%dT%H%M%SKST)}"

REPO="/mnt/c/Users/jiheo/tca_map"
PYTHON_BIN="/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python"
RUNNER="scripts/epoch5_xvla_libero10_task8_eval.py"
IDENTITY_BASE="20260711"
EVAL_HORIZON="900"
SETTLE_STEPS="10"
DENOISE_STEPS="10"

CASES=(
  "libero_goal|0|20260733|open the middle drawer of the cabinet"
  "libero_goal|0|20260734|open the middle drawer of the cabinet"
  "libero_goal|0|20260735|open the middle drawer of the cabinet"
  "libero_object|0|20260733|pick up the alphabet soup and place it in the basket"
  "libero_object|0|20260734|pick up the alphabet soup and place it in the basket"
  "libero_object|0|20260735|pick up the alphabet soup and place it in the basket"
  "libero_spatial|5|20260731|pick up the black bowl on the ramekin and place it on the plate"
  "libero_spatial|5|20260732|pick up the black bowl on the ramekin and place it on the plate"
  "libero_spatial|5|20260735|pick up the black bowl on the ramekin and place it on the plate"
)

cd "$REPO"
mkdir -p "$OUTPUT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"

cat > "$OUTPUT_ROOT/generality_manifest.json" <<EOF
{
  "schema_version": "2026-07-18.epoch5_wrist_dropout_generality_worker.v1",
  "stage": "epoch_5_wrist_dropout_condition_generality_frozen_prior_only",
  "policy": "X-VLA-Libero",
  "git_commit": "${GIT_COMMIT}",
  "task_count": 3,
  "reset_identities_per_task": 3,
  "clean_episode_count": 9,
  "dropout_episode_count": 9,
  "paired_clean_dropout_identities": true,
  "condition": "wrist_blackout",
  "identity_base": ${IDENTITY_BASE},
  "eval_horizon": ${EVAL_HORIZON},
  "settle_steps": ${SETTLE_STEPS},
  "denoise_steps": ${DENOISE_STEPS},
  "tasks": [
    {
      "suite": "libero_goal",
      "task_id": 0,
      "instruction": "open the middle drawer of the cabinet",
      "identities": [20260733, 20260734, 20260735],
      "existing_clean_evidence": [
        "reports/post_task5_exhausted_libero_goal_20260733_prior_scan_result.json",
        "reports/post_task5_exhausted_libero_goal_20260734_prior_scan_result.json",
        "reports/post_task5_exhausted_libero_goal_20260735_prior_scan_result.json"
      ]
    },
    {
      "suite": "libero_object",
      "task_id": 0,
      "instruction": "pick up the alphabet soup and place it in the basket",
      "identities": [20260733, 20260734, 20260735],
      "existing_clean_evidence": [
        "reports/post_task5_exhausted_libero_object_20260733_prior_scan_result.json",
        "reports/post_task5_exhausted_libero_object_20260734_prior_scan_result.json",
        "reports/post_task5_exhausted_libero_object_20260735_prior_scan_result.json"
      ]
    },
    {
      "suite": "libero_spatial",
      "task_id": 5,
      "instruction": "pick up the black bowl on the ramekin and place it on the plate",
      "identities": [20260731, 20260732, 20260735],
      "existing_clean_evidence": [
        "reports/ocr_xvla_trace_observability_result.json",
        "reports/final_cap_libero_spatial_20260735_convergence_result.json"
      ]
    }
  ],
  "training_happened": false,
  "optimizer_step_happened": false,
  "checkpoint_written": false,
  "ours_rollout_happened": false,
  "control_rollout_happened": false,
  "method_selected": false,
  "broad_natural_reset_sweep_happened": false,
  "purpose": "bounded condition-generality study before any further method design"
}
EOF

cat > "$OUTPUT_ROOT/exact_resume_command.txt" <<EOF
wsl -d Ubuntu-22.04 --cd "$REPO" -- bash scripts/run_xvla_wrist_dropout_generality_worker.sh "$OUTPUT_ROOT"
EOF

echo "$$" > "$OUTPUT_ROOT/generality_worker_pid.txt"
rm -f "$OUTPUT_ROOT/generality_exit_code.txt" "$OUTPUT_ROOT/generality_finished_at.txt"
: > "$OUTPUT_ROOT/generality_stdout.log"
: > "$OUTPUT_ROOT/generality_stderr.log"

study_exit=0
for case in "${CASES[@]}"; do
  IFS="|" read -r suite task_id identity instruction <<< "$case"
  for condition in clean dropout; do
    if [[ "$condition" == "clean" ]]; then
      perturbation="none"
    else
      perturbation="wrist_blackout"
    fi
    date -Is > "$OUTPUT_ROOT/generality_heartbeat.txt"
    echo "running suite=$suite task_id=$task_id identity=$identity condition=$condition" > "$OUTPUT_ROOT/generality_status.txt"
    run_dir="$OUTPUT_ROOT/${suite}_task${task_id}_identity_${identity}_${condition}"
    mkdir -p "$run_dir"
    {
      echo "suite=$suite task_id=$task_id identity=$identity condition=$condition perturbation=$perturbation"
      date -Is
      "$PYTHON_BIN" "$RUNNER" \
        --run-dir "$run_dir" \
        --identities "$identity" \
        --task-suite "$suite" \
        --task-id "$task_id" \
        --task-description "$instruction" \
        --identity-base "$IDENTITY_BASE" \
        --eval-horizon "$EVAL_HORIZON" \
        --settle-steps "$SETTLE_STEPS" \
        --denoise-steps "$DENOISE_STEPS" \
        --rgb-input-perturbation "$perturbation"
      code="$?"
      echo "$code" > "$run_dir/exit_code.txt"
      if [[ "$code" -ne 0 ]]; then
        study_exit=1
      fi
    } > "$run_dir/stdout.log" 2> "$run_dir/stderr.log" || {
      code="$?"
      echo "$code" > "$run_dir/exit_code.txt"
      study_exit=1
    }
  done
done

date -Is > "$OUTPUT_ROOT/generality_heartbeat.txt"
echo "summarizing" > "$OUTPUT_ROOT/generality_status.txt"
"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

jobdir = pathlib.Path(sys.argv[1])
rows = []
pattern = re.compile(r"(?P<suite>libero_[a-z]+)_task(?P<task_id>\d+)_identity_(?P<identity>\d+)_(?P<condition>clean|dropout)$")
for result_path in sorted(jobdir.glob("*/result.json")):
    match = pattern.search(result_path.parent.name)
    if not match:
        continue
    obj = json.loads(result_path.read_text(encoding="utf-8"))
    episode = (obj.get("episodes") or [{}])[0]
    rows.append(
        {
            "suite": match.group("suite"),
            "task_id": int(match.group("task_id")),
            "reset_identity": int(match.group("identity")),
            "condition": match.group("condition"),
            "result_path": str(result_path),
            "completed": episode.get("completed"),
            "success": episode.get("success"),
            "steps": episode.get("steps"),
            "action_chunk_count": episode.get("action_chunk_count"),
            "infrastructure_failure_count": obj.get("infrastructure_failure_count"),
            "decision": obj.get("decision"),
            "cuda_name": obj.get("cuda_name"),
            "cuda_memory": episode.get("cuda_memory"),
            "input_perturbation": obj.get("input_perturbation"),
            "elapsed_seconds": obj.get("elapsed_seconds"),
        }
    )
summary = {
    "schema_version": "2026-07-18.epoch5_wrist_dropout_generality_summary.v1",
    "jobdir": str(jobdir),
    "policy": "X-VLA-Libero",
    "condition": "wrist_blackout",
    "row_count": len(rows),
    "clean_episode_count": sum(1 for row in rows if row.get("condition") == "clean"),
    "dropout_episode_count": sum(1 for row in rows if row.get("condition") == "dropout"),
    "completed_episode_count": sum(1 for row in rows if row.get("completed")),
    "successful_episode_count": sum(1 for row in rows if row.get("success")),
    "infrastructure_failure_count": sum(1 for row in rows if row.get("infrastructure_failure_count")),
    "model_forward_count": sum(int(row.get("action_chunk_count") or 0) for row in rows),
    "cuda_devices": sorted({str(row.get("cuda_name")) for row in rows if row.get("cuda_name")}),
    "peak_cuda_max_allocated_mib": max(
        [
            float((row.get("cuda_memory") or {}).get("max_allocated_mib") or 0.0)
            for row in rows
        ]
        or [0.0]
    ),
    "training_happened": False,
    "optimizer_step_happened": False,
    "checkpoint_written": False,
    "ours_rollout_happened": False,
    "control_rollout_happened": False,
    "method_selected": False,
    "broad_natural_reset_sweep_happened": False,
    "rows": rows,
}
payload = json.dumps(summary, indent=2, sort_keys=True)
(jobdir / "generality_summary.json").write_text(payload, encoding="utf-8")
(jobdir / "generality_summary.sha256").write_text(
    hashlib.sha256(payload.encode("utf-8")).hexdigest() + "\n",
    encoding="utf-8",
)
PY

date -Is > "$OUTPUT_ROOT/generality_finished_at.txt"
echo "$study_exit" > "$OUTPUT_ROOT/generality_exit_code.txt"
if [[ "$study_exit" -eq 0 ]]; then
  echo "complete" > "$OUTPUT_ROOT/generality_status.txt"
else
  echo "complete_with_episode_errors" > "$OUTPUT_ROOT/generality_status.txt"
fi
exit "$study_exit"
