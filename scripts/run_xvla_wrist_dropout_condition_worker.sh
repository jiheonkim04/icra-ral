#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-runs/xvla_prior/claim_condition_wrist_dropout_task5_discovery_$(date +%Y%m%dT%H%M%SKST)}"

REPO="/mnt/c/Users/jiheo/tca_map"
PYTHON_BIN="/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python"
RUNNER="scripts/epoch5_xvla_libero10_task8_eval.py"
TASK_SUITE="libero_spatial"
TASK_ID="5"
TASK_DESCRIPTION="pick up the black bowl on the ramekin and place it on the plate"
IDENTITY_BASE="20260711"
EVAL_HORIZON="900"
SETTLE_STEPS="10"
DENOISE_STEPS="10"
PERTURBATION="wrist_blackout"
IDENTITIES=(20260731 20260732)

cd "$REPO"
mkdir -p "$OUTPUT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"

cat > "$OUTPUT_ROOT/wrist_dropout_manifest.json" <<EOF
{
  "schema_version": "2026-07-18.epoch5_wrist_dropout_condition_worker.v1",
  "stage": "epoch_5_claim_specific_condition_wrist_camera_dropout_prior_verification",
  "policy": "X-VLA-Libero",
  "git_commit": "${GIT_COMMIT}",
  "task_suite": "${TASK_SUITE}",
  "task_id": ${TASK_ID},
  "task_description": "${TASK_DESCRIPTION}",
  "discovery_identities": [20260731, 20260732],
  "clean_baseline_artifact": "reports/ocr_xvla_trace_observability_result.json",
  "perturbation": "${PERTURBATION}",
  "episode_count": 2,
  "identity_base": ${IDENTITY_BASE},
  "eval_horizon": ${EVAL_HORIZON},
  "settle_steps": ${SETTLE_STEPS},
  "denoise_steps": ${DENOISE_STEPS},
  "training_happened": false,
  "optimizer_step_happened": false,
  "checkpoint_written": false,
  "ours_rollout_happened": false,
  "control_rollout_happened": false,
  "broad_identity_sweep_happened": false,
  "purpose": "verify official-prior degradation under one preregistered claim-specific partial-observation condition before selecting any Ours method"
}
EOF

cat > "$OUTPUT_ROOT/exact_resume_command.txt" <<EOF
wsl -d Ubuntu-22.04 --cd "$REPO" -- bash scripts/run_xvla_wrist_dropout_condition_worker.sh "$OUTPUT_ROOT"
EOF

echo "$$" > "$OUTPUT_ROOT/dropout_worker_pid.txt"
rm -f "$OUTPUT_ROOT/dropout_exit_code.txt" "$OUTPUT_ROOT/dropout_finished_at.txt"
: > "$OUTPUT_ROOT/dropout_stdout.log"
: > "$OUTPUT_ROOT/dropout_stderr.log"

dropout_exit=0
for identity in "${IDENTITIES[@]}"; do
  date -Is > "$OUTPUT_ROOT/dropout_heartbeat.txt"
  echo "running identity=$identity task_id=$TASK_ID perturbation=$PERTURBATION" > "$OUTPUT_ROOT/dropout_status.txt"
  identity_dir="$OUTPUT_ROOT/identity_${identity}"
  mkdir -p "$identity_dir"
  {
    echo "identity=$identity task_id=$TASK_ID perturbation=$PERTURBATION"
    date -Is
    "$PYTHON_BIN" "$RUNNER" \
      --run-dir "$identity_dir" \
      --identities "$identity" \
      --task-suite "$TASK_SUITE" \
      --task-id "$TASK_ID" \
      --task-description "$TASK_DESCRIPTION" \
      --identity-base "$IDENTITY_BASE" \
      --eval-horizon "$EVAL_HORIZON" \
      --settle-steps "$SETTLE_STEPS" \
      --denoise-steps "$DENOISE_STEPS" \
      --rgb-input-perturbation "$PERTURBATION"
    code="$?"
    echo "$code" > "$identity_dir/exit_code.txt"
    if [[ "$code" -ne 0 ]]; then
      dropout_exit=1
    fi
  } > "$identity_dir/stdout.log" 2> "$identity_dir/stderr.log" || {
    code="$?"
    echo "$code" > "$identity_dir/exit_code.txt"
    dropout_exit=1
  }
done

date -Is > "$OUTPUT_ROOT/dropout_heartbeat.txt"
echo "summarizing" > "$OUTPUT_ROOT/dropout_status.txt"
"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys

jobdir = pathlib.Path(sys.argv[1])
rows = []
for result_path in sorted(jobdir.glob("identity_*/result.json")):
    identity_dir = result_path.parent
    obj = json.loads(result_path.read_text(encoding="utf-8"))
    episode = (obj.get("episodes") or [{}])[0]
    rows.append(
        {
            "identity_dir": str(identity_dir),
            "result_path": str(result_path),
            "task_suite": obj.get("task_suite"),
            "task_id": obj.get("task_id"),
            "reset_identity": episode.get("reset_identity"),
            "completed": episode.get("completed"),
            "success": episode.get("success"),
            "steps": episode.get("steps"),
            "action_chunk_count": episode.get("action_chunk_count"),
            "infrastructure_failure_count": obj.get("infrastructure_failure_count"),
            "input_perturbation": obj.get("input_perturbation"),
            "decision": obj.get("decision"),
            "elapsed_seconds": obj.get("elapsed_seconds"),
        }
    )
summary = {
    "schema_version": "2026-07-18.epoch5_wrist_dropout_condition_summary.v1",
    "jobdir": str(jobdir),
    "policy": "X-VLA-Libero",
    "task_suite": "libero_spatial",
    "task_id": 5,
    "perturbation": "wrist_blackout",
    "discovery_identity_count": len(rows),
    "completed_episode_count": sum(1 for row in rows if row.get("completed")),
    "successful_episode_count": sum(1 for row in rows if row.get("success")),
    "infrastructure_failure_count": sum(1 for row in rows if row.get("infrastructure_failure_count")),
    "training_happened": False,
    "optimizer_step_happened": False,
    "checkpoint_written": False,
    "ours_rollout_happened": False,
    "control_rollout_happened": False,
    "broad_identity_sweep_happened": False,
    "rows": rows,
}
payload = json.dumps(summary, indent=2, sort_keys=True)
(jobdir / "wrist_dropout_summary.json").write_text(payload, encoding="utf-8")
(jobdir / "wrist_dropout_summary.sha256").write_text(
    hashlib.sha256(payload.encode("utf-8")).hexdigest() + "\n",
    encoding="utf-8",
)
PY

date -Is > "$OUTPUT_ROOT/dropout_finished_at.txt"
echo "$dropout_exit" > "$OUTPUT_ROOT/dropout_exit_code.txt"
if [[ "$dropout_exit" -eq 0 ]]; then
  echo "complete" > "$OUTPUT_ROOT/dropout_status.txt"
else
  echo "complete_with_episode_errors" > "$OUTPUT_ROOT/dropout_status.txt"
fi
exit "$dropout_exit"
