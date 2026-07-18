#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-runs/xvla_prior/awf_xvla_stage0_wrist_dropout_task5_discovery_$(date +%Y%m%dT%H%M%SKST)}"

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
MITIGATION="agentview_fill"
IDENTITIES=(20260731 20260732)

cd "$REPO"
mkdir -p "$OUTPUT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"

cat > "$OUTPUT_ROOT/awf_stage0_manifest.json" <<EOF
{
  "schema_version": "2026-07-18.epoch5_awf_xvla_stage0_worker.v1",
  "stage": "epoch_5_awf_xvla_stage0_wrist_dropout_discovery",
  "method": "AWF-XVLA",
  "method_name": "Agentview-Wrist Fill for X-VLA",
  "git_commit": "${GIT_COMMIT}",
  "task_suite": "${TASK_SUITE}",
  "task_id": ${TASK_ID},
  "task_description": "${TASK_DESCRIPTION}",
  "discovery_identities": [20260731, 20260732],
  "condition": "${PERTURBATION}",
  "mitigation": "${MITIGATION}",
  "episode_count": 2,
  "identity_base": ${IDENTITY_BASE},
  "eval_horizon": ${EVAL_HORIZON},
  "settle_steps": ${SETTLE_STEPS},
  "denoise_steps": ${DENOISE_STEPS},
  "baseline_condition_artifact": "reports/wrist_dropout_condition_verification_result.json",
  "training_happened": false,
  "optimizer_step_happened": false,
  "checkpoint_written": false,
  "ours_rollout_happened": true,
  "control_rollout_happened": false,
  "broad_identity_sweep_happened": false,
  "purpose": "Stage 0 empirical test for exactly one no-training inference module under the verified wrist-dropout condition"
}
EOF

cat > "$OUTPUT_ROOT/exact_resume_command.txt" <<EOF
wsl -d Ubuntu-22.04 --cd "$REPO" -- bash scripts/run_awf_xvla_stage0_worker.sh "$OUTPUT_ROOT"
EOF

echo "$$" > "$OUTPUT_ROOT/awf_worker_pid.txt"
rm -f "$OUTPUT_ROOT/awf_exit_code.txt" "$OUTPUT_ROOT/awf_finished_at.txt"
: > "$OUTPUT_ROOT/awf_stdout.log"
: > "$OUTPUT_ROOT/awf_stderr.log"

awf_exit=0
for identity in "${IDENTITIES[@]}"; do
  date -Is > "$OUTPUT_ROOT/awf_heartbeat.txt"
  echo "running identity=$identity task_id=$TASK_ID condition=$PERTURBATION mitigation=$MITIGATION" > "$OUTPUT_ROOT/awf_status.txt"
  identity_dir="$OUTPUT_ROOT/identity_${identity}"
  mkdir -p "$identity_dir"
  {
    echo "identity=$identity task_id=$TASK_ID condition=$PERTURBATION mitigation=$MITIGATION"
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
      --rgb-input-perturbation "$PERTURBATION" \
      --wrist-dropout-mitigation "$MITIGATION"
    code="$?"
    echo "$code" > "$identity_dir/exit_code.txt"
    if [[ "$code" -ne 0 ]]; then
      awf_exit=1
    fi
  } > "$identity_dir/stdout.log" 2> "$identity_dir/stderr.log" || {
    code="$?"
    echo "$code" > "$identity_dir/exit_code.txt"
    awf_exit=1
  }
done

date -Is > "$OUTPUT_ROOT/awf_heartbeat.txt"
echo "summarizing" > "$OUTPUT_ROOT/awf_status.txt"
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
            "mitigation_triggered_step_count": episode.get("wrist_dropout_mitigation_triggered_step_count"),
            "input_perturbation": obj.get("input_perturbation"),
            "wrist_dropout_mitigation": obj.get("wrist_dropout_mitigation"),
            "training_happened": obj.get("training_happened"),
            "optimizer_step_happened": obj.get("optimizer_step_happened"),
            "checkpoint_written": obj.get("checkpoint_written"),
            "ours_rollout_happened": obj.get("ours_rollout_happened"),
            "decision": obj.get("decision"),
            "elapsed_seconds": obj.get("elapsed_seconds"),
        }
    )
summary = {
    "schema_version": "2026-07-18.epoch5_awf_xvla_stage0_summary.v1",
    "jobdir": str(jobdir),
    "method": "AWF-XVLA",
    "condition": "wrist_blackout",
    "mitigation": "agentview_fill",
    "discovery_identity_count": len(rows),
    "completed_episode_count": sum(1 for row in rows if row.get("completed")),
    "successful_episode_count": sum(1 for row in rows if row.get("success")),
    "training_happened": False,
    "optimizer_step_happened": False,
    "checkpoint_written": False,
    "ours_rollout_happened": True,
    "control_rollout_happened": False,
    "broad_identity_sweep_happened": False,
    "rows": rows,
}
payload = json.dumps(summary, indent=2, sort_keys=True)
(jobdir / "awf_stage0_summary.json").write_text(payload, encoding="utf-8")
(jobdir / "awf_stage0_summary.sha256").write_text(
    hashlib.sha256(payload.encode("utf-8")).hexdigest() + "\n",
    encoding="utf-8",
)
PY

date -Is > "$OUTPUT_ROOT/awf_finished_at.txt"
echo "$awf_exit" > "$OUTPUT_ROOT/awf_exit_code.txt"
if [[ "$awf_exit" -eq 0 ]]; then
  echo "complete" > "$OUTPUT_ROOT/awf_status.txt"
else
  echo "complete_with_episode_errors" > "$OUTPUT_ROOT/awf_status.txt"
fi
exit "$awf_exit"
