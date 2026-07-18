#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-runs/xvla_prior/ocr_trace_acquisition_task5_discovery_$(date +%Y%m%dT%H%M%SKST)}"

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
TRACE_RGB_SIZE="64"
IDENTITIES=(20260727 20260730 20260731 20260732 20260733)

cd "$REPO"
mkdir -p "$OUTPUT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"

cat > "$OUTPUT_ROOT/trace_acquisition_manifest.json" <<EOF
{
  "schema_version": "2026-07-18.epoch5_ocr_trace_acquisition_worker.v1",
  "stage": "epoch_5_ocr_xvla_task5_bounded_no_training_trace_acquisition",
  "policy": "X-VLA-Libero",
  "git_commit": "${GIT_COMMIT}",
  "task_suite": "${TASK_SUITE}",
  "task_id": ${TASK_ID},
  "task_description": "${TASK_DESCRIPTION}",
  "discovery_identities": [20260727, 20260730, 20260731, 20260732, 20260733],
  "residual_failure_discovery_identities": [20260727, 20260730, 20260733],
  "clean_retention_discovery_identities": [20260731, 20260732],
  "held_out_confirmatory_identities_not_used": [20260734, 20260735, 20260736, 20260737],
  "episode_count": 5,
  "identity_base": ${IDENTITY_BASE},
  "eval_horizon": ${EVAL_HORIZON},
  "settle_steps": ${SETTLE_STEPS},
  "denoise_steps": ${DENOISE_STEPS},
  "trace_rgb_size": ${TRACE_RGB_SIZE},
  "training_happened": false,
  "optimizer_step_happened": false,
  "checkpoint_written": false,
  "ours_rollout_happened": false,
  "control_rollout_happened": false,
  "broad_identity_sweep_happened": false,
  "purpose": "one bounded trace-acquisition pass to test OCR trigger observability from legal prior traces only"
}
EOF

cat > "$OUTPUT_ROOT/exact_resume_command.txt" <<EOF
wsl -d Ubuntu-22.04 --cd "$REPO" -- bash scripts/run_xvla_ocr_trace_acquisition_worker.sh "$OUTPUT_ROOT"
EOF

echo "$$" > "$OUTPUT_ROOT/trace_worker_pid.txt"
rm -f "$OUTPUT_ROOT/trace_exit_code.txt" "$OUTPUT_ROOT/trace_finished_at.txt"
: > "$OUTPUT_ROOT/trace_stdout.log"
: > "$OUTPUT_ROOT/trace_stderr.log"

trace_exit=0
for identity in "${IDENTITIES[@]}"; do
  date -Is > "$OUTPUT_ROOT/trace_heartbeat.txt"
  echo "running identity=$identity task_id=$TASK_ID" > "$OUTPUT_ROOT/trace_status.txt"
  identity_dir="$OUTPUT_ROOT/identity_${identity}"
  mkdir -p "$identity_dir"
  {
    echo "identity=$identity task_id=$TASK_ID"
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
      --trace-dir "$identity_dir/legal_trace" \
      --trace-rgb-size "$TRACE_RGB_SIZE"
    code="$?"
    echo "$code" > "$identity_dir/exit_code.txt"
    if [[ "$code" -ne 0 ]]; then
      trace_exit=1
    fi
  } > "$identity_dir/stdout.log" 2> "$identity_dir/stderr.log" || {
    code="$?"
    echo "$code" > "$identity_dir/exit_code.txt"
    trace_exit=1
  }
done

date -Is > "$OUTPUT_ROOT/trace_heartbeat.txt"
echo "summarizing" > "$OUTPUT_ROOT/trace_status.txt"
"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys

jobdir = pathlib.Path(sys.argv[1])
rows = []
for result_path in sorted(jobdir.glob("identity_*/result.json")):
    identity_dir = result_path.parent
    try:
        obj = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        rows.append(
            {
                "identity_dir": str(identity_dir),
                "result_path": str(result_path),
                "parse_error": f"{type(exc).__name__}: {exc}",
            }
        )
        continue
    episode = (obj.get("episodes") or [{}])[0]
    trace_artifact = episode.get("trace_artifact")
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
            "decision": obj.get("decision"),
            "trace_artifact": trace_artifact,
            "elapsed_seconds": obj.get("elapsed_seconds"),
        }
    )
summary = {
    "schema_version": "2026-07-18.epoch5_ocr_trace_acquisition_summary.v1",
    "jobdir": str(jobdir),
    "policy": "X-VLA-Libero",
    "task_suite": "libero_spatial",
    "task_id": 5,
    "discovery_identity_count": len(rows),
    "completed_episode_count": sum(1 for row in rows if row.get("completed")),
    "successful_episode_count": sum(1 for row in rows if row.get("success")),
    "infrastructure_failure_count": sum(1 for row in rows if row.get("infrastructure_failure_count")),
    "trace_episode_count": sum(1 for row in rows if row.get("trace_artifact")),
    "training_happened": False,
    "optimizer_step_happened": False,
    "checkpoint_written": False,
    "ours_rollout_happened": False,
    "control_rollout_happened": False,
    "broad_identity_sweep_happened": False,
    "rows": rows,
}
payload = json.dumps(summary, indent=2, sort_keys=True)
(jobdir / "trace_acquisition_summary.json").write_text(payload, encoding="utf-8")
(jobdir / "trace_acquisition_summary.sha256").write_text(
    hashlib.sha256(payload.encode("utf-8")).hexdigest() + "\n",
    encoding="utf-8",
)
PY

date -Is > "$OUTPUT_ROOT/trace_finished_at.txt"
echo "$trace_exit" > "$OUTPUT_ROOT/trace_exit_code.txt"
if [[ "$trace_exit" -eq 0 ]]; then
  echo "complete" > "$OUTPUT_ROOT/trace_status.txt"
else
  echo "complete_with_episode_errors" > "$OUTPUT_ROOT/trace_status.txt"
fi
exit "$trace_exit"
