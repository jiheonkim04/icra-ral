#!/usr/bin/env bash
set -euo pipefail

IDENTITY="${1:-}"
if [[ -z "$IDENTITY" ]]; then
  echo "usage: $0 RESET_IDENTITY [OUTPUT_ROOT] [START_TASK_ID] [TASK_COUNT] [TASK_SUITE]" >&2
  exit 2
fi

OUTPUT_ROOT="${2:-runs/xvla_prior/failure_scan_libero10_identity${IDENTITY}_$(date +%Y%m%dT%H%M%SKST)}"
START_TASK_ID="${3:-0}"
TASK_COUNT="${4:-10}"
TASK_SUITE="${5:-libero_10}"

REPO="/mnt/c/Users/jiheo/tca_map"
PYTHON_BIN="/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python"
RUNNER="scripts/epoch5_xvla_libero10_task8_eval.py"
IDENTITY_BASE="20260711"
EVAL_HORIZON="900"
SETTLE_STEPS="10"
DENOISE_STEPS="10"

cd "$REPO"
mkdir -p "$OUTPUT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"

cat > "$OUTPUT_ROOT/scan_manifest.json" <<EOF
{
  "schema_version": "2026-07-17.epoch5_xvla_prior_failure_scan_identity.v2",
  "stage": "epoch_5_official_prior_residual_mining_after_mpr_no_pass",
  "policy": "X-VLA-Libero",
  "git_commit": "${GIT_COMMIT}",
  "reset_identity": ${IDENTITY},
  "task_suite": "${TASK_SUITE}",
  "start_task_id": ${START_TASK_ID},
  "task_count": ${TASK_COUNT},
  "identity_base": ${IDENTITY_BASE},
  "eval_horizon": ${EVAL_HORIZON},
  "settle_steps": ${SETTLE_STEPS},
  "denoise_steps": ${DENOISE_STEPS},
  "training_happened": false,
  "optimizer_step_happened": false,
  "checkpoint_written": false,
  "closed_loop_ours_evaluation_happened": false,
  "retuning_from_mpr_xvla_result_allowed": false,
  "purpose": "official-prior residual mining only; do not treat prior failures or successes as Ours"
}
EOF

cat > "$OUTPUT_ROOT/exact_resume_command.txt" <<EOF
wsl -d Ubuntu-22.04 --cd "$REPO" -- bash scripts/run_xvla_prior_failure_scan_identity_worker.sh "$IDENTITY" "$OUTPUT_ROOT" "$START_TASK_ID" "$TASK_COUNT" "$TASK_SUITE"
EOF

echo "$$" > "$OUTPUT_ROOT/scan_worker_pid.txt"
rm -f "$OUTPUT_ROOT/scan_exit_code.txt" "$OUTPUT_ROOT/scan_finished_at.txt"
: > "$OUTPUT_ROOT/scan_stdout.log"
: > "$OUTPUT_ROOT/scan_stderr.log"

scan_exit=0
end_task=$((START_TASK_ID + TASK_COUNT - 1))
for task_id in $(seq "$START_TASK_ID" "$end_task"); do
  date -Is > "$OUTPUT_ROOT/scan_heartbeat.txt"
  echo "running task_id=$task_id" > "$OUTPUT_ROOT/scan_status.txt"
  task_dir="$OUTPUT_ROOT/task_${task_id}"
  mkdir -p "$task_dir"
  {
    echo "task_id=$task_id"
    date -Is
    "$PYTHON_BIN" "$RUNNER" \
      --run-dir "$task_dir" \
      --identities "$IDENTITY" \
      --task-suite "$TASK_SUITE" \
      --task-id "$task_id" \
      --identity-base "$IDENTITY_BASE" \
      --eval-horizon "$EVAL_HORIZON" \
      --settle-steps "$SETTLE_STEPS" \
      --denoise-steps "$DENOISE_STEPS"
    code="$?"
    echo "$code" > "$task_dir/exit_code.txt"
    if [[ "$code" -ne 0 ]]; then
      scan_exit=1
    fi
  } > "$task_dir/stdout.log" 2> "$task_dir/stderr.log" || {
    code="$?"
    echo "$code" > "$task_dir/exit_code.txt"
    scan_exit=1
  }
done

date -Is > "$OUTPUT_ROOT/scan_heartbeat.txt"
echo "summarizing" > "$OUTPUT_ROOT/scan_status.txt"
"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import json
import pathlib
import sys

jobdir = pathlib.Path(sys.argv[1])
rows = []
for result_path in sorted(jobdir.glob("task_*/result.json")):
    task_dir = result_path.parent
    try:
        obj = json.loads(result_path.read_text())
    except Exception as exc:
        rows.append(
            {
                "task_dir": str(task_dir),
                "result_path": str(result_path),
                "parse_error": f"{type(exc).__name__}: {exc}",
            }
        )
        continue
    rows.append(
        {
            "task_dir": str(task_dir),
            "result_path": str(result_path),
            "task_id": obj.get("task_id"),
            "task_description": obj.get("task_description"),
            "completed_episode_count": obj.get("completed_episode_count"),
            "successful_episode_count": obj.get("successful_episode_count"),
            "infrastructure_failure_count": obj.get("infrastructure_failure_count"),
            "failures": obj.get("failures"),
            "decision": obj.get("decision"),
            "elapsed_seconds": obj.get("elapsed_seconds"),
        }
    )
summary = {
    "schema_version": "2026-07-17.epoch5_xvla_prior_failure_scan_identity_summary.v2",
    "jobdir": str(jobdir),
    "policy": "X-VLA-Libero",
    "training_happened": False,
    "optimizer_step_happened": False,
    "checkpoint_written": False,
    "closed_loop_ours_evaluation_happened": False,
    "task_count": len(rows),
    "completed_task_count": sum(1 for row in rows if row.get("completed_episode_count") == 1),
    "successful_task_count": sum(1 for row in rows if row.get("successful_episode_count") == 1),
    "failure_task_ids": [
        row.get("task_id")
        for row in rows
        if row.get("completed_episode_count") == 1 and row.get("successful_episode_count") == 0
    ],
    "infrastructure_failure_task_ids": [row.get("task_id") for row in rows if row.get("infrastructure_failure_count")],
    "rows": rows,
}
(jobdir / "scan_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
PY

date -Is > "$OUTPUT_ROOT/scan_finished_at.txt"
echo "$scan_exit" > "$OUTPUT_ROOT/scan_exit_code.txt"
if [[ "$scan_exit" -eq 0 ]]; then
  echo "complete" > "$OUTPUT_ROOT/scan_status.txt"
else
  echo "complete_with_task_errors" > "$OUTPUT_ROOT/scan_status.txt"
fi
exit "$scan_exit"
