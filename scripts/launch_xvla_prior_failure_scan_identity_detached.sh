#!/usr/bin/env bash
set -euo pipefail

IDENTITY="${1:-}"
if [[ -z "$IDENTITY" ]]; then
  echo "usage: $0 RESET_IDENTITY [OUTPUT_ROOT] [START_TASK_ID] [TASK_COUNT]" >&2
  exit 2
fi

OUTPUT_ROOT="${2:-runs/xvla_prior/failure_scan_libero10_identity${IDENTITY}_$(date +%Y%m%dT%H%M%SKST)}"
START_TASK_ID="${3:-0}"
TASK_COUNT="${4:-10}"

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
  "schema_version": "2026-07-17.epoch5_xvla_prior_failure_scan_identity.v1",
  "stage": "epoch_5_xvla_prior_residual_mining_after_br_xvla_no_pass",
  "policy": "X-VLA-Libero",
  "git_commit": "${GIT_COMMIT}",
  "reset_identity": ${IDENTITY},
  "task_suite": "libero_10",
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
  "retuning_from_br_xvla_result_allowed": false,
  "purpose": "official-prior residual mining only; do not treat prior failures or successes as Ours"
}
EOF

cat > "$OUTPUT_ROOT/exact_resume_command.txt" <<EOF
wsl -d Ubuntu-22.04 --cd "$REPO" -- bash scripts/launch_xvla_prior_failure_scan_identity_detached.sh "$IDENTITY" "$OUTPUT_ROOT" "$START_TASK_ID" "$TASK_COUNT"
EOF

rm -f "$OUTPUT_ROOT/scan_exit_code.txt" "$OUTPUT_ROOT/scan_finished_at.txt"
: > "$OUTPUT_ROOT/scan_stdout.log"
: > "$OUTPUT_ROOT/scan_stderr.log"

nohup bash -c '
set -euo pipefail
jobdir="$1"
identity="$2"
start_task="$3"
task_count="$4"
repo="$5"
python_bin="$6"
runner="$7"
identity_base="$8"
eval_horizon="$9"
settle_steps="${10}"
denoise_steps="${11}"

echo "$$" > "$jobdir/scan_worker_pid.txt"
while true; do date -Is > "$jobdir/scan_heartbeat.txt"; sleep 60; done &
heartbeat_pid="$!"

scan_exit=0
end_task=$((start_task + task_count - 1))
for task_id in $(seq "$start_task" "$end_task"); do
  task_dir="$jobdir/task_${task_id}"
  mkdir -p "$task_dir"
  {
    echo "task_id=$task_id"
    date -Is
    "$python_bin" "$runner" \
      --run-dir "$task_dir" \
      --identities "$identity" \
      --task-suite libero_10 \
      --task-id "$task_id" \
      --identity-base "$identity_base" \
      --eval-horizon "$eval_horizon" \
      --settle-steps "$settle_steps" \
      --denoise-steps "$denoise_steps"
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

"$python_bin" - "$jobdir" <<'"'"'PY'"'"'
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
        rows.append({
            "task_dir": str(task_dir),
            "result_path": str(result_path),
            "parse_error": f"{type(exc).__name__}: {exc}",
        })
        continue
    rows.append({
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
    })
summary = {
    "schema_version": "2026-07-17.epoch5_xvla_prior_failure_scan_identity_summary.v1",
    "jobdir": str(jobdir),
    "policy": "X-VLA-Libero",
    "training_happened": False,
    "optimizer_step_happened": False,
    "checkpoint_written": False,
    "closed_loop_ours_evaluation_happened": False,
    "task_count": len(rows),
    "completed_task_count": sum(1 for row in rows if row.get("completed_episode_count") == 1),
    "successful_task_count": sum(1 for row in rows if row.get("successful_episode_count") == 1),
    "failure_task_ids": [row.get("task_id") for row in rows if row.get("completed_episode_count") == 1 and row.get("successful_episode_count") == 0],
    "infrastructure_failure_task_ids": [row.get("task_id") for row in rows if row.get("infrastructure_failure_count")],
    "rows": rows,
}
(jobdir / "scan_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
PY

date -Is > "$jobdir/scan_finished_at.txt"
echo "$scan_exit" > "$jobdir/scan_exit_code.txt"
kill "$heartbeat_pid" 2>/dev/null || true
exit "$scan_exit"
' bash "$OUTPUT_ROOT" "$IDENTITY" "$START_TASK_ID" "$TASK_COUNT" "$REPO" "$PYTHON_BIN" "$RUNNER" "$IDENTITY_BASE" "$EVAL_HORIZON" "$SETTLE_STEPS" "$DENOISE_STEPS" \
  > "$OUTPUT_ROOT/scan_stdout.log" 2> "$OUTPUT_ROOT/scan_stderr.log" &

echo "$!" > "$OUTPUT_ROOT/scan_launcher_pid.txt"
cat "$OUTPUT_ROOT/scan_launcher_pid.txt"
