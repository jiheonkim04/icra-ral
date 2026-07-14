#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/c/Users/jiheo/tca_map}"
PYTHON_BIN="${PYTHON_BIN:-/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/eac_vla_stage_a}"
PARTIAL_RESULT="reports/eac_vla/stage_a_partial_result.json"
FINAL_RESULT="reports/eac_vla/stage_a_result.json"
RUNNER_VALIDATION="reports/eac_vla/stage_a_runner_validation.json"
REPORT_STATUS="reports/eac_vla/stage_a_status.json"

cd "$REPO_ROOT"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_ROOT}/${timestamp}"
mkdir -p "$run_dir"

resume_command="cd ${REPO_ROOT} && ${PYTHON_BIN} scripts/run_eac_vla_stage_a.py --mode stage-a"
printf '%s\n' "$resume_command" > "${run_dir}/resume_command.txt"

partial_count() {
  "$PYTHON_BIN" -c 'import json, pathlib
p = pathlib.Path("reports/eac_vla/stage_a_partial_result.json")
print(len(json.loads(p.read_text()).get("episodes", [])) if p.exists() else 0)' 2>/dev/null || printf '0\n'
}

write_status() {
  local status="$1"
  local exit_code="${2:-null}"
  local child_pid="${3:-null}"
  local count
  count="$(partial_count)"
  cat > "${run_dir}/status.json" <<STATUS
{
  "status": "${status}",
  "updated_utc": "$(date -u +%FT%TZ)",
  "wrapper_pid": $$,
  "child_pid": ${child_pid},
  "exit_code": ${exit_code},
  "planned_episode_count": 50,
  "partial_episode_count": ${count},
  "partial_result": "${PARTIAL_RESULT}",
  "final_result": "${FINAL_RESULT}",
  "runner_validation": "${RUNNER_VALIDATION}",
  "report_status": "${REPORT_STATUS}",
  "resume_command_file": "${run_dir}/resume_command.txt"
}
STATUS
  cp "${run_dir}/status.json" "${run_dir}/heartbeat.json"
}

cat > "${run_dir}/launch.json" <<LAUNCH
{
  "status": "launching",
  "updated_utc": "$(date -u +%FT%TZ)",
  "planned_episode_count": 50,
  "partial_result": "${PARTIAL_RESULT}",
  "final_result": "${FINAL_RESULT}",
  "runner_validation": "${RUNNER_VALIDATION}",
  "resume_command": "${resume_command}"
}
LAUNCH

nohup bash -c '
set -euo pipefail
repo_root="$1"
python_bin="$2"
run_dir="$3"
cd "$repo_root"

partial_count() {
  "$python_bin" -c '"'"'import json, pathlib
p = pathlib.Path("reports/eac_vla/stage_a_partial_result.json")
print(len(json.loads(p.read_text()).get("episodes", [])) if p.exists() else 0)'"'"' 2>/dev/null || printf '"'"'0\n'"'"'
}

write_status() {
  local status="$1"
  local exit_code="${2:-null}"
  local child_pid="${3:-null}"
  local count
  count="$(partial_count)"
  cat > "${run_dir}/status.json" <<STATUS
{
  "status": "${status}",
  "updated_utc": "$(date -u +%FT%TZ)",
  "wrapper_pid": $$,
  "child_pid": ${child_pid},
  "exit_code": ${exit_code},
  "planned_episode_count": 50,
  "partial_episode_count": ${count},
  "partial_result": "reports/eac_vla/stage_a_partial_result.json",
  "final_result": "reports/eac_vla/stage_a_result.json",
  "runner_validation": "reports/eac_vla/stage_a_runner_validation.json",
  "report_status": "reports/eac_vla/stage_a_status.json",
  "resume_command_file": "${run_dir}/resume_command.txt"
}
STATUS
  cp "${run_dir}/status.json" "${run_dir}/heartbeat.json"
}

write_status "starting" null null
"$python_bin" scripts/run_eac_vla_stage_a.py --mode stage-a --stage-a-status-output reports/eac_vla/stage_a_status.json &
child_pid=$!
printf "%s\n" "$child_pid" > "${run_dir}/child_pid.txt"
write_status "running" null "$child_pid"

while jobs -r -p | grep -qx "$child_pid"; do
  write_status "running" null "$child_pid"
  sleep 60
done

set +e
wait "$child_pid"
code=$?
set -e
printf "%s\n" "$code" > "${run_dir}/exit_code.txt"
if [ "$code" -eq 0 ]; then
  write_status "completed" "$code" "$child_pid"
else
  write_status "failed" "$code" "$child_pid"
fi
exit "$code"
' eac_stage_a "$REPO_ROOT" "$PYTHON_BIN" "$run_dir" > "${run_dir}/stdout.log" 2> "${run_dir}/stderr.log" < /dev/null &

wrapper_pid=$!
printf '%s\n' "$wrapper_pid" > "${run_dir}/pid.txt"
write_status "launched" null null

for _ in $(seq 1 20); do
  if [ -s "${run_dir}/child_pid.txt" ]; then
    break
  fi
  sleep 1
done

child_pid="null"
if [ -s "${run_dir}/child_pid.txt" ]; then
  child_pid="$(cat "${run_dir}/child_pid.txt")"
fi

cat <<EOF
RUN_DIR=${run_dir}
WRAPPER_PID=${wrapper_pid}
CHILD_PID=${child_pid}
PARTIAL_RESULT=${PARTIAL_RESULT}
FINAL_RESULT=${FINAL_RESULT}
RUNNER_VALIDATION=${RUNNER_VALIDATION}
STDOUT=${run_dir}/stdout.log
STDERR=${run_dir}/stderr.log
STATUS=${run_dir}/status.json
HEARTBEAT=${run_dir}/heartbeat.json
RESUME_COMMAND_FILE=${run_dir}/resume_command.txt
EOF
