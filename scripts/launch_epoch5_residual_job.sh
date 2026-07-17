#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "openvla" && "$MODE" != "smolvla" ]]; then
  echo "usage: $0 {openvla|smolvla}" >&2
  exit 2
fi

cd /mnt/c/Users/jiheo/tca_map

OPENVLA_PYTHON_BIN="/home/jiheon/venvs/openvla-oft-int4-rtx5080/bin/python"
SMOLVLA_PYTHON_BIN="/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python"
JOBDIR="runs/openvla_oft_int4/epoch5_libero10_residual_v1"
TASK_SPECS="libero_10:8:epoch5_two_moka_pots,libero_10:9:epoch5_microwave_close"
RESET_IDENTITIES="20260716,20260717,20260718,20260719,20260720,20260721,20260722,20260723"
MANIFEST_LABEL="epoch5_libero10_residual_v1"

mkdir -p "$JOBDIR"

if [[ "$MODE" == "openvla" ]]; then
  CMD=(
    /usr/bin/time -v
    "$OPENVLA_PYTHON_BIN"
    -m tca_map.openvla_oft_int4_gate
    hard-slice-rollout
    --load-in-4bit
    --task-specs "$TASK_SPECS"
    --reset-identities "$RESET_IDENTITIES"
    --manifest-label "$MANIFEST_LABEL"
    --out runs/openvla_oft_int4/epoch5_libero10_residual_openvla_int4.json
  )
else
  CMD=(
    /usr/bin/time -v
    "$SMOLVLA_PYTHON_BIN"
    -m tca_map.smolvla.exact_hard_slice_rollout
    rollout
    --task-specs "$TASK_SPECS"
    --reset-identities "$RESET_IDENTITIES"
    --manifest-label "$MANIFEST_LABEL"
    --video-dir runs/openvla_oft_int4/epoch5_libero10_residual_smolvla_exact_videos
    --out runs/openvla_oft_int4/epoch5_libero10_residual_smolvla_exact.json
  )
fi

printf "%q " "${CMD[@]}" > "$JOBDIR/${MODE}_resume_command.txt"
printf "\n" >> "$JOBDIR/${MODE}_resume_command.txt"
rm -f "$JOBDIR/${MODE}_exit_code.txt" "$JOBDIR/${MODE}_finished_at.txt"
: > "$JOBDIR/${MODE}_stdout.log"
: > "$JOBDIR/${MODE}_stderr.log"

nohup bash -c '
jobdir="$1"
mode="$2"
shift 2
echo "$$" > "$jobdir/${mode}_worker_pid.txt"
while true; do date -Is > "$jobdir/${mode}_heartbeat.txt"; sleep 60; done &
heartbeat_pid="$!"
"$@"
code="$?"
echo "$code" > "$jobdir/${mode}_exit_code.txt"
date -Is > "$jobdir/${mode}_finished_at.txt"
kill "$heartbeat_pid" 2>/dev/null || true
exit "$code"
' bash "$JOBDIR" "$MODE" "${CMD[@]}" > "$JOBDIR/${MODE}_stdout.log" 2> "$JOBDIR/${MODE}_stderr.log" &

echo "$!" > "$JOBDIR/${MODE}_launcher_pid.txt"
cat "$JOBDIR/${MODE}_launcher_pid.txt"
