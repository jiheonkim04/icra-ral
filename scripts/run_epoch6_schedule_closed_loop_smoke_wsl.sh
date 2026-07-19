#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_DIR HOST_LOCK_PATH" >&2
  exit 2
fi

export EPOCH6_HOST_SMOKE_LOCK_PATH="$2"
source /home/jiheon/miniconda3-official/etc/profile.d/conda.sh
conda activate official-smolvla-libero
cd /mnt/c/Users/jiheo/tca_map
set +e
python scripts/run_epoch6_schedule_closed_loop.py \
  --mode resource-smoke \
  --run-dir "$1" \
  --child
exit_code=$?
set -e
exit_file="$1/closed_loop_resource_smoke_child_exit_code.txt"
printf '%s\n' "$exit_code" > "$exit_file.tmp"
mv "$exit_file.tmp" "$exit_file"
exit "$exit_code"
