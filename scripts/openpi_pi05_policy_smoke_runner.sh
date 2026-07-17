#!/usr/bin/env bash
set +e

OUT_PATH="${1:?missing output JSON path}"
EXIT_CODE_PATH="${2:?missing exit-code path}"

cd /mnt/c/assets/repos/openpi || exit 97
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/home/jiheon/assets/checkpoints/openpi}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

/home/jiheon/venvs/openpi-uv/bin/python \
  /mnt/c/Users/jiheo/tca_map/scripts/openpi_pi05_policy_smoke.py \
  --out "${OUT_PATH}"
ec=$?
echo "${ec}" > "${EXIT_CODE_PATH}"
exit "${ec}"
