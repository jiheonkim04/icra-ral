#!/usr/bin/env bash
set -euo pipefail

mode="${1:?mode required}"
run_id="${2:?run id required}"

export PATH="/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin:${PATH}"
# The public prior's 2-D image projection was produced at the author's
# immediately preceding c197a01 architecture and does not load against the
# later 4-D Conv2d source. Use that exact author tree; the runner verifies it.
export PYTHONPATH="/mnt/c/Users/jiheo/tca_map:/home/jiheon/assets/repos/a2c2-libero-checkpoint-compat-c197a01/src"
export HF_HOME="/mnt/c/assets/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH="/home/jiheon/.libero"
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

exec /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python \
  /mnt/c/Users/jiheo/tca_map/scripts/run_a2c2_fidelity_corrected.py \
  --mode "${mode}" \
  --run-id "${run_id}"
