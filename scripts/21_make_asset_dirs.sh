#!/usr/bin/env bash
set -u

ASSET_ROOT="${ASSET_ROOT:-C:/assets}"
ALLOW_CREATE_DIRS="${ALLOW_CREATE_DIRS:-0}"
DIRS=(
  "$ASSET_ROOT/checkpoints"
  "$ASSET_ROOT/data"
  "$ASSET_ROOT/repos"
  "$ASSET_ROOT/hf_home"
)

echo "TCA-Map asset directory planner"
echo "asset_root: $ASSET_ROOT"
echo "dry_run: $([[ "$ALLOW_CREATE_DIRS" == "1" ]] && echo false || echo true)"
echo "No downloads are performed."

for dir in "${DIRS[@]}"; do
  if [[ "$ALLOW_CREATE_DIRS" == "1" ]]; then
    mkdir -p "$dir"
    echo "created_or_exists: $dir"
  else
    echo "would_create: $dir"
  fi
done

cat <<'YAML'

Matching configs/paths.local.yaml template:
assets:
  openvla_oft_ckpt: "C:/assets/checkpoints/openvla-oft"
  smolvla_ckpt: "C:/assets/checkpoints/smolvla"
  libero_root: "C:/assets/repos/LIBERO"
  libero_data_root: "C:/assets/data/libero"
  robosuite_root: "C:/assets/repos/robosuite"
  data_root: "C:/assets/data"
  checkpoint_root: "C:/assets/checkpoints"
  hf_home: "C:/assets/hf_home"
  wandb_api_key: null
YAML

echo ""
echo "To actually create directories, rerun with:"
echo "ALLOW_CREATE_DIRS=1 bash scripts/21_make_asset_dirs.sh"
