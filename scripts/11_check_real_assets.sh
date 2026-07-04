#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PATHS_FILE="${PATHS_FILE:-configs/paths.local.yaml}"

config_value() {
  local key="$1"
  if [[ ! -f "$PATHS_FILE" ]]; then
    return 0
  fi
  awk -v wanted="$key" '
    /^assets[[:space:]]*:/ { in_assets=1; next }
    in_assets && /^[^[:space:]]/ { exit }
    in_assets {
      line=$0
      sub(/^[[:space:]]+/, "", line)
      split(line, parts, ":")
      if (parts[1] == wanted) {
        sub(/^[^:]+:[[:space:]]*/, "", line)
        gsub(/^\"|\"$/, "", line)
        gsub(/^'\''|'\''$/, "", line)
        if (line != "null") print line
        exit
      }
    }
  ' "$PATHS_FILE"
}

resolve_value() {
  local key="$1"
  local env_name="$2"
  local env_value="${!env_name:-}"
  if [[ -n "$env_value" ]]; then
    printf '%s' "$env_value"
  else
    config_value "$key"
  fi
}

check_asset() {
  local key="$1"
  local env_name="$2"
  local label="$3"
  local value
  value="$(resolve_value "$key" "$env_name")"
  local configured=false
  local exists=false
  if [[ -n "$value" ]]; then
    configured=true
    if [[ -e "$value" ]]; then
      exists=true
    fi
  fi
  ASSET_EXISTS["$key"]="$exists"
  if [[ "$exists" != true ]]; then
    MISSING_ASSETS+=("$env_name")
  fi
  printf 'asset.%s.env: %s\n' "$key" "$env_name"
  printf 'asset.%s.label: %s\n' "$key" "$label"
  printf 'asset.%s.configured: %s\n' "$key" "$configured"
  printf 'asset.%s.exists: %s\n' "$key" "$exists"
}

declare -A ASSET_EXISTS
MISSING_ASSETS=()

echo "TCA-Map real asset readiness check"
echo "repo_root: $REPO_ROOT"
echo "policy.local_paths_only: true"
echo "policy.downloads_performed: false"
echo "policy.gpu_jobs_performed: false"
echo "policy.heavy_model_imports_performed: false"
echo "policy.real_rollouts_performed: false"

check_asset "openvla_oft_ckpt" "OPENVLA_OFT_CKPT" "OpenVLA-OFT checkpoint or local model directory"
check_asset "smolvla_ckpt" "SMOLVLA_CKPT" "SmolVLA checkpoint or local model directory"
check_asset "libero_root" "LIBERO_ROOT" "LIBERO source checkout"
check_asset "libero_data_root" "LIBERO_DATA_ROOT" "LIBERO data/demos root"
check_asset "robosuite_root" "ROBOSUITE_ROOT" "RoboSuite checkout/install root"
check_asset "data_root" "DATA_ROOT" "General data root"
check_asset "checkpoint_root" "CHECKPOINT_ROOT" "Checkpoint root"
check_asset "hf_home" "HF_HOME" "Hugging Face cache root"

ready_for_smolvla_smoke=false
if [[ "${ASSET_EXISTS[smolvla_ckpt]}" == true && ( "${ASSET_EXISTS[hf_home]}" == true || "${ASSET_EXISTS[checkpoint_root]}" == true ) ]]; then
  ready_for_smolvla_smoke=true
fi

ready_for_openvla_oft_smoke=false
if [[ "${ASSET_EXISTS[openvla_oft_ckpt]}" == true && "${ASSET_EXISTS[hf_home]}" == true && "${ASSET_EXISTS[checkpoint_root]}" == true ]]; then
  ready_for_openvla_oft_smoke=true
fi

ready_for_libero_rollout_path_check=false
if [[ "${ASSET_EXISTS[libero_root]}" == true && "${ASSET_EXISTS[libero_data_root]}" == true && "${ASSET_EXISTS[robosuite_root]}" == true ]]; then
  ready_for_libero_rollout_path_check=true
fi
ready_for_libero_rollout=false

echo "ready_for_smolvla_smoke: $ready_for_smolvla_smoke"
echo "ready_for_openvla_oft_smoke: $ready_for_openvla_oft_smoke"
echo "ready_for_libero_rollout_path_check: $ready_for_libero_rollout_path_check"
echo "ready_for_libero_rollout: $ready_for_libero_rollout"
echo "ready_for_libero_rollout_reason: Rollout readiness is never inferred from paths alone; run a separate simulator import/render/rollout risk gate."
echo "missing_assets: ${MISSING_ASSETS[*]:-none}"

if [[ "$ready_for_smolvla_smoke" == true ]]; then
  echo "recommended_next_step: Continue to the risk-assessed bounded SmolVLA load-only adapter smoke. Do not train."
elif [[ "$ready_for_openvla_oft_smoke" == true ]]; then
  echo "recommended_next_step: OpenVLA-OFT assets are present, but SmolVLA-first is still recommended on RTX 5080 16GB."
else
  echo "recommended_next_step: Configure missing local paths, preferably SMOLVLA_CKPT plus HF_HOME or CHECKPOINT_ROOT first."
fi

exit 0
