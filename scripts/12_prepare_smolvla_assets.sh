#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PATHS_FILE="${PATHS_FILE:-configs/paths.local.yaml}"
ASSET_ROOT="${ASSET_ROOT:-C:/assets}"

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
        gsub(/^"|"$/, "", line)
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
  local default_value="$3"
  local env_value="${!env_name:-}"
  local config
  config="$(config_value "$key")"
  if [[ -n "$env_value" ]]; then
    printf '%s' "$env_value"
  elif [[ -n "$config" ]]; then
    printf '%s' "$config"
  else
    printf '%s' "$default_value"
  fi
}

checkpoint_root="$(resolve_value checkpoint_root CHECKPOINT_ROOT "$ASSET_ROOT/checkpoints")"
hf_home="$(resolve_value hf_home HF_HOME "$ASSET_ROOT/hf_home")"
smolvla_ckpt="$(resolve_value smolvla_ckpt SMOLVLA_CKPT "$checkpoint_root/smolvla")"

allow_downloads=false
if [[ "${ALLOW_DOWNLOADS:-}" == "1" ]]; then
  allow_downloads=true
fi

allow_create_dirs=false
if [[ "${ALLOW_CREATE_DIRS:-}" == "1" ]]; then
  allow_create_dirs=true
fi

echo "SmolVLA asset preparation"
echo "repo_root: $REPO_ROOT"
echo "dry_run: $([[ "$allow_create_dirs" == true ]] && echo false || echo true)"
echo "downloads_allowed_gate_set: $allow_downloads"
echo "downloads_performed: false"
echo "tokens_committed: false"
echo

for dir in "$checkpoint_root" "$hf_home" "$smolvla_ckpt"; do
  if [[ "$allow_create_dirs" == true ]]; then
    mkdir -p "$dir"
    echo "created_or_exists: $dir"
  else
    echo "would_create: $dir"
  fi
done

echo
echo "resolved.SMOLVLA_CKPT: $smolvla_ckpt"
echo "resolved.CHECKPOINT_ROOT: $checkpoint_root"
echo "resolved.HF_HOME: $hf_home"

if [[ "$allow_downloads" == true ]]; then
  echo
  echo "ALLOW_DOWNLOADS=1 is set, but this scaffold does not download automatically."
  echo "Use your authenticated Hugging Face workflow outside this script, then rerun scripts/13_check_smolvla_adapter_smoke.sh."
else
  echo
  echo "To allow a future explicit download-capable script, set ALLOW_DOWNLOADS=1. This script still performs no downloads."
fi

echo
echo "next_safe_check: bash scripts/13_check_smolvla_adapter_smoke.sh"
exit 0
