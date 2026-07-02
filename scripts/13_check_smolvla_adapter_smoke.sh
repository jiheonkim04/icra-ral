#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PATHS_FILE="${PATHS_FILE:-configs/paths.local.yaml}"
PYTHON_BIN="${PYTHON:-python}"

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
  local env_value="${!env_name:-}"
  local config
  config="$(config_value "$key")"
  if [[ -n "$env_value" ]]; then
    printf '%s' "$env_value"
  else
    printf '%s' "$config"
  fi
}

first_existing() {
  local root="$1"
  shift
  local found=()
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  for name in "$@"; do
    if [[ -f "$root/$name" ]]; then
      found+=("$name")
    fi
  done
  printf '%s\n' "${found[@]}"
}

glob_existing() {
  local root="$1"
  shift
  local found=()
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  for pattern in "$@"; do
    while IFS= read -r file; do
      [[ -n "$file" ]] && found+=("$(basename "$file")")
    done < <(find "$root" -maxdepth 1 -type f -name "$pattern" 2>/dev/null)
  done
  printf '%s\n' "${found[@]}" | sort -u
}

smolvla_ckpt="$(resolve_value smolvla_ckpt SMOLVLA_CKPT)"
checkpoint_root="$(resolve_value checkpoint_root CHECKPOINT_ROOT)"
hf_home="$(resolve_value hf_home HF_HOME)"

ckpt_exists=false
if [[ -n "$smolvla_ckpt" && -d "$smolvla_ckpt" ]]; then
  ckpt_exists=true
fi

mapfile -t config_files < <(first_existing "$smolvla_ckpt" config.json)
mapfile -t tokenizer_files < <(first_existing "$smolvla_ckpt" tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt tokenizer.model sentencepiece.bpe.model)
mapfile -t weight_files < <({ first_existing "$smolvla_ckpt" model.safetensors pytorch_model.bin model-00001-of-00001.safetensors pytorch_model-00001-of-00001.bin; glob_existing "$smolvla_ckpt" "*.safetensors" "*.bin"; } | sort -u)

gpu_name=""
gpu_memory_mb=""
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_line="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
  if [[ -n "$gpu_line" ]]; then
    gpu_name="${gpu_line%%,*}"
    gpu_memory_mb="${gpu_line##*, }"
  fi
fi

estimated_load_mb=12000
required_headroom_mb=2048
memory_fits=false
if [[ "$gpu_memory_mb" =~ ^[0-9]+$ ]] && (( gpu_memory_mb >= estimated_load_mb + required_headroom_mb )); then
  memory_fits=true
fi

adapter_import_ok=false
adapter_import_error=""
if "$PYTHON_BIN" -c "from tca_map.adapters.lora_policy import validate_lora_policy_config; print('lightweight_adapter_guard_import_ok')" >/tmp/tca_map_smolvla_adapter_import.txt 2>&1; then
  adapter_import_ok=true
else
  adapter_import_error="$(cat /tmp/tca_map_smolvla_adapter_import.txt 2>/dev/null || true)"
fi

ready=false
if [[ "$ckpt_exists" == true && ${#config_files[@]} -gt 0 && ${#tokenizer_files[@]} -gt 0 && ${#weight_files[@]} -gt 0 && "$memory_fits" == true && "$adapter_import_ok" == true ]]; then
  ready=true
fi

echo "SmolVLA adapter smoke readiness check"
echo "repo_root: $REPO_ROOT"
echo "policy.downloads_performed: false"
echo "policy.gpu_training_performed: false"
echo "policy.training_performed: false"
echo "policy.heavy_model_imports_performed: false"
echo "policy.openvla_oft_executed: false"
echo "policy.real_rollouts_performed: false"
echo "policy.libero_required: false"
echo "policy.heavy_import_gate_set: $([[ "${ALLOW_HEAVY_IMPORT:-}" == "1" ]] && echo true || echo false)"
echo "ready_for_smolvla_adapter_smoke: $ready"
echo "smolvla_ckpt.configured: $([[ -n "$smolvla_ckpt" ]] && echo true || echo false)"
echo "smolvla_ckpt.exists: $ckpt_exists"
echo "cache.checkpoint_root.configured: $([[ -n "$checkpoint_root" ]] && echo true || echo false)"
echo "cache.hf_home.configured: $([[ -n "$hf_home" ]] && echo true || echo false)"
echo "expected.config_found: ${config_files[*]:-none}"
echo "expected.tokenizer_found: ${tokenizer_files[*]:-none}"
echo "expected.weights_found: ${weight_files[*]:-none}"
echo "adapter.lightweight_adapter_guard_import_ok: $adapter_import_ok"
if [[ -n "$adapter_import_error" ]]; then
  echo "adapter.lightweight_adapter_guard_import_error: $adapter_import_error"
fi
echo "adapter.actual_smolvla_heavy_import_attempted: false"
echo "memory.gpu_name: ${gpu_name:-unknown}"
echo "memory.gpu_memory_total_mb: ${gpu_memory_mb:-unknown}"
echo "memory.estimated_load_mb: $estimated_load_mb"
echo "memory.required_headroom_mb: $required_headroom_mb"
echo "memory.fits_rtx_5080_16gb_budget: $memory_fits"

if [[ "$ready" == true ]]; then
  echo "recommended_next_step: Ready for a separately approved SmolVLA load-only adapter smoke. Do not train."
elif [[ "$ckpt_exists" != true ]]; then
  echo "recommended_next_step: Configure SMOLVLA_CKPT to a local checkpoint directory first."
else
  echo "recommended_next_step: Check config/tokenizer/weights files and memory estimate before real adapter smoke."
fi

exit 0
