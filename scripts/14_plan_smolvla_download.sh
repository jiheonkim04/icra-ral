#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PATHS_FILE="${PATHS_FILE:-configs/paths.local.yaml}"
ASSET_ROOT="${ASSET_ROOT:-C:/assets}"

read_asset_key() {
  local key="$1"
  local file="$2"
  if [[ ! -f "${file}" ]]; then
    return 0
  fi
  awk -v key="${key}" '
    { sub(/\r$/, "", $0) }
    /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
    /^assets[[:space:]]*:/ { in_assets=1; next }
    in_assets && /^[^[:space:]]/ { exit }
    in_assets {
      pattern="^[[:space:]]+" key "[[:space:]]*:"
      if ($0 ~ pattern) {
        value=$0
        sub(/^[^:]*:[[:space:]]*/, "", value)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        gsub(/^["\047]|["\047]$/, "", value)
        if (value != "" && tolower(value) != "null") {
          print value
          exit
        }
      }
    }
  ' "${file}"
}

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

configured_or_default() {
  local env_name="$1"
  local config_key="$2"
  local default_value="$3"
  local env_value="${!env_name:-}"
  if [[ -n "${env_value}" ]]; then
    printf '%s|env:%s\n' "${env_value}" "${env_name}"
    return 0
  fi
  local config_value
  config_value="$(read_asset_key "${config_key}" "${PATHS_FILE}" || true)"
  if [[ -n "${config_value}" && "${config_value}" != "null" ]]; then
    printf '%s|%s\n' "${config_value}" "${PATHS_FILE}"
    return 0
  fi
  printf '%s|default\n' "${default_value}"
}

checkpoint_pair="$(configured_or_default CHECKPOINT_ROOT checkpoint_root "${ASSET_ROOT}/checkpoints")"
CHECKPOINT_ROOT_VALUE="${checkpoint_pair%%|*}"
CHECKPOINT_ROOT_SOURCE="${checkpoint_pair#*|}"

hf_pair="$(configured_or_default HF_HOME hf_home "${ASSET_ROOT}/hf_home")"
HF_HOME_VALUE="${hf_pair%%|*}"
HF_HOME_SOURCE="${hf_pair#*|}"

smolvla_pair="$(configured_or_default SMOLVLA_CKPT smolvla_ckpt "${CHECKPOINT_ROOT_VALUE}/smolvla")"
SMOLVLA_CKPT_VALUE="${smolvla_pair%%|*}"
SMOLVLA_CKPT_SOURCE="${smolvla_pair#*|}"

ALLOW_DOWNLOADS_SET=false
if [[ "${ALLOW_DOWNLOADS:-}" == "1" ]]; then
  ALLOW_DOWNLOADS_SET=true
fi

echo "SmolVLA checkpoint acquisition plan"
echo "Repo root: ${REPO_ROOT}"
echo "Dry run only: true"
echo "ALLOW_DOWNLOADS set: ${ALLOW_DOWNLOADS_SET}"
echo "Downloads performed: false"
echo "Heavy VLA imports performed: false"
echo "GPU jobs performed: false"
echo "Training performed: false"
echo "Rollouts performed: false"
echo "OpenVLA-OFT executed: false"
echo
echo "Intended paths:"
echo "SMOLVLA_CKPT=${SMOLVLA_CKPT_VALUE} [${SMOLVLA_CKPT_SOURCE}]"
echo "CHECKPOINT_ROOT=${CHECKPOINT_ROOT_VALUE} [${CHECKPOINT_ROOT_SOURCE}]"
echo "HF_HOME=${HF_HOME_VALUE} [${HF_HOME_SOURCE}]"
echo
echo "Required files:"
echo "- config: config.json"
echo "- tokenizer_any: tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, tokenizer.model, sentencepiece.bpe.model"
echo "- weights_any: model.safetensors, pytorch_model.bin, *.safetensors, *.bin"
echo
if [[ "${ALLOW_DOWNLOADS_SET}" == "true" ]]; then
  echo "ALLOW_DOWNLOADS=1 is set, but this planner still performs no downloads."
else
  echo "ALLOW_DOWNLOADS is not set. This is the expected planning-only state."
fi
echo
SMOLVLA_CKPT_JSON="$(json_escape "${SMOLVLA_CKPT_VALUE}")"
SMOLVLA_CKPT_SOURCE_JSON="$(json_escape "${SMOLVLA_CKPT_SOURCE}")"
CHECKPOINT_ROOT_JSON="$(json_escape "${CHECKPOINT_ROOT_VALUE}")"
CHECKPOINT_ROOT_SOURCE_JSON="$(json_escape "${CHECKPOINT_ROOT_SOURCE}")"
HF_HOME_JSON="$(json_escape "${HF_HOME_VALUE}")"
HF_HOME_SOURCE_JSON="$(json_escape "${HF_HOME_SOURCE}")"
cat <<JSON
{
  "policy": {
    "dry_run_only": true,
    "allow_downloads_gate_set": ${ALLOW_DOWNLOADS_SET},
    "downloads_performed": false,
    "directories_created": false,
    "gpu_jobs_performed": false,
    "training_performed": false,
    "real_rollouts_performed": false,
    "heavy_model_imports_performed": false,
    "openvla_oft_executed": false,
    "tokens_read_or_written": false
  },
  "intended_paths": {
    "smolvla_ckpt": {
      "env": "SMOLVLA_CKPT",
      "value": "${SMOLVLA_CKPT_JSON}",
      "source": "${SMOLVLA_CKPT_SOURCE_JSON}"
    },
    "checkpoint_root": {
      "env": "CHECKPOINT_ROOT",
      "value": "${CHECKPOINT_ROOT_JSON}",
      "source": "${CHECKPOINT_ROOT_SOURCE_JSON}"
    },
    "hf_home": {
      "env": "HF_HOME",
      "value": "${HF_HOME_JSON}",
      "source": "${HF_HOME_SOURCE_JSON}"
    }
  },
  "required_files": {
    "config": ["config.json"],
    "tokenizer_any": ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "tokenizer.model", "sentencepiece.bpe.model"],
    "tokenizer_also_accepted_by_readiness_checker": ["special_tokens_map.json"],
    "weights_any": ["model.safetensors", "pytorch_model.bin", "*.safetensors", "*.bin"]
  },
  "readiness_semantics": {
    "path_ready_is_not_adapter_smoke_ready": true,
    "adapter_smoke_requires_config_tokenizer_weights": true,
    "adapter_smoke_requires_hf_home_or_checkpoint_root": true,
    "smolvla_smoke_is_interface_validation_only": true,
    "paper_grade_requires_real_benchmark_data_and_rollouts_later": true
  },
  "recommended_next_step": "Manually place a SmolVLA-compatible checkpoint under SMOLVLA_CKPT, then run scripts/11_check_real_assets.ps1 and scripts/13_check_smolvla_adapter_smoke.ps1."
}
JSON
