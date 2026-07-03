#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUDGET_FILE="${BUDGET_FILE:-configs/compute_budget.yaml}"
shift_args=("$@")

scalar_value() {
  local file="$1"
  local key="$2"
  local default_value="$3"
  if [[ ! -f "$file" ]]; then
    printf '%s' "$default_value"
    return
  fi
  local value
  value="$(awk -v wanted="$key" '
    $1 == wanted":" {
      $1=""; sub(/^[[:space:]]+/, "", $0); gsub(/^"|"$/, "", $0); print $0; exit
    }
  ' "$file" | tr -d '\r')"
  if [[ -n "$value" ]]; then printf '%s' "$value"; else printf '%s' "$default_value"; fi
}

truthy_key() {
  local file="$1"
  local key="$2"
  grep -Eq "^[[:space:]]*${key}[[:space:]]*:[[:space:]]*true[[:space:]]*$" "$file"
}

numeric_key() {
  local file="$1"
  local key="$2"
  awk -v wanted="$key" '
    $1 == wanted":" {
      print $2; exit
    }
  ' "$file" | tr -d '\r'
}

MAX_GRID="$(scalar_value "$BUDGET_FILE" max_heatmap_grid_initial 8)"
MAX_PARAMS="$(scalar_value "$BUDGET_FILE" max_trainable_params_millions_initial 50)"
MAX_STEPS="$(scalar_value "$BUDGET_FILE" max_local_pilot_steps_initial 300)"
ERRORS=()
WARNINGS=()
CHECKED=()

if [[ ! -f "$BUDGET_FILE" ]]; then
  ERRORS+=("Missing compute budget file: $BUDGET_FILE")
fi

CONFIGS=()
if [[ ${#shift_args[@]} -gt 0 ]]; then
  CONFIGS=("${shift_args[@]}")
else
  while IFS= read -r file; do
    case "$file" in
      configs/paths.example.yaml|configs/paths.local.yaml.example|configs/compute_budget.yaml) ;;
      *) CONFIGS+=("$file") ;;
    esac
  done < <(find configs -maxdepth 1 -name '*.yaml' -type f | sort)
fi

for file in "${CONFIGS[@]}"; do
  if [[ ! -f "$file" ]]; then
    ERRORS+=("Missing config: $file")
    continue
  fi
  CHECKED+=("$file")
  for key in openvla_oft_full_finetune openvla_oft_full_rollout openvla_oft_multiseed_sweep high_resolution_voxel_heatmap full_finetune full_rollout multiseed_sweep train_backbone; do
    if truthy_key "$file" "$key"; then
      ERRORS+=("$file enables forbidden key ${key}: true")
    fi
  done
  grid="$(numeric_key "$file" grid_size)"
  if [[ -n "$grid" && "$grid" -gt "$MAX_GRID" ]]; then
    ERRORS+=("$file sets grid_size=$grid above max_heatmap_grid_initial=$MAX_GRID")
  fi
  params="$(numeric_key "$file" trainable_params_millions_estimate)"
  if [[ -n "$params" && "$params" -gt "$MAX_PARAMS" ]]; then
    ERRORS+=("$file estimates trainable_params_millions=$params above limit=$MAX_PARAMS")
  fi
  steps="$(numeric_key "$file" max_steps)"
  if [[ -n "$steps" && "$steps" -gt "$MAX_STEPS" ]]; then
    ERRORS+=("$file sets max_steps=$steps above max_local_pilot_steps_initial=$MAX_STEPS")
  fi
  lower="$(tr -d '\r' < "$file" | tr '[:upper:]' '[:lower:]')"
  openvla_active=false
  if [[ "$lower" == *openvla* && "$lower" != *"openvla_oft_enabled: false"* ]]; then
    if [[ "$lower" == *"openvla_oft:"* && "$lower" == *"enabled: false"* ]]; then
      openvla_active=false
    else
      openvla_active=true
    fi
  fi
  if [[ "$openvla_active" == true && "$lower" != *frozen* && "$lower" != *load* && "$lower" != *smoke* ]]; then
    WARNINGS+=("$file mentions active OpenVLA without an obvious frozen/load/smoke context")
  fi
  if [[ "$openvla_active" == true && ( "$lower" == *"train: true"* || "$lower" == *"train_heads: true"* ) ]]; then
    WARNINGS+=("$file mentions active OpenVLA and training. Verify this is not OpenVLA training")
  fi
done

printf 'compute_budget_report:\n'
printf '  budget_file: %s\n' "$BUDGET_FILE"
printf '  max_heatmap_grid_initial: %s\n' "$MAX_GRID"
printf '  max_trainable_params_millions_initial: %s\n' "$MAX_PARAMS"
printf '  max_local_pilot_steps_initial: %s\n' "$MAX_STEPS"
printf '  downloads_performed: false\n'
printf '  gpu_jobs_performed: false\n'
printf '  rollouts_performed: false\n'
printf '  checked_configs:\n'
for file in "${CHECKED[@]}"; do printf '    - %s\n' "$file"; done
printf '  warnings:\n'
if [[ ${#WARNINGS[@]} -eq 0 ]]; then printf '    - none\n'; else for warning in "${WARNINGS[@]}"; do printf '    - %s\n' "$warning"; done; fi
printf '  errors:\n'
if [[ ${#ERRORS[@]} -eq 0 ]]; then printf '    - none\n'; else for error in "${ERRORS[@]}"; do printf '    - %s\n' "$error"; done; fi

if [[ ${#ERRORS[@]} -gt 0 ]]; then
  exit 1
fi
exit 0
