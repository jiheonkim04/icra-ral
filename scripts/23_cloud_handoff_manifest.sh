#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p reports

COMMIT_HASH="$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unavailable)"
PYTHON_BIN="${PYTHON:-python}"
PYTHON_VERSION="$($PYTHON_BIN --version 2>&1 || echo unavailable)"
CONDA_ENV_VALUE="${CONDA_DEFAULT_ENV:-}"

cat > reports/cloud_handoff_manifest.json <<JSON
{
  "policy": {
    "downloads_performed": false,
    "gpu_jobs_performed": false,
    "training_performed": false,
    "real_rollouts_performed": false,
    "provider_secrets_included": false,
    "require_download_gate": "ALLOW_DOWNLOADS=1 after green remote/cloud risk assessment",
    "require_cloud_gate": "ALLOW_CLOUD_HANDOFF=1"
  },
  "git": {
    "commit_hash": "$COMMIT_HASH",
    "branch": "$BRANCH",
    "repository": "https://github.com/jiheonkim04/icra-ral.git"
  },
  "environment": {
    "conda_env": "$CONDA_ENV_VALUE",
    "python": "$PYTHON_VERSION",
    "os_note": "Generate this manifest locally, then re-run on remote Linux after clone."
  },
  "required_assets": [
    "OPENVLA_OFT_CKPT",
    "SMOLVLA_CKPT",
    "LIBERO_ROOT",
    "LIBERO_DATA_ROOT",
    "ROBOSUITE_ROOT",
    "DATA_ROOT",
    "CHECKPOINT_ROOT",
    "HF_HOME"
  ],
  "expected_resources": {
    "disk": "300GB minimum for focused OpenVLA-OFT/LIBERO work; 500GB-1TB recommended for multi-seed full baselines",
    "vram": "24GB minimum OpenVLA-OFT frozen/head-only; 48GB recommended larger baseline; 80GB recommended multi-seed full baseline",
    "ram": "64GB minimum; 128GB recommended"
  },
  "configs_to_upload": [
    "configs/paths.local.yaml after removing secrets or replacing local paths with remote paths",
    "experiment configs once created",
    "reports/real_asset_setup_plan.md",
    "reports/local_papergrade_plan.md"
  ],
  "remote_commands": [
    "git clone https://github.com/jiheonkim04/icra-ral.git tca_map",
    "cd tca_map",
    "git checkout $BRANCH",
    "conda activate tca_map",
    "bash scripts/00_preflight.sh",
    "bash scripts/11_check_real_assets.sh",
    "bash scripts/20_system_readiness.sh",
    "bash scripts/22_plan_local_experiment_matrix.sh"
  ],
  "transfer": {
    "rsync_to_remote": "rsync -av --exclude configs/paths.local.yaml --exclude runs/ --exclude reports/system_readiness.json ./ user@remote:/path/to/tca_map/",
    "collect_results": "rsync -av user@remote:/path/to/tca_map/reports/ ./reports/"
  },
  "download_policy": "Download or cache models on remote only after a green remote/cloud risk assessment and task-local ALLOW_DOWNLOADS=1. Do not store provider tokens in tracked files."
}
JSON

cat > reports/cloud_handoff_manifest.md <<MD
# Cloud Handoff Manifest

This manifest is generated for remote Linux GPU preparation only. It does not launch cloud jobs, download assets, run training, or run rollouts.

## Git

- Repository: https://github.com/jiheonkim04/icra-ral.git
- Branch: $BRANCH
- Commit hash: $COMMIT_HASH

## Python / Conda Summary

- Conda env: $CONDA_ENV_VALUE
- Python: $PYTHON_VERSION

## Required Assets

- OPENVLA_OFT_CKPT
- SMOLVLA_CKPT
- LIBERO_ROOT
- LIBERO_DATA_ROOT
- ROBOSUITE_ROOT
- DATA_ROOT
- CHECKPOINT_ROOT
- HF_HOME

## Expected Resources

- Disk: 300GB minimum for focused OpenVLA-OFT/LIBERO work; 500GB-1TB recommended for multi-seed full baselines.
- VRAM: 24GB minimum for OpenVLA-OFT frozen/head-only; 48GB recommended for larger baseline; 80GB recommended for multi-seed full baseline.
- RAM: 64GB minimum; 128GB recommended.

## Configs To Upload

- `configs/paths.local.yaml` after replacing local paths with remote paths and removing secrets.
- Experiment configs once created.
- `reports/real_asset_setup_plan.md`.
- `reports/local_papergrade_plan.md`.

## Remote Linux Commands

```bash
git clone https://github.com/jiheonkim04/icra-ral.git tca_map
cd tca_map
git checkout $BRANCH
conda activate tca_map
bash scripts/00_preflight.sh
bash scripts/11_check_real_assets.sh
bash scripts/20_system_readiness.sh
bash scripts/22_plan_local_experiment_matrix.sh
```

## Transfer Examples

```bash
rsync -av --exclude configs/paths.local.yaml --exclude runs/ --exclude reports/system_readiness.json ./ user@remote:/path/to/tca_map/
rsync -av user@remote:/path/to/tca_map/reports/ ./reports/
```

## Download Policy

Download or cache models only after a green remote/cloud risk assessment and task-local `ALLOW_DOWNLOADS=1`. Do not include provider-specific secrets or tokens in tracked files.
MD

echo "Wrote reports/cloud_handoff_manifest.json"
echo "Wrote reports/cloud_handoff_manifest.md"
