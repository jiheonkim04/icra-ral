# Cloud Handoff Manifest

This manifest is generated for remote Linux GPU preparation only. It does not launch cloud jobs, download assets, run training, or run rollouts.

## Git

- Repository: https://github.com/jiheonkim04/icra-ral.git
- Branch: codex/replace-hard-stops-with-risk-assessed-autonomy
- Commit hash: c78ef1a029b09064bcf4e453e3391f7d0042dfc8

## Python / Conda Summary

- Conda env: unavailable
- Python: unavailable

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

- configs/paths.local.yaml after replacing local paths with remote paths and removing secrets.
- Experiment configs once created.
- reports/real_asset_setup_plan.md.
- reports/local_papergrade_plan.md.

## Remote Linux Commands

git clone https://github.com/jiheonkim04/icra-ral.git tca_map
cd tca_map
git checkout codex/replace-hard-stops-with-risk-assessed-autonomy
conda activate tca_map
bash scripts/00_preflight.sh
bash scripts/11_check_real_assets.sh
bash scripts/20_system_readiness.sh
bash scripts/22_plan_local_experiment_matrix.sh

## Transfer Examples

rsync -av --exclude configs/paths.local.yaml --exclude runs/ --exclude reports/system_readiness.json ./ user@remote:/path/to/tca_map/
rsync -av user@remote:/path/to/tca_map/reports/ ./reports/

## Download Policy

Download or cache models only after a green remote/cloud risk assessment and task-local ALLOW_DOWNLOADS=1. Do not include provider-specific secrets or tokens in tracked files.
