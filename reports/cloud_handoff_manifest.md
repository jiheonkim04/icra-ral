# Cloud Handoff Manifest

This starter manifest is tracked for review. Run `scripts/23_cloud_handoff_manifest.ps1` or `scripts/23_cloud_handoff_manifest.sh` locally to regenerate it with the current checkout commit, branch, and environment summary.

## Git

- Repository: https://github.com/jiheonkim04/icra-ral.git
- Branch: codex/local-papergrade-runner
- Commit hash: regenerate locally with `scripts/23_cloud_handoff_manifest.*`

## Required Assets

- `OPENVLA_OFT_CKPT`
- `SMOLVLA_CKPT`
- `LIBERO_ROOT`
- `LIBERO_DATA_ROOT`
- `ROBOSUITE_ROOT`
- `DATA_ROOT`
- `CHECKPOINT_ROOT`
- `HF_HOME`

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
git checkout codex/local-papergrade-runner
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

Download or cache models only after explicit approval and `ALLOW_DOWNLOADS=1`. Do not include provider-specific secrets or tokens in tracked files.
