# TCA-Map

Target-Conditioned ActionMap for counterfactual grounding in vision-language-action policies.

This repository starts with a conservative scaffold for a two-week kill-or-continue pilot. The immediate milestone is not a full paper run. It is:

1. scaffold,
2. preflight,
3. dummy smoke test,
4. one real adapter smoke test only when local paths exist,
5. one tiny offline pilot later,
6. go/no-go report.

## Safety policy

- No automatic downloads of OpenVLA-OFT, SmolVLA, LIBERO, RoboCasa, checkpoints, or datasets.
- No GPU training in the scaffold step.
- No real rollouts until simulator paths pass preflight.
- Missing assets should not block dummy smoke or interface validation.
- Offline proxy metrics are engineering validation only and must not be described as final standard success.
- OpenVLA-OFT large experiments are forbidden on local hardware; OpenVLA-OFT may only be used for frozen/load smoke unless a separate explicit approval branch changes this policy.
- Any SOTA claim must be restricted to low-compute target-conditioned action decoding or counterfactual robustness unless full standard baselines are directly reproduced.

Heavy actions require explicit environment gates:

- `ALLOW_DOWNLOADS=1`
- `ALLOW_HEAVY_IMPORT=1`
- `ALLOW_GPU_TRAINING=1`
- `ALLOW_ROLLOUTS=1`
- `ALLOW_CLOUD_HANDOFF=1`

## Local-first execution

On Windows PowerShell from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/99_tree_check.ps1
powershell -ExecutionPolicy Bypass -File scripts/00_preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts/04_train_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts/05_eval_smoke.ps1
```

The shell script `scripts/00_preflight.sh` is included for Linux/WSL users, but PowerShell preflight is the supported first path on Windows.

## Real asset readiness

The current recommendation is **SmolVLA-first** for the first real-adapter smoke on an RTX 5080 16GB local machine. OpenVLA-OFT remains the primary paper-grade baseline target, but full OpenVLA-OFT fine-tuning should not be attempted locally.

Read the plan:

```powershell
Get-Content reports/real_asset_setup_plan.md
```

Check local paths without downloads, heavy model imports, GPU jobs, or rollouts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/11_check_real_assets.ps1
```

Plan the later SmolVLA load-only smoke without importing SmolVLA or loading a model:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/15_plan_smolvla_load_only_smoke.ps1
```

Check the bounded load-only execution scaffold. It will stop before heavy import/model load unless the appropriate gate and runtime dependencies are present:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/16_smolvla_load_only_smoke.ps1
```

Check SmolVLA runtime dependencies without installing or importing heavy models:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/17_check_smolvla_runtime_deps.ps1
```

Linux/WSL equivalent:

```bash
bash scripts/11_check_real_assets.sh
```

## Low-compute path without OpenVLA-OFT large runs

We avoid large OpenVLA-OFT work locally because a 16GB GPU and 24GB system RAM are better suited for adapter smoke, cached-feature pilots, and head-only experiments than full large-backbone fine-tuning or multi-seed rollouts.

SmolVLA-first is the local real-adapter path because it gives a cheaper way to validate interfaces, cached feature extraction, ActionMap, and TCA-Map before moving larger baselines to WSL2/Linux or remote GPU.

Frozen OpenVLA-OFT smoke is not a result. It is only a load/interface/VRAM feasibility check. It must not be used as a training result, rollout result, or paper-grade performance claim.

Enforce the local compute policy before running any new config:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/30_enforce_compute_budget.ps1
powershell -ExecutionPolicy Bypass -File scripts/30_enforce_compute_budget.ps1 -Config configs\tca_map_head_only_lowcompute.yaml
```

Linux/WSL equivalent:

```bash
bash scripts/30_enforce_compute_budget.sh
bash scripts/30_enforce_compute_budget.sh configs/tca_map_head_only_lowcompute.yaml
```

## Distributional TCA-Select + optional LoRA low-compute method

The publishable low-compute method is **Distributional TCA-Map**: TCA-Map plus Distributional TCA-Select, with optional LoRA/QLoRA only if head-only training underfits.

TCA-Select must be distributional for the final method, not only heuristic geometry. The final selector samples `K=4` candidate actions from the target-conditioned action heatmap and selects among them using internal heatmap-distribution signals:

- log probability under the full action heatmap,
- condition-masked action heatmap KL,
- counterfactual negative action heatmap KL or JS,
- target heatmap consistency or margin,
- entropy penalty.

It must not use external verifiers or privileged simulator state at inference. Heuristic target/action consistency remains useful as an ablation, not the final method.

LoRA/QLoRA are support tools. They can reduce training cost or memory pressure, but they are not the main novelty. The default is frozen SmolVLA backbone, cached features, and head-only ActionMap/TCA-Map training.

Read the method notes:

```powershell
Get-Content reports/tca_select_method.md
Get-Content reports/final_method_spec_distributional_tca_map.md
Get-Content reports/mg_select_vs_distributional_tca_select.md
Get-Content reports/lora_inference_ablation_plan.md
Get-Content reports/lora_vs_inference_trick_strategy.md
Get-Content reports/publishability_criteria.md
```

## Path to paper-grade experiments without leaving home

1. Local Windows scaffold validation: run tree check, preflight, dummy train/eval smoke, pytest, asset checks, compute-budget enforcement, and system readiness checks.
2. WSL2/Linux rollout setup: use `scripts/24_wsl2_setup_check.ps1`, then install Ubuntu manually if needed and validate GPU visibility from WSL2.
3. SmolVLA-first local smoke: configure `SMOLVLA_CKPT`, `HF_HOME`, and `CHECKPOINT_ROOT`, then run readiness checks. Model execution remains a later approved task.
4. Small local rollout: after WSL2/Linux, LIBERO, RoboSuite, and data paths pass checks, run only a separately approved tiny rollout task.
5. OpenVLA-OFT frozen smoke: keep OpenVLA-OFT as the paper-grade baseline target, but attempt only frozen/load smoke locally and only after memory checks pass.
6. Cloud/remote GPU handoff: use `scripts/23_cloud_handoff_manifest.*` to prepare a manifest for 24GB, 48GB, or 80GB GPU classes depending on baseline scale.

Planner scripts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/20_system_readiness.ps1
powershell -ExecutionPolicy Bypass -File scripts/21_make_asset_dirs.ps1
powershell -ExecutionPolicy Bypass -File scripts/22_plan_local_experiment_matrix.ps1
powershell -ExecutionPolicy Bypass -File scripts/23_cloud_handoff_manifest.ps1
powershell -ExecutionPolicy Bypass -File scripts/24_wsl2_setup_check.ps1
```

Publication-oriented tables should separate:

- native SmolVLA head,
- ActionMap,
- ActionMap + counterfactual augmentation,
- TCA-Map,
- TCA-Map + heuristic TCA-Select,
- TCA-Map + Distributional TCA-Select,
- optional LoRA/QLoRA variants if used,
- latency, VRAM, and trainable parameters.

## Asset configuration

Copy `configs/paths.local.yaml.example` to `configs/paths.local.yaml` and fill in local paths if available. The local file is ignored by git.

Environment variables with equivalent meaning are also supported:

- `OPENVLA_OFT_CKPT`
- `SMOLVLA_CKPT`
- `LIBERO_ROOT`
- `LIBERO_DATA_ROOT`
- `ROBOSUITE_ROOT`
- `DATA_ROOT`
- `CHECKPOINT_ROOT`
- `HF_HOME`
- `WANDB_API_KEY`

## Current scaffold contents

The Python package contains lightweight, dependency-minimal dummy components for interface validation:

- dummy LIBERO-style dataset samples,
- counterfactual target-swap generation,
- dummy VLA adapter,
- ActionMap-style heatmap head,
- target-conditioned TCA-Map head,
- heuristic and distributional TCA-Select inference-time candidate selection,
- optional LoRA/QLoRA policy guards,
- compute-budget guards,
- local paper-grade planning scripts,
- cloud handoff planning artifacts,
- offline proxy metrics,
- preflight and smoke entrypoints.
