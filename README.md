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
- OpenVLA-OFT large experiments are forbidden on local hardware. OpenVLA-OFT may only be used for frozen/load smoke unless a separate explicit approval branch changes this policy.

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

Linux/WSL equivalent:

```bash
bash scripts/11_check_real_assets.sh
```

## Low-compute path without OpenVLA-OFT large runs

We avoid large OpenVLA-OFT work locally because a 16GB GPU and 24GB system RAM are better suited for adapter smoke, cached-feature pilots, and head-only experiments than full large-backbone fine-tuning or multi-seed rollouts.

SmolVLA-first is the local real-adapter path because it gives a cheaper way to validate interfaces, cached feature extraction, ActionMap, and TCA-Map before moving larger baselines to WSL2/Linux or remote GPU.

Frozen OpenVLA-OFT smoke is not a result. It is only a load/interface/VRAM feasibility check. It must not be used as a training result, rollout result, or paper-grade performance claim.

The publication-oriented table stack should include:

- SmolVLA native head.
- ActionMap.
- ActionMap + counterfactual augmentation.
- TCA-Map.
- Offline proxy metrics clearly labeled as proxy metrics.
- Small LIBERO rollout metrics once WSL2/LIBERO passes.
- Ablations for target head only, counterfactual alignment, and diagnostic/loss weighting when implemented.
- Latency, VRAM, and trainable parameter counts.

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
- offline proxy metrics,
- preflight and smoke entrypoints.
