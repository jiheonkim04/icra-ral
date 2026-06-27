#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p reports
cat > reports/local_experiment_matrix.md <<'MD'
# Local Experiment Matrix

This matrix is a plan only. It does not authorize downloads, GPU training, heavy model imports, or rollouts.

## Safety Gates

- Downloads require `ALLOW_DOWNLOADS=1`.
- Heavy VLA imports require `ALLOW_HEAVY_IMPORT=1`.
- GPU training requires `ALLOW_GPU_TRAINING=1`.
- Rollouts require `ALLOW_ROLLOUTS=1`.
- Cloud handoff requires `ALLOW_CLOUD_HANDOFF=1`.

## Experiments

| Stage | Experiment | Assets | Execution target | Cloud required | Notes |
| --- | --- | --- | --- | --- | --- |
| Local smoke | dummy | none | Windows or Linux CPU | no | Already used for scaffold validation. |
| Local smoke | SmolVLA adapter smoke | `SMOLVLA_CKPT`, `HF_HOME` or `CHECKPOINT_ROOT` | Local RTX 5080, load-only later | no | Recommended first real adapter smoke. |
| Local smoke | OpenVLA-OFT frozen smoke | `OPENVLA_OFT_CKPT`, `HF_HOME`, `CHECKPOINT_ROOT` | Local only if memory check passes | maybe | Paper-grade target, no full fine-tuning locally. |
| Offline proxy | native head | tiny local subset | Windows or Linux | no | Report as `offline_standard_proxy`, not standard success. |
| Offline proxy | ActionMap | tiny local subset | Windows or Linux | no | Low-resolution heatmap. |
| Offline proxy | ActionMap + counterfactual augmentation | tiny target-swap split | Windows or Linux | no | Tests augmentation-only baseline. |
| Offline proxy | TCA-Map | tiny target-swap split | Windows or Linux | no | Target-conditioned head. |
| Small rollout | SmolVLA baseline | LIBERO assets and simulator | WSL2/Linux recommended | no | Requires explicit rollout gate. |
| Small rollout | ActionMap | LIBERO tiny subset | WSL2/Linux recommended | no | Low-resolution heatmap only. |
| Small rollout | TCA-Map | LIBERO tiny target-swap/counterfactual split | WSL2/Linux recommended | no | No privileged inference. |
| Small rollout | tiny LIBERO subset | `LIBERO_ROOT`, `LIBERO_DATA_ROOT`, `ROBOSUITE_ROOT` | WSL2/Linux | no | Paper-grade standard success begins here, not offline proxy. |
| Small rollout | tiny target-swap/counterfactual split | LIBERO assets plus generated JSONL | WSL2/Linux | no | Robustness evidence. |
| Large baseline | OpenVLA-OFT native | full local/remote checkpoint and data | remote Linux GPU | yes | 24GB minimum frozen/head-only, 48GB+ preferred. |
| Large baseline | ActionMap | full benchmark subset | remote Linux GPU | yes | Strong baseline. |
| Large baseline | TCA-Map | full benchmark subset | remote Linux GPU | yes | Full method. |
| Large baseline | full LIBERO/RoboCasa subset | full simulator and data assets | remote Linux GPU | yes | Multi-task paper-grade evidence. |

## Go / No-Go Thresholds

Continue only if:

- Standard metric degradation is no more than 1-2 percentage points versus the strongest implemented baseline.
- Robust/counterfactual gain is at least +10 percentage points.
- Wrong-target rate reduction is at least 20 percent relative.
- VRAM headroom is at least 2 GB for real adapter smoke or rollout.
- Default inference uses no privileged simulator state.

Stop or pivot if:

- Offline proxy is the only improvement and rollout evidence fails.
- TCA-Map does not beat ActionMap + counterfactual augmentation.
- OpenVLA-OFT or SmolVLA cannot be run without downloads or unsupported local dependencies.
- Simulator setup is not reproducible under WSL2/Linux.
MD

echo "Wrote reports/local_experiment_matrix.md"
