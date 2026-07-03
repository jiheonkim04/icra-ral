# No-Large-OpenVLA Low-Compute Strategy

## Strategic Decision

Do not run OpenVLA-OFT large experiments on local hardware. Do not plan local OpenVLA-OFT full fine-tuning, multi-seed full rollout, or large ActionMap/TCA-Map training.

The local path is a low-compute publishable pilot centered on SmolVLA-first real adapter smoke, frozen-backbone or head-only training, cached features, and low-resolution or coarse-to-fine heatmap heads.

## Frozen Smoke Is Not A Result

OpenVLA-OFT frozen smoke is only a load/interface/VRAM feasibility check. It answers narrow engineering questions:

- Can local paths resolve the checkpoint/cache?
- Can the environment import the required stack?
- Can the model load without running out of VRAM/system RAM?
- Can the adapter interface produce the expected shapes?

Frozen smoke is not a training trick, not a baseline result, not a rollout result, and not paper evidence for standard manipulation success. It must not be reported as performance evidence.

## Actual Low-Compute Tricks

The low-compute protocol relies on:

1. SmolVLA-first real adapter smoke.
2. Frozen backbone for local pilots.
3. Head-only ActionMap/TCA-Map training.
4. Cached hidden features so small heads can train without repeatedly running the backbone.
5. Low-resolution or coarse-to-fine heatmaps.
6. Required LoRA/QLoRA experimental tracks after head-only validation, using only small adapters and no full fine-tuning.
7. No OpenVLA-OFT full fine-tuning.

## Role Of OpenVLA-OFT

OpenVLA-OFT is kept only as:

1. A paper-grade reference target.
2. Optional frozen/load smoke.
3. Optional published-number context if benchmark conditions match exactly enough to be stated honestly.

Do not claim OpenVLA-OFT SOTA unless OpenVLA-OFT is directly reproduced under comparable conditions. Published numbers can motivate or contextualize results, but they cannot substitute for a direct baseline when making SOTA claims.

## Valid Publishable Claim

The intended publishable claim is:

> TCA-Map improves target-conditioned action decoding under strict compute constraints, preserving standard performance while improving counterfactual target grounding over ActionMap and native heads.

This claim still requires honest evidence:

- SmolVLA native head baseline.
- ActionMap baseline.
- ActionMap plus counterfactual augmentation baseline.
- TCA-Map head-only/frozen-backbone result.
- Offline proxy metrics clearly labeled as proxy metrics.
- Simulator rollout evidence before claiming standard manipulation success.

## Invalid Claims

Do not claim:

- OpenVLA-OFT SOTA without direct reproduction.
- Standard manipulation success from offline proxy metrics.
- Language grounding is solved.
- Real-world deployability without real-robot evidence.
- OpenVLA-OFT training or rollout results from frozen smoke.

## Local Compute Boundary

Local hardware can be used for development, readiness checks, cached feature preparation, head-only pilots, and small WSL2/Linux rollouts after simulator checks pass.

Cloud/remote GPU is required for large OpenVLA-OFT baselines, broad multi-seed sweeps, high-resolution heatmap training, and paper-grade large ablation matrices.
