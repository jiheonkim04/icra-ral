# Ours Candidate Selection After RL4IL Prior

- Prerequisite prior decision: `RL4IL_ACTION_ORACLE_PRIOR_LOCAL_RESIDUAL_ESTABLISHED`
- Candidate count: `2`
- Selected method: `RIFA_XVLA`
- Ours training/rollout: none.

The RL4IL local prior improved the frozen dropout condition from `0/9` to `3/9`, but left a residual: goal/task0 and spatial/task5 still failed under mask_1, and clean retention was only `4/9`.

## Candidates

1. `RIFA_XVLA`: Reliability-conditioned Imputed-Feature Adapter for X-VLA.
   - Core: learned adapter conditions X-VLA action hidden states on missing-camera status, RL4IL imputed in-hand latent features, and reliability signals such as donor dispersion, retrieval descriptor margin, and policy entropy.
   - Base-preserving init: zero residual projections; clean gates initially prefer frozen X-VLA.
   - Low-compute path: freeze X-VLA and train only LoRA/adapter/gating parameters.

2. `CVLR_XVLA`: Cross-View Latent Reconstruction Adapter for X-VLA.
   - Core: predict missing wrist-view visual latents from agent-view, language, and proprioception, then inject them into X-VLA’s visual-token stream.
   - Not selected because it requires more invasive token-stream surgery and uses the validated RL4IL prior less directly.

## Selection

`RIFA_XVLA` is selected because it is the narrower learned method around the observed residual. It keeps the VLA closed-loop action path, uses the RL4IL prior as a reliability-aware feature source, and is easier to test under the low-compute adapter constraints.

First comparison roles remain: frozen X-VLA, `MECHANISM_FAITHFUL_RL4IL_LOCAL_PORT`, `RIFA_XVLA`, RIFA no-reliability ablation, and AWF-XVLA as `SIMPLE_CAMERA_IMPUTATION_CONTROL`.

Next action: preregister `RIFA_XVLA` Stage 0 before any Ours VLA training.
