# RIFA-XVLA Stage 0 Preregistration

- Decision: `RIFA_XVLA_STAGE0_PREREGISTERED_NOT_LAUNCHED`
- Selected method: `RIFA_XVLA`
- Next execution classification: `OURS_VLA_TRAINING`
- Ours training/rollout: none.

RIFA-XVLA is a reliability-conditioned imputed-feature adapter for X-VLA. It uses RL4IL imputed in-hand latent features and reliability metrics while preserving the frozen X-VLA base path.

## Required Stage 0 gates

- Real X-VLA forward path.
- Trainable parameter count greater than zero.
- CUDA tensors.
- Finite nonzero gradients.
- Optimizer steps and weight changes.
- Checkpoint write and disk reload.
- Base-preserving initialization.
- Missing-modality signal observability.
- Nonconstant RL4IL reliability features.
- Full versus no-reliability ablation output difference.
- Bounded action delta from frozen X-VLA.
- Clean validation retention diagnostic.

## Frozen scope

Freeze X-VLA base weights except declared adapter/LoRA parameters. Freeze the RL4IL prior checkpoints and CLIP encoders used as feature/reliability sources. No full-model fine-tuning, no CPU/disk offload, and no broad natural-reset mining.

Key ablation: `RIFA_XVLA_NO_RELIABILITY`, with the same adapter parameter count but reliability features/gating removed or fixed neutral.
