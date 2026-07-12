# Epoch 1 Corrected Adjudication

Date: 2026-07-12 KST

The previous fixed-cycle terminal decision is reclassified as `EPOCH_1_COMPLETED_PIVOT_REQUIRED`. Epoch 1 produced useful negative evidence, but not a valid global no-method terminal state.

## Cycle 1: DICD-VLA

Corrected status: `UNDERPOWERED_STAGE_A_NON_GO_ARCHIVED`

Evidence:

- full DICD: `1 / 10`
- direct chunk-index delay baseline: `2 / 10`
- no-history ablation: `1 / 10`

Reason:

Ten episodes per policy cannot justify a permanent scientific kill from a one-episode difference. The current formulation should not be rerun or rescued, but the result is an underpowered non-GO archive rather than evidence that the whole route space is dead.

## Cycle 2: FEDO-VLA

Corrected status: `VALID_CURRENT_FORMULATION_KILL`

Evidence:

- faulted full FEDO: `1 / 10`
- static inverse gain: `2 / 10`
- APEX-style feedback proxy: `2 / 10`
- no-feedback ablation: `2 / 10`
- clean frozen SmolVLA: `4 / 10`
- clean FEDO full: `0 / 10`

Reason:

The full method lost to baselines and ablation under faults, and clean performance collapsed from `4 / 10` to `0 / 10`. Do not revive this formulation.

## Cycle 3: GCAP-VLA

Corrected status: `UNDERPOWERED_TARGET_AXIS_NON_GO_ARCHIVED`

Evidence:

- occluded full GCAP: `3 / 10`
- occluded frozen SmolVLA: `4 / 10`
- Sobel edge boost: `5 / 10`
- no-temporal ablation: `4 / 10`
- clean frozen SmolVLA: `1 / 10`
- clean GCAP full: `5 / 10`

Reason:

Ten episodes per policy cannot justify a permanent family kill. Full GCAP was below frozen and Sobel on the target occlusion axis, so this formulation is non-GO and should not be rescued. Clean GCAP outperforming clean frozen creates unresolved variance or mechanism evidence, so the broader perception-repair family is not dead.

## Epoch 1 Result

Epoch 1 completed three related non-GO/current-formulation failures. Governor C requires an Epoch 2 pivot that changes at least two core dimensions relative to DICD, FEDO, and GCAP.
