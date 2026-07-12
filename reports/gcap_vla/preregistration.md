# GCAP-VLA Preregistration

Date: 2026-07-12 KST

Proposal frozen before Stage A.

## Hypothesis

Patchwise temporal repair plus local geometric-continuity enhancement improves frozen SmolVLA closed-loop success under controlled camera occlusion more than simple hold-last or edge-only baselines, while preserving clean behavior.

## Fixed Evaluation

- backbone: official frozen SmolVLA-LIBERO
- tasks: `libero_spatial/task_4`, `libero_10/task_4`
- evaluation identities: `20260713,20260714,20260715,20260716,20260717`
- variants: `occluded_frozen_smolvla`, `full_frame_hold_last`, `sobel_edge_boost`, `gcap_no_temporal_ablation`, `gcap_full`, `clean_frozen_smolvla`, `clean_gcap_full`
- planned episodes: `70`
- primary metric: task-balanced closed-loop success
- secondary metrics: exception count, mean mask fraction, clean retention

## GO

`gcap_full` reaches GO only if:

- zero rollout exceptions,
- task-balanced success is at least 5 absolute points above the strongest occluded non-ablation baseline,
- it beats `full_frame_hold_last`,
- it beats `gcap_no_temporal_ablation`,
- clean retention drop is at most 2 absolute points.

## KILL

- hold-last matches or beats full: `SIMPLE_TEMPORAL_BASELINE_EXPLAINS_METHOD`
- no-temporal ablation matches or beats full: `TEMPORAL_COMPONENT_NOT_USEFUL`
- full does not beat occluded frozen: `NO_OCCLUSION_ROBUSTNESS_GAIN`
- clean drop exceeds 0.02: `CLEAN_RETENTION_FAILURE`
- exceptions occur: `MEASUREMENT_INVALID_REPAIR_OR_KILL`

One narrow measurement repair is allowed only for concrete code/runtime bugs. No threshold or occlusion-schedule changes are allowed after outcome inspection.
