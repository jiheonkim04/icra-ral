# SACF-VLA Preregistration

Date: 2026-07-12 KST

Proposal hash: `1C43D99A42AD97C29C1BDBDED1AB1326214C8FF0F514F79309266738C5FD1A20`

Decision before implementation: `IMPLEMENT_STAGE_A_PROTOTYPE`

## Fixed Variants

1. `frozen_smolvla`
2. `task_phase_mean_prefix`
3. `plain_bc_prefix`
4. `cag_null_guidance`
5. `sacf_full`

## Fixed Stage A Task/Reset Allocation

Tasks:

- `libero_spatial/task_4`
- `libero_object/task_4`

Reset identities:

- `20260713`
- `20260714`
- `20260715`
- `20260716`
- `20260717`

Total Stage A episodes:

- `5 variants * 2 tasks * 5 identities = 50 episodes`

## Fixed Training Data

Use standard local LIBERO HDF5 demos for:

- `libero_spatial`
- `libero_object`

Do not use LIBERO-90 in Stage A.

Do not use HDF5 object-state fields as default inference features.

## Fixed Hyperparameters

- semantic slot feature width: `16`
- phase bins: `8`
- hidden size: `64`
- training epochs: `160`
- learning rate: `0.003`
- factor loss weight: `0.35`
- shared invariance loss weight: `0.05`
- prefix fraction: `0.35`
- CAG null-guidance scale: `0.5`
- max training rows per task: `240`
- train demos per task: all available up to row cap

No hyperparameter may be changed after Stage A result inspection.

## Metrics

Primary:

- task-balanced closed-loop success rate.

Secondary:

- successes/counts per variant;
- per-task success rate;
- factor activation norm;
- full-vs-plain action delta during prefix;
- CAG full-vs-null action delta;
- latency and CUDA memory;
- loss decrease and checkpoint hashes.

## GO / KILL

Use `reports/current_research_governance.md`.

Stage A permanent kill if:

- implementation or data mechanism invalid;
- `sacf_full` is at least 30 absolute task-balanced points below the strongest baseline or `plain_bc_prefix`;
- `sacf_full` has `0 / 10` while a paired baseline has at least `4 / 10`;
- exact trivial equivalence to `plain_bc_prefix`, `task_phase_mean_prefix`, or `cag_null_guidance` is demonstrated.

Advance to Stage B if:

- `sacf_full` beats frozen and `plain_bc_prefix`, or
- result is noisy/tied/small-negative but mechanism activation is valid and no permanent-kill condition holds.

## One Allowed Measurement Repair

If the first real-demo training shows an action-dimension or action-range convention mismatch, one repair is allowed before Stage A:

- document the mismatch;
- fix preprocessing/postprocessing once;
- rerun synthetic and real-demo training;
- do not inspect Stage A before the repair.
