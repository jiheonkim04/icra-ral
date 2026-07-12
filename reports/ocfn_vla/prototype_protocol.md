# OCFN-VLA Prototype Protocol

Date: 2026-07-12 KST

## Method

OCFN-VLA supplies a task-conditioned initial noise tensor to the frozen SmolVLA flow sampler. It does not replace, filter, damp, rank, verify, or residual-correct the emitted action.

## Stage Order

1. Synthetic noise-prior smoke: verify deterministic noise-bank generation, task-conditioned selection, shuffled ablation behavior, and nonzero action differences in a tiny mock flow map.
2. Train acquisition: run 16 official SmolVLA-LIBERO train episodes to label noise identities.
3. Stage A: run 50 held-out official SmolVLA-LIBERO episodes across the five fixed variants.
4. Stage B if required by governance: run 40 paired official SmolVLA-LIBERO episodes per key policy on a disjoint fixed reset block, then expand once to at most 80 paired episodes per key policy only if Stage B remains unresolved.
5. Adjudicate under `reports/current_research_governance.md`.

## Fixed Variants

1. `frozen_smolvla`
2. `zero_noise_smolvla`
3. `global_success_noise_prior`
4. `task_shuffled_noise_prior`
5. `ocfn_full`

## Fixed Task Split

Train identities: `20260711`, `20260712`

Held-out identities: `20260713`, `20260714`, `20260715`, `20260716`, `20260717`

Stage B identities: `20260718` through `20260737` for the first 40 paired episodes per key policy. If the Stage B decision is unresolved, the one allowed expansion extends the same manifest prefix through `20260757` for 80 paired episodes per key policy.

Tasks: `libero_spatial/task_4`, `libero_10/task_4`

## Required Artifacts

- `reports/ocfn_vla/synthetic_result.json`
- `reports/ocfn_vla/synthetic_result.md`
- `reports/ocfn_vla/train_acquisition_result.json`
- `reports/ocfn_vla/train_acquisition_result.md`
- `reports/ocfn_vla/stage_a_partial_result.json`
- `reports/ocfn_vla/stage_a_result.json`
- `reports/ocfn_vla/stage_a_result.md`
- `reports/ocfn_vla/stage_b_partial_result.json`
- `reports/ocfn_vla/stage_b_result.json`
- `reports/ocfn_vla/stage_b_result.md`

## Measurement Validity

The run is invalid if:

- the custom noise tensor is not passed to `policy.select_action`;
- train and held-out identities are mixed;
- Stage A changes the selection rule after train acquisition;
- Stage B changes the selection rule after train acquisition;
- any variant uses simulator success, reward, reset identity, or object pose at inference;
- any Stage A or Stage B held-out episode exception is caused by an implementation error that affects only one method variant.

## Final Stage A Decision

Use the preregistered GO/KILL criteria in `reports/ocfn_vla/preregistration.md`.
