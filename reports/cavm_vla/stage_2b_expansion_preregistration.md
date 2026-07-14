# CAVM-VLA Stage 2B Expansion Preregistration

Date: 2026-07-14 KST

Trigger: Stage 2B returned `STAGE_2B_UNRESOLVED_EXPANSION_OPTIONAL`.

Original Stage 2B evidence:

- `cavm_full`: `16 / 40`, task-balanced `0.40`
- strongest baseline `nearest_success_replay`: `15 / 40`, task-balanced `0.375`
- `frozen_smolvla`: `14 / 40`, task-balanced `0.35`
- `success_only_memory_proxy`: `13 / 40`, task-balanced `0.325`
- `cavm_no_contrast_ablation`: `13 / 40`, task-balanced `0.325`
- paired full minus nearest-success: delta `0.025`, CI `[-0.10, 0.15]`

## Expansion Scope

Use the remaining unused official initial-state indices under `CAVM_RESET_IDENTITY_BASE = 20260901`.

Additional identities:

- `20260942..20260950`
- exact official indices: `41..49`

Additional episodes:

- `9` identities x `2` tasks x `5` variants = `90` episodes
- `18` additional paired episodes per variant

Combined maximum after expansion:

- `58` paired episodes per variant
- `290` total Stage 2B plus expansion episodes

The active governance permits one expansion to at most `80` paired episodes per key policy. The local official initial-state inventory leaves only `18` additional paired cases without reusing acquisition, calibration, Stage 2A, or original Stage 2B official indices. Reusing training/calibration official indices is forbidden for CAVM because memory was fit from those traces.

## Fixed Variants

Same variants as Stage 2B:

1. `frozen_smolvla`
2. `success_only_memory_proxy`
3. `nearest_success_replay`
4. `cavm_no_contrast_ablation`
5. `cavm_full`

No hyperparameters, memory records, gates, clipping constants, tasks, or baselines may change.

## Final Expansion Decision Rules

After combining original Stage 2B and the expansion:

`STAGE_2B_EXPANDED_PROTOTYPE_GO` only if:

- `cavm_full` beats the strongest baseline and `cavm_no_contrast_ablation`;
- absolute task-balanced gain over the strongest baseline is at least `0.10`, or paired evidence is consistently positive with meaningful failure-rate reduction;
- mechanism activation remains positive;
- no privileged inference signal is used;
- heavy policy calls per step are not above frozen SmolVLA.

`STAGE_2B_EXPANDED_PERMANENT_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT` if any of:

- `success_only_memory_proxy` matches or beats `cavm_full`;
- `nearest_success_replay` matches or beats `cavm_full`;
- `cavm_no_contrast_ablation` matches or beats `cavm_full`.

`STAGE_2B_EXPANDED_PERMANENT_KILL_WORSE_THAN_FROZEN` if `cavm_full` is below frozen SmolVLA.

`STAGE_2B_EXPANDED_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED` if `cavm_full` does not beat the strongest baseline and the paired upper confidence bound versus that baseline is at most `0.10`.

`STAGE_2B_EXPANDED_NON_GO_NO_THIRD_EXPANSION` if the result remains positive but too weak for GO after this maximum feasible expansion.

No third expansion, threshold retuning, memory rebuild, task swap, identity reuse, or CAVM rescue is allowed.
