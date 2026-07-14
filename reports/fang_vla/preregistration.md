# FANG-VLA Preregistration

Date: 2026-07-14 KST

Proposal hash: `6837DBA2A1307F7C9938FA9F5463ED483907AF3C168F1C0514F6E281804E859B`

Decision before implementation: `CONDITIONAL_PROCEED_TO_DEVELOPMENT_AUDIT`.

## Fixed Tasks

Use exactly two hard local SmolVLA tasks for the first prototype:

1. `libero_spatial/task_4`
2. `libero_10/task_4`

Rationale: these are the established hard tasks from prior frozen SmolVLA and CAVM work, and the existing development traces contain both success and failure labels for both tasks. They are not selected from FANG outcomes.

## Evidence Partitions And Identities

Discovery/development source:

- existing CAVM acquisition identities: `20260901..20260912`;
- existing CAVM calibration identities: `20260913..20260916`;
- these are non-confirmatory for FANG and may be used only for audit, training, and validation.

FANG internal split:

- `DISCOVERY_TRAIN`: identities `20260901..20260910`;
- `VALIDATION`: identities `20260911..20260916`.

Forbidden for FANG training and validation:

- CAVM Stage 2A identities `20260917..20260921`;
- CAVM Stage 2B and expansion identities `20260922..20260950`;
- any future FANG confirmatory identities.

FANG confirmatory identities, if audit and validation pass:

- Stage A: `20260951..20260960`;
- Stage B: `20260961..20261000`;
- optional one-time expansion, only if preregistered unresolved: `20261001..20261040`.

No confirmatory identity may be used to train, validate, calibrate, select a configuration, choose thresholds, or debug the method.

## Features

Inference feature:

`x_t = [q_t, a_t, a_{t-1}, rho_t, task_one_hot]`.

Dimension: `25`.

Allowed inference values:

- official 8D proprioceptive state;
- current frozen SmolVLA 7D base action;
- previous executed 7D action;
- chunk-index fraction;
- task key.

Forbidden inference values:

- simulator object pose;
- reward before terminal logging;
- BDDL predicate;
- current episode success label;
- future actions;
- held-out identity membership;
- any feature derived from confirmatory outcomes.

## Stage 0: Development Audit

Run before training beyond a tiny smoke and before any closed-loop FANG rollout.

Hard stop as `DATA_FAILURE` if:

- either selected task lacks at least `2` success identities and `2` failure identities in discovery/validation;
- either class has fewer than `250` trace rows total across selected tasks;
- success/failure labels are all-zero or all-one in any task;
- duplicate `(split, task_key, identity, step)` rows exist;
- any confirmatory identity appears in audit/training/validation.

Hard stop as `NO_HEADROOM` if:

- success/failure class-conditional action-field separation is below the preregistered minimum on validation;
- a trivial nearest-success or class-mean diagnostic explains all available target variance.

Hard stop as `IMPLEMENTATION_FAILURE` if:

- checkpoint save/reload fails;
- expected trainable parameters receive zero, NaN, or Inf gradients;
- model outputs invalid actions on the validation batch.

Hard stop as `DESIGN_FAILURE` if:

- validation action deltas are globally destructive;
- gate activates almost everywhere with large residuals;
- clean retention proxy is unacceptable before rollout.

## Bounded Validation Search

Search budget:

- exactly six maximum configurations;
- one architecture: MLP trunk width `64`, depth `2`;
- one seed for initial lightweight training;
- no more than two seeds if a configuration is rerun for implementation repeatability;
- no additional coefficient grid;
- no confirmatory identities.

Configurations:

| Config | `alpha` | `lambda_delta` | `lambda_gate_fit` | `lambda_gate_sparse` | `beta` |
| --- | ---: | ---: | ---: | ---: |
| `fang_c01` | 0.10 | 0.10 | 1.00 | 0.01 | 0.50 |
| `fang_c02` | 0.20 | 0.10 | 1.00 | 0.01 | 0.50 |
| `fang_c03` | 0.35 | 0.10 | 1.00 | 0.01 | 0.50 |
| `fang_c04` | 0.10 | 0.30 | 1.00 | 0.01 | 0.50 |
| `fang_c05` | 0.20 | 0.30 | 1.00 | 0.01 | 0.50 |
| `fang_c06` | 0.35 | 0.30 | 1.00 | 0.01 | 0.50 |

Gate reliability target:

- computed from discovery records only;
- uses same-task success/failure neighbor action-field separation;
- `eta = 0.05`;
- `gamma = max(q75 - q25, 0.05)` over discovery separations;
- never computed from confirmatory identities or outcomes.

Gate threshold calibration:

- for each trained candidate, compute validation gate logits;
- set `tau` deterministically so the fraction with `sigmoid(logit - tau) > 0.05` is approximately `0.50`;
- save the selected `tau`;
- do not search over multiple target activation fractions;
- do not recalibrate `tau` on confirmatory identities or outcomes.

Validation score:

`score = 0.35 * mechanism_separation + 0.25 * clean_retention + 0.20 * action_validity + 0.10 * bounded_activation + 0.10 * compute_efficiency`.

Definitions:

- `mechanism_separation`: `min(median(||m_plus(x) - m_minus(x)||_2) / 0.10, 1)`.
- `clean_retention`: `1 - clipped(mean_delta_l2 / 0.20, 0, 1)`.
- `action_validity`: fraction of validation actions within action bounds.
- `bounded_activation`: `1` when gate activation is between `0.05` and `0.60`; below `0.05` use `gate_activation / 0.05`; above `0.60` use `max(0, 1 - (gate_activation - 0.60) / 0.40)`.
- `compute_efficiency`: `1` for a single lightweight head and no extra VLA calls.

Select exactly one final configuration by highest validation score, then freeze it. Save all tried configurations and negative results.

## Fixed First-Comparison Policies

Exactly five policies:

1. `base_smolvla`
2. `afil_local_proxy`
3. `fang_full`
4. `fang_no_failure_ablation`
5. `nearest_success_replay`

Definitions:

- `base_smolvla`: unmodified frozen SmolVLA.
- `afil_local_proxy`: dual success/failure action-field guidance without validation-calibrated identity-preserving gate. It is not an official AFIL reproduction.
- `fang_full`: full selected FANG configuration.
- `fang_no_failure_ablation`: success residual and gate only; no failure residual in the action.
- `nearest_success_replay`: same CAVM-style nearest successful same-task action blending, with no failure model.

## Stage A

Run `10` paired episodes per policy:

- identities `20260951..20260960`;
- two tasks;
- five policies;
- total `100` episodes if each identity maps to both tasks for each policy.

Stage A may kill only for:

- mechanism invalidity;
- no headroom;
- catastrophic degradation;
- clear prior or ablation dominance;
- exact trivial equivalence;
- privileged inference violation.

Small differences advance to Stage B.

## Stage B

Run at least `40` paired episodes per key policy:

- identities `20260961..20261000`;
- two tasks;
- five policies;
- matched manifest and identical reset allocation.

Report:

- successes/counts;
- task-balanced success;
- per-task breakdown;
- paired wins/losses/ties;
- bootstrap confidence interval;
- failure-rate reduction;
- mechanism activation;
- clean retention;
- latency, VRAM, and forward-pass count.

## Prototype GO

`fang_full` reaches prototype GO only if:

- it beats `base_smolvla`;
- it beats `afil_local_proxy`;
- it beats `fang_no_failure_ablation`;
- it beats `nearest_success_replay`;
- the gain satisfies active governance Stage B useful-improvement criteria or its one allowed unresolved expansion;
- clean validation behavior is retained;
- the intended mechanism evidence supports failure-aware residual guidance.

## Kill Criteria

Kill the current formulation if:

- audit returns `DATA_FAILURE`, `NO_HEADROOM`, `IMPLEMENTATION_FAILURE`, or `DESIGN_FAILURE`;
- `afil_local_proxy`, `fang_no_failure_ablation`, or `nearest_success_replay` matches or beats `fang_full` in Stage B;
- `fang_full` is clearly worse than Base;
- upper confidence bound excludes useful improvement after Stage B or the one allowed expansion;
- clean retention fails;
- any confirmatory outcome is used to retune the method.

No third expansion is allowed.
