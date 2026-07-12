# FEDO-VLA Preregistration

Date: `2026-07-12 KST`

## Frozen Method

FEDO-VLA predicts an additive residual command from:

- current frozen SmolVLA action;
- previous sent command;
- previous measured realized action;
- previous execution error;
- normalized rollout step fraction;
- task role and coarse phase.

Inference may not use simulator state, object pose, success predicates, BDDL predicates, or ground-truth target labels.

## Fault Condition

The prototype uses a deterministic action-realization wrapper. It maps sent command `u_t` to realized action `e_t` with phase-dependent damping and small bias on translation and gripper dimensions. The policy receives observations from the environment after `e_t` is applied.

The measured realized action `e_t` is available to every variant that uses feedback. This models low-level controller/action feedback, not a simulator success oracle.

## Tasks And Identities

Tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

Evaluation identities:

- `20260713`
- `20260714`
- `20260715`
- `20260716`
- `20260717`

## Variants

1. `faulted_frozen_smolvla`
2. `static_inverse_gain`
3. `apex_feedback_proxy`
4. `fedo_no_feedback_ablation`
5. `fedo_full`

## Primary Metric

Task-balanced closed-loop success rate under the faulted condition.

## Secondary Metrics

- per-task success;
- pairwise full-vs-baseline win/loss/tie over identical task/identity pairs;
- clean no-fault retention on the same task/identity manifest for `faulted_frozen_smolvla` and `fedo_full`;
- mean residual norm;
- mean realized action error norm.

## GO Criteria

Prototype GO requires all:

- `fedo_full` beats the strongest faulted prototype baseline by at least 5 absolute task-balanced success points;
- `fedo_full` beats `apex_feedback_proxy`;
- `fedo_full` beats `static_inverse_gain`;
- `fedo_full` beats `fedo_no_feedback_ablation`;
- clean retention drop is not more than 2 absolute task-balanced points in the small retention check;
- zero rollout exceptions or only measurement-invalid exceptions that receive one narrow repair before result adjudication.

## Kill Criteria

Kill with `SIMPLE_BASELINE_EXPLAINS_METHOD` if `static_inverse_gain >= fedo_full`.

Kill with `DIRECT_PRIOR_EXPLAINS_METHOD` if `apex_feedback_proxy >= fedo_full`.

Kill with `KEY_COMPONENT_NOT_USEFUL` if `fedo_no_feedback_ablation >= fedo_full`.

Kill with `NO_FAULT_ROBUSTNESS_GAIN` if `fedo_full <= faulted_frozen_smolvla`.

Kill with `CLEAN_RETENTION_FAILURE` if clean retention degrades materially.

## Budget

Maximum Cycle 2 wall time: `12 h`.

Maximum total remaining campaign GPU budget after Cycle 1: approximately `21.38 h`.

The prototype must checkpoint partial closed-loop rows if it is expected to exceed `4 h`.
