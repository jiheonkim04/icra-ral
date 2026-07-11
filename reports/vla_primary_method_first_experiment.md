# ECHO-VLA First Decisive Prototype Experiment

Date: 2026-07-11 KST

This is a future experiment design only. It was not run.

## Decisive Question

Does the counterfactual effect-credit mechanism itself improve closed-loop task success over the strongest simple local baselines?

## Backbone

Primary first backbone: official SmolVLA-LIBERO.

OpenVLA-OFT INT4 is not required until the method survives this first gate.

## Exact Tasks

Select tasks for predicate diversity, not because of previous SmolVLA-only failures:

| Suite | Task ID | Instruction | Predicate role |
| --- | ---: | --- | --- |
| `libero_spatial` | `0` | pick up the black bowl between the plate and the ramekin and place it on the plate | spatial grasp and place |
| `libero_object` | `4` | pick up the ketchup and place it in the basket | object-conditioned pick/place |
| `libero_goal` | `0` | open the middle drawer of the cabinet | articulated object / state change |
| `libero_10` | `0` | put both the alphabet soup and the tomato sauce in the basket | multi-object sequential effects |

Task IDs and instructions are from `reports/official_closed_loop_task_manifest.json`.

## Exact Policies

- `frozen_smolvla`: official frozen SmolVLA-LIBERO policy.
- `standard_adaptation`: standard backbone adaptation only if the implementation goal already has an approved local adaptation path; otherwise this baseline is recorded as not applicable.
- `heuristic_effect`: hand-coded predicate-progress heuristic using current visual/predicate estimates but no counterfactual effect model.
- `progress_value_head`: small progress/value head trained on the same labels.
- `pre_vla_style_head`: local safety/advantage-style validity head using the same data budget.
- `echo_no_counterfactual`: ECHO architecture without matched-context counterfactual ranking.
- `echo_full`: full ECHO-VLA.

## Exact Data

- Demonstrations: official LIBERO demonstrations for the four selected tasks.
- Training labels: chunk-level predicate deltas from BDDL/simulator state.
- Counterfactuals: maximum `4` counterfactual chunks per positive chunk:
  - action jitter,
  - same-suite wrong-phase chunk,
  - nearest action chunk with different predicate delta,
  - optional one short simulator-evaluated chunk when safe in the later implementation goal.
- Initial cap: `<= 2,000` labeled chunk examples total.

## Training Budget

- ECHO head only for first gate.
- Maximum wall-clock target: `<= 6 hours` training after data labels exist.
- Maximum epochs: `20`.
- Early stopping metric: validation effect F1 and pairwise effect-ranking accuracy.
- No large model download.
- No full benchmark.

## Rollout Episode Count

- `4` tasks.
- `10` official initial states per task.
- `3` rollout seeds per policy/task when stochastic.
- Maximum first-gate episodes: `4 tasks * 10 init states * 3 policy variants of interest = 120` for the final local comparison, with a smaller `40` episode smoke allowed first.

## Baselines Required

- Frozen backbone.
- Standard adaptation if relevant.
- Simplest heuristic effect/progress version.
- Strongest direct recent local baseline: Pre-VLA-style validity/advantage head or progress/value head, whichever is stronger in validation.
- Ablation without counterfactual ranking.

## Metrics

Primary:

- closed-loop task success.

Secondary:

- task-balanced success,
- predicate-effect success rate,
- phase-required effect F1,
- action-effect calibration ECE/Brier,
- intervention/candidate count,
- policy forward passes,
- ECHO forward passes,
- latency,
- VRAM.

## Runtime Estimate

- Label extraction: `2-6 hours` once implementation exists.
- Head training: `1-6 hours`.
- First smoke rollouts: `2-4 hours`.
- Final first-gate rollouts: `8-18 hours` depending on simulator speed.
- Total target: approximately `1-2 days`.

## Success Threshold

Support the method if:

- `echo_full` beats the strongest simple baseline by at least `5` absolute success points on task-balanced closed-loop success,
- `echo_full` beats `echo_no_counterfactual`,
- effect F1 improves and correlates with success,
- latency overhead remains `<20%` or is explicitly justified by success gain.

## Kill Threshold

Kill or redesign if:

- `echo_full` does not beat `echo_no_counterfactual`,
- a progress/value/verifier head matches `echo_full`,
- task success does not improve,
- improvement appears only in offline effect metrics,
- gains depend on one task or favorable seed selection,
- privileged simulator labels are needed at inference.
