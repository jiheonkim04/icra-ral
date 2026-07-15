# G3P-VLA Researcher A Rebuttal

Date: 2026-07-15 KST

Frozen proposal reviewed: `reports/g3p_vla/researcher_proposal.md`

Proposal hash: `BEE3822D8F54EFBD09C1CA47A9BF126EBE694B7B6219002FF770C5794ED7AA71`

Reviewer attack: `reports/g3p_vla/reviewer_attack.md`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Rebuttal decision: `G3P_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

## Response Summary

Researcher A accepts the Reviewer B constraints. G3P-VLA will not claim broad novelty for grounded 3D point injection, gripper-relative displacement, or action-head conditioning. The only claim preserved for the local method is:

frozen SmolVLA can benefit from a legally deployable, source-gated, identity-preserving 3D point-conditioning adapter if Stage 0 proves the point source is observable, noncollapsed, split-clean, useful, and not explained by a simple 2D or phase heuristic.

No implementation, training, validation search, rollout, or confirmatory-test tuning is authorized by this rebuttal. The next permitted step is mathematical mechanism audit.

## Accepted Novelty Narrowing

Accepted. The closest prior already contains the main 3D point displacement and action-head injection idea. G3P's local contribution is restricted to:

- frozen SmolVLA integration;
- source-gated proof that no privileged point is used at inference;
- identity-preserving adapter initialization;
- bounded 7D action deltas on the local SmolVLA action interface;
- matched comparison to `g3p_3d_point_proxy`, `g3p_no_3d_no_injection_ablation`, and `simple_2d_phase_or_nearest_object_heuristic`.

If the closest-prior proxy matches or beats G3P on the frozen claim axis, the local G3P contribution is killed or archived. The proxy will remain labeled as a faithful transparent local proxy unless official code/checkpoint/protocol equivalence is separately verified before confirmatory testing.

## Accepted Source-Legality Gate

Accepted. Stage 0 must prove source legality before training, validation search, or rollout.

Inference may use only:

- official deployment RGB observations;
- language instruction;
- proprioception available to the policy, including end-effector state if exposed by the official runner;
- frozen Base features/actions available through the local deployment path.

Inference may not use:

- simulator object pose;
- placement coordinates;
- reward or success;
- reset identity;
- future observations;
- confirmatory-test task/reset metadata;
- oracle labels or target coordinates hidden outside deployment inputs.

Oracle geometry may be used only on discovery/validation data to create labels, inspect headroom, and audit source predictability. Any source-gate failure is a pre-rollout `DATA_OR_SUPERVISION_FAILURE`, `DESIGN_FAILURE`, or `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, not a closed-loop scientific result.

## Accepted Trivial-Baseline Constraints

Accepted. The first serious comparison remains exactly:

1. `frozen_smolvla`
2. `g3p_3d_point_proxy`
3. `g3p_full`
4. `g3p_no_3d_no_injection_ablation`
5. `simple_2d_phase_or_nearest_object_heuristic`

The simple heuristic remains live through Stage A/B. Validation may choose the single strongest simple 2D, phase, or nearest-object heuristic before confirmatory testing, but confirmatory outcomes may not alter it.

If the simple heuristic accounts for G3P's gain, G3P is not a paper candidate.

## Accepted Coordinate And Calibration Requirements

Accepted. The mathematical audit must define:

- point coordinate frame;
- gripper coordinate frame;
- transform path from deployable point estimate to gripper-relative vector;
- units and normalization;
- sign convention for `d = p_t - p_g`;
- action component bounds;
- translation, rotation, and gripper delta limits.

Stage 0 must sanity-check the direction and scale of example vectors on development identities before any rollout. A sign, frame, or unit failure stops as `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.

## Accepted SmolVLA Interface Constraints

Accepted. G3P is not allowed to arbitrarily replace strong pretrained actions.

Before rollout, G3P must show:

- initial Base passthrough;
- checkpoint disk reload;
- finite nonzero gradients in intended point/adaptation parameters;
- bounded translation, rotation, and gripper deltas;
- clean validation retention;
- mechanism activation in relevant states rather than everywhere;
- no privileged inference input.

Global destructive action change, gripper collapse, or all-state activation is an implementation/design failure, not a valid closed-loop result.

## Accepted Data-Health And Observability Requirements

Accepted. Stage 0 must reject if:

- labels are collapsed by task, phase, object, or frame;
- confidence is always high or always low;
- target and placement cases are indistinguishable;
- point prediction cannot beat trivial deployment-observable baselines;
- train/validation/test identities overlap;
- oracle headroom shows no useful spatial gain;
- full, proxy, ablation, or simple heuristic are action-equivalent before rollout.

The point predictor must be evaluated against majority, task/language-only, phase, 2D, and nearest-object baselines where locally available.

## Accepted Mathematical Objective Restrictions

Accepted. The mathematical audit must not use KL between deterministic 7D actions. It must document variables, tensor shapes, objective scales, units, gradient paths, small-batch loss magnitudes, gradient norms, and required ablations before implementation.

The audit must specify whether the point predictor is:

- trained jointly;
- frozen after validation-only selection;
- generated offline from development labels;
- or replaced by a nontrainable source gate for the closest-prior proxy.

## Frozen Next Step

Proceed to mathematical mechanism audit only.

The audit must be completed and committed before any implementation, Stage 0 data construction, validation search, training, manifest freeze, or rollout.
