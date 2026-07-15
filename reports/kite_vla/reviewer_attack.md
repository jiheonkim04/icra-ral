# KITE-VLA Reviewer B Attack

Date: 2026-07-15 KST

Frozen Researcher proposal SHA-256:
`FA00DE56D14E4C69388BE1642F7D52153841D58E77FD5A3F5C68B6C624A152B8`.

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`.

## Prior-Art Attack

GeoPredict already demonstrates positive predictive-kinematics supervision in
VLA manipulation. StyleVLA already uses kinematic consistency in a VLA driving
domain. PointWorld and predictive world-model work condition future state on
action sequences. KITE may claim neither future-state prediction nor generic
kinematic consistency as new.

The provisional novelty survives only as this precise mechanism:

1. fit one task-agnostic, discovery-only manipulation realization operator;
2. apply it to the generated clean SmolVLA action chunk;
3. backpropagate measured future-state realization error through that action;
4. remove the operator entirely at inference.

No exact primary source was found that combines those four elements for a
continuous-flow manipulation VLA. Novelty remains provisional until the final
search.

## Mathematical Attack

With affine `F_H`, squared or locally quadratic Huber realization loss is a
weighted cumulative-action loss. If future state displacement is almost fully
determined by command sums, KITE may add no information beyond directly
matching cumulative demonstration actions.

Essential response: replace the proposal's endpoint-only ablation with a
matched multi-horizon cumulative-action-target ablation. It must use the same
horizons, normalization, LoRA, data, steps, and coefficient while omitting
`ee_states` and `F_H`. KITE must beat it.

## Operator Attack

- `F_H` must be one global task-agnostic operator per horizon.
- Task identity, language, image features, validation data, and confirmatory
  data may not enter fitting.
- Discovery standardization, ridge coefficient, coefficients, intercept, and
  source hashes must persist before validation scoring.
- The design must report matrix rank, singular values, per-task validation
  error, and the discovery-mean baseline.
- Do not call `F_H` forward kinematics or SE(3) physics. It is an empirical
  local action-to-`ee_states` realization operator.

## Data And Leakage Attack

Future `ee_states` are privileged training labels. They are legal only if:

- absent at inference;
- drawn from the same aligned demonstration;
- masked at episode boundaries;
- never sourced from confirmatory rollouts;
- split before operator fitting;
- not used to select tasks or resets.

All serialized normalization and operator values must be converted to ordinary
JSON lists before hashing. The HASTE NumPy serialization defect makes a
round-trip hash/parse test essential before the Stage 0A worker launch.

## Headroom Attack

High operator fit alone does not show policy headroom. Stage 0A must compare
the frozen Base reconstructed clean chunk's realized-state error against the
demonstrated action operator residual on the same validation rows. A global
average cannot hide a task with no rows or nonfinite behavior.

## Objective And Gradient Attack

Before optimization report:

- `L_flow` and `L_kite` magnitudes;
- LoRA gradient norms from each term;
- finite nonzero KITE gradient into expected action-expert LoRA targets;
- gradient ratio;
- action-unit unnormalization identity;
- Base/initialized/reloaded flow and action equality.

KITE is invalid if its loss only trains a discarded head or frozen operator.

## Baseline Decision

The exact five policies must be:

1. Base;
2. transparent GeoPredict-style kinematics proxy;
3. KITE;
4. cumulative-action-target ablation;
5. standard LoRA.

Standard LoRA is essential because all trainable methods receive the same
demonstrations and KITE uses LoRA infrastructure. No sixth policy is justified.

## False-Negative Safeguard

A Stage 0A failure is not a scientific kill. Distinguish:

- `DATA_FAILURE`: invalid alignment, collapsed target, or unusable operator;
- `NO_HEADROOM`: valid operator but no Base realization deficit;
- `DESIGN_FAILURE`: legal objective cannot send useful gradient through
  generated actions;
- `IMPLEMENTATION_FAILURE`: hash, serialization, identity, persistence,
  processor, or execution defect;
- `LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT`: demonstrated rank-4 capacity
  bottleneck after a valid micro-fit.

Only the last class permits one preregistered rank-8 capacity diagnostic, and
only before confirmatory testing with the scientific method unchanged.

## Essential Conditions

Pass to mathematical audit only if Researcher A accepts:

1. the cumulative-action-target ablation;
2. task-agnostic discovery-only operators;
3. narrow empirical-realization terminology;
4. explicit privileged-label masking;
5. prelaunch JSON round-trip validation;
6. objective magnitude and gradient-path audit;
7. exact five-policy comparison;
8. no HASTE repair or event supervision.
