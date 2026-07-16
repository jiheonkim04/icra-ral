# CSPR-VLA Researcher A Rebuttal

Date: 2026-07-16 KST

Decision: `CSPR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Method: `CSPR-VLA`

Proposal: `reports/cspr_vla/researcher_proposal.md`

Reviewer attack: `reports/cspr_vla/reviewer_attack.md`

Proposal SHA-256:
`CC83324F9AB37DAEEF4E2BA158C821F336383A8C4F96ADFFF4DE7B79E276D0D7`

## Response

Researcher A accepts all Reviewer B conditions without modification.

## Accepted Conditions

1. DySL-VLA remains policy 2 in the first serious comparison.

The first comparison remains exactly:

1. `smolvla_base`
2. `dysl_action_importance_proxy`
3. `cspr_full`
4. `cspr_uniform_refinement_ablation`
5. `critical_step_threshold_simple_killer`

2. Novelty is narrowed.

CSPR claims only critical-step selective action refinement for frozen SmolVLA
action chunks. It does not claim general action-importance discovery, broad
dynamic inference, DySL reproduction, or generic LoRA adaptation.

3. DySL proxy transparency is mandatory.

If official DySL code cannot be faithfully adapted to the local backbone and
budget, policy 2 will be reported as a transparent proxy. It may implement
action-importance-conditioned capacity allocation, but it may not use CSPR's
learned residual action correction.

4. Local cache identities are frozen before Stage 0.

The current legal development cache is fixed to the verified `640` SmolVLA
Base rows over `libero_10/task_5`, `libero_goal/task_5`,
`libero_object/task_3`, and `libero_spatial/task_3`, demo ids `0..9`, unless a
later preregistered cache-coverage audit authorizes additional rows. DCCG
identities and cache choices remain closed.

5. Criticality label health is a hard pretraining gate.

Stage 0 must report positive/negative counts, per-task and per-phase coverage,
variance, duplicate keys, split overlap, and all-zero/all-one checks. Label
collapse stops as `DATA_FAILURE`.

6. Deployment observability is required.

The legal criticality predictor must beat trivial baselines using only current
observation features, proprioception, language/task features, and Base chunk
summaries. Baselines include gripper-transition threshold, Base
curvature/velocity threshold, task-mean criticality, and a frame-index/phase
audit proxy that is never allowed as an inference input.

7. Identity preservation is mandatory.

CSPR must show exact Base passthrough at initialization, checkpoint reload,
unchanged Base weights, bounded action deltas, postprocessed action validity,
and clean validation retention before rollout.

8. Objective scale and gradients must be audited.

The mathematical audit will define tensors, shapes, formulas, units, gradient
paths, term magnitudes, and gradient norm ratios for criticality, residual fit,
clean retention, and action-validity terms. Deterministic 7D-action KL is not
allowed.

9. The simple killer remains live.

`critical_step_threshold_simple_killer` remains policy 5. If it explains the
gain, CSPR is not a paper candidate.

10. No hidden confirmatory-test access is allowed.

No confirmatory identities, rewards, success flags, done flags, simulator
state, object pose, future observations, or demonstration time index may be
used for method selection or inference. Confirmatory outcomes may not retune
CSPR.

## Current Status

No CSPR implementation, training, validation search, rollout, simulator
access, or confirmatory-test access has happened. CSPR may proceed to
mathematical mechanism audit.
