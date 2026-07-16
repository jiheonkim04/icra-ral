# CSPR-VLA Reviewer B Attack

Date: 2026-07-16 KST

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Method: `CSPR-VLA`

Reviewed proposal: `reports/cspr_vla/researcher_proposal.md`

Proposal SHA-256:
`CC83324F9AB37DAEEF4E2BA158C821F336383A8C4F96ADFFF4DE7B79E276D0D7`

## Summary

CSPR is not rejected before development. It has a plausible new mechanism:
action-importance-conditioned bounded residual refinement with exact Base
passthrough outside predicted critical action cells.

However, the proposal needs a strict rebuttal before mathematical audit. The
main risk is novelty and comparison inflation: DySL-VLA is primarily an
efficiency prior, not a direct action-improvement prior. CSPR can remain a
valid prior extension only if the claim is narrowed to importance-conditioned
capacity allocation at the action interface and the DySL proxy is transparent.

## Required Conditions

1. Keep DySL-VLA as policy 2 in the first serious comparison.

The first comparison must remain exactly:

1. `smolvla_base`
2. `dysl_action_importance_proxy`
3. `cspr_full`
4. `cspr_uniform_refinement_ablation`
5. `critical_step_threshold_simple_killer`

Do not replace policy 2 with a weaker convenience baseline. Do not add a
standard-LoRA sixth policy before the first serious comparison unless the
method itself becomes a LoRA scientific claim, which is currently forbidden.

2. Narrow the novelty claim.

CSPR may claim only:

`critical-step selective action refinement for frozen SmolVLA action chunks`.

It may not claim broad action-importance discovery, general dynamic inference,
DySL reproduction, or generic LoRA adaptation.

3. Make the DySL proxy transparent.

If official DySL code cannot be faithfully run on the local backbone and
budget, the proxy must be declared as a transparent proxy. It must implement
action-importance-conditioned capacity allocation without CSPR's learned
residual action correction. Its provenance, inputs, and mismatch to official
DySL must be reported.

4. Freeze local cache identities before Stage 0.

The current legal development cache is:

- `640` SmolVLA Base rows;
- tasks `libero_10/task_5`, `libero_goal/task_5`,
  `libero_object/task_3`, `libero_spatial/task_3`;
- demo ids `0..9`.

Any new cache coverage must be separately audited before use. DCCG identities
and cache choices may not be altered or reinterpreted.

5. Prove criticality labels are healthy before training.

Stage 0 must report positive/negative counts, per-task and per-phase coverage,
variance, duplicate keys, split overlap, and all-zero/all-one checks. Label
collapse is a `DATA_FAILURE`, not a scientific kill.

6. Prove criticality is deployment-observable.

A legal predictor using current observation features, proprioception, language
or task feature, and Base chunk summaries must beat trivial baselines. The
trivial baselines must include at least:

- gripper-transition threshold;
- Base action curvature/velocity threshold;
- task-mean criticality;
- frame-index or phase proxy used only as an audit baseline, never as an
  inference input.

If the legal predictor does not beat these baselines, stop as `DESIGN_FAILURE`.

7. Preserve identity before any rollout.

CSPR must show exact Base passthrough at initialization, checkpoint reload,
unchanged Base weights, bounded action deltas, postprocessed action validity,
and clean validation retention. A module that changes most action cells is an
`IMPLEMENTATION_FAILURE` or `DESIGN_FAILURE`.

8. Audit objective scale and gradients.

The mathematical audit must define all tensors, shapes, formulas, units,
gradient paths, term magnitudes, and gradient norm ratios for criticality,
residual fit, clean retention, and action-validity terms. No deterministic
7D-action KL is allowed.

9. Keep the simple killer live.

`critical_step_threshold_simple_killer` must remain policy 5. If it explains
the apparent gain, CSPR is not a paper candidate.

10. No hidden confirmatory-test access.

No confirmatory identities, rewards, success flags, done flags, simulator
state, object pose, future observations, or demonstration time index may be
used for method selection or inference. Confirmatory outcomes may not retune
CSPR.

## Reviewer Decision

Conditional pass to Researcher A rebuttal. Researcher A must explicitly accept
all ten conditions or revise the proposal boundary before mathematical audit,
preregistration, implementation, validation search, rollout, or confirmatory
test access.
