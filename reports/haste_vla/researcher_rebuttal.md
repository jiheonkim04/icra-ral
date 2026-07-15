# HASTE-VLA Researcher A Rebuttal

Date: 2026-07-15 KST

Proposal hash:
`5415BC1533A24EC55CC511DDEB014BB11D9C19F603C59D1F1D3E151E15B930A6`.

Decision: `HASTE_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

## Claim Narrowing

HASTE claims neither event-aware training nor keyframe supervision broadly.
Those belong to StaKe, EventVLA, KEMO, and FrameSkip. The provisional claim is
only that a censored command-event hazard plus current-centered cumulative
action displacement can improve the same event-transition claim axis beyond a
matched StaKe proxy.

The labels mark gripper command changes. They do not assert physical contact,
successful grasp, or release. The six-dimensional target is a sum in frozen
postprocessed action coordinates, not an exact SE(3) composition.

## Censoring

For every row, observable future length is
`m_t = min(H_e, T-1-t)`. Survival likelihood is evaluated only for intervals
`1..m_t`. A transition in that range is uncensored. No transition before the
demonstration boundary is boundary-censored and is never converted into a
known no-event target beyond `m_t`.

Rows with `m_t < 1` are excluded by the manifest. The manifest persists
`m_t`, event offset or null, censor reason, source hash, and split.

## Frozen Trivial Baselines

Hazard trivial baseline: one global per-offset hazard vector fitted from
discovery event/survival counts with Laplace smoothing `1.0`. It receives no
task identity or validation statistics.

Displacement trivial baseline: one global six-dimensional discovery mean in
normalized coordinates. It receives no task identity or validation statistics.

The frozen-feature probes use discovery-only optimization and validation-only
evaluation. Probe hyperparameters are fixed in the prototype protocol and are
not a model search.

## Objective Scale And Conflict

Every hazard loss is divided by its valid interval count. Displacement Huber is
averaged over six coordinates and uncensored rows only. Stage 0B reports raw
term means before weighting, LoRA gradient norms, head gradient norms, and all
pairwise LoRA-gradient cosines.

Any nonfinite gradient, absent required gradient, or unexplained maximum/minimum
nonzero LoRA-gradient norm ratio above `100` is an implementation failure.
Negative cosine is reported, not automatically repaired. The six frozen
validation configurations are the only coefficient/horizon choices.

## Identity

Auxiliary heads read an audited existing SmolVLA representation. They do not
append active inference tokens or modify normalization. LoRA B matrices are
zero initialized. Stage 0A compares frozen Base and initialized HASTE flow
vectors and decoded actions on identical inputs and noisy action states before
any optimizer step and after disk reload.

The Base checkpoint hash must remain unchanged. Identity failure blocks all
training and rollout.

## Honest Stops

The proposal accepts all Reviewer B stops. Collapsed labels are `DATA_FAILURE`;
no event-near Base deficit is `NO_HEADROOM`; targets not predictable from legal
deployment representations are `DESIGN_FAILURE`; source, identity, gradient,
finite, or persistence defects are `IMPLEMENTATION_FAILURE`.

None is a scientific kill. No threshold, task, source, horizon, architecture,
event definition, or target repair is allowed within HASTE after execution.
