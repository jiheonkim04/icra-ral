# CSPR-VLA Researcher A Proposal

Date: 2026-07-16 KST

Decision: `CSPR_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `CSPR-VLA`

Full name: Critical-Step Selective Policy Refinement for SmolVLA

Closest positive prior: DySL-VLA

Primary sources:

- https://arxiv.org/abs/2602.22896
- https://github.com/PKU-SEC-Lab/DYSL_VLA

## Fixed Claim

CSPR-VLA tests whether action-importance-conditioned intervention can improve
SmolVLA manipulation by refining only critical action cells while preserving
exact Base behavior elsewhere.

The scientific mechanism is not LoRA. LoRA may only parameterize a low-compute
residual/gate implementation. The claim is critical-step selective action
refinement at the action interface.

## External Prior Anchor

DySL-VLA demonstrates that robot action steps have unequal importance and that
an action-importance signal can allocate VLA computation selectively. Its
positive result is dynamic-static layer skipping with official code, a reported
`2.1%` CALVIN success-length gain, `85.7x` fewer trainable parameters, and
`3.75x` speedup at iso-accuracy.

CSPR extends the same importance-conditioned capacity-allocation axis from
selective internal computation to selective action correction:

- DySL-VLA: important actions receive more model computation.
- CSPR-VLA: important action cells receive bounded residual refinement.

This is a prior extension, not a reproduction claim that DySL's efficiency
result alone improves local LIBERO success.

## Local Data Boundary

CSPR uses only development identities for design and validation. The current
verified local cache has `640` cached SmolVLA Base rows on:

- `libero_10/task_5`
- `libero_goal/task_5`
- `libero_object/task_3`
- `libero_spatial/task_3`

with demo ids `0..9`.

DCCG remains closed as `DCCG_STAGE_0_DATA_FAILURE`. CSPR does not change DCCG
tasks, reset identities, cache source, thresholds, or interpretation.

## Mechanism

Let the Base action chunk be:

- `B in R^[N, 50, 7]`

where the final dimension is the normalized 7D action vector. Let deployment
features be:

- current RGB/visual feature `v in R^[N, 960]` when cached or locally
  extracted from legal observations;
- proprioception `p in R^[N, 8]`;
- task/language embedding or task identity feature `l`;
- Base chunk summary features from `B`, including velocity, acceleration,
  curvature, gripper-transition indicators, and per-group magnitudes.

CSPR learns:

- criticality predictor `c_phi(v, p, l, B) -> [0, 1]^[N, 50, 7]`;
- residual proposal `r_theta(v, p, l, B) -> R^[N, 50, 7]`;
- bounded gate `g = 1[c_phi >= tau]` or a validation-frozen soft gate.

The emitted action chunk is:

`A = postprocess(B + g * cap_group(tanh(r_theta), delta_max))`.

At initialization, `r_theta = 0` and the gate defaults to Base passthrough, so
`A = B` exactly.

## Supervision

Criticality labels are generated only from discovery/validation data. Candidate
label sources are:

- Base-vs-demonstration action error by group;
- demonstration curvature and acceleration;
- local action-change energy;
- gripper open/close transition boundaries;
- task/phase-local robust normalization.

The labels are diagnostics and training supervision only. At inference, CSPR
may not use demonstration frame index, future action, object pose, simulator
state, reward, success, done, or any confirmatory-test identity.

## Objective

For development training, use:

- criticality loss: binary cross-entropy or focal loss on noncollapsed
  critical action-cell labels;
- residual fit loss: Huber loss between `A` and demonstration action on
  high-criticality cells only;
- clean-retention loss: Huber or L2 action distance to Base on low-criticality
  cells;
- action-validity penalty: soft bound penalty before postprocessing.

No KL divergence is used between deterministic 7D actions.

Before any training beyond a tiny smoke, CSPR must report term magnitudes and
gradient norms on a small batch. No term may dominate another by scale without
normalization or a documented validation-only coefficient choice.

## Identity-Preserving Integration

CSPR is Base-preserving by construction:

- residual branch initialized to zero;
- gate initialized to Base passthrough;
- exact Base is included as a required diagnostic;
- groupwise residual caps for translation, rotation, and gripper;
- postprocessed action validity must be preserved;
- clean validation behavior must be retained before rollout.

A configuration that changes nearly all action cells or fails action validity
is an implementation or design failure, not a closed-loop scientific result.

## Bounded Validation Search

Maximum search budget: `6` configurations.

Allowed factors:

- residual cap: at most `3` values;
- criticality threshold or soft-gate temperature: at most `2` values.

No seed cherry-picking. No confirmatory-test tuning. Selection uses a frozen
validation score combining:

- validation closed-loop success if affordable, otherwise the closest legal
  proxy;
- clean retention;
- criticality activation localization;
- action validity;
- Base and prior separation;
- compute overhead.

Offline action L2 alone cannot select the final configuration.

## First Serious Comparison

The first serious comparison must use exactly five policies:

1. `smolvla_base`
2. `dysl_action_importance_proxy`
3. `cspr_full`
4. `cspr_uniform_refinement_ablation`
5. `critical_step_threshold_simple_killer`

`dysl_action_importance_proxy` must be transparent. If official DySL code can
be faithfully adapted to the local backbone and budget, use it. If not, use a
documented proxy that implements the prior's action-importance-conditioned
capacity allocation without CSPR's residual action correction.

The simple killer is a nonlearned threshold over Base chunk curvature,
velocity, and gripper transitions. It must remain in the first serious
comparison.

## Development Gates Before Rollout

CSPR may not proceed to large rollout if any of the following hold:

- cache coverage does not match the preregistered identities;
- criticality labels collapse to all zero or all one;
- criticality is not predictable above trivial baselines;
- the residual is nonacting;
- the residual globally changes nearly all action cells;
- Base action validity or clean retention fails;
- the DySL proxy or simple killer clearly explains the method;
- hidden confirmatory identities or privileged inference inputs are used.

Stop classes must be recorded honestly as `DATA_FAILURE`, `NO_HEADROOM`,
`IMPLEMENTATION_FAILURE`, or `DESIGN_FAILURE` when appropriate.

## Current Status

No CSPR implementation, training, validation search, rollout, simulator
access, or confirmatory-test access has happened. The next step is Reviewer B
attack on this frozen proposal.
