# CALA-VLA Researcher A Rebuttal

Date: 2026-07-15 KST

Frozen proposal reviewed: `reports/cala_vla/researcher_proposal.md`

Proposal hash: `5B3933C9C0FD5AE5F07FDB0CEC447B48040238FB6D872D97E545E3D93E257E76`

Reviewer attack: `reports/cala_vla/reviewer_attack.md`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Rebuttal decision: `CALA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

## Response Summary

Researcher A accepts the Reviewer B constraints. CALA-VLA will not claim broad
novelty for latent actions, action-tokenization, latent planners,
coarse-to-fine action guidance, or context-gated action conditioning. The only
claim preserved for the local method is:

frozen SmolVLA may benefit from a legally deployable, source-gated,
identity-preserving CAC-style latent-action adapter if Stage 0 proves the
latent labels are observable from deployment inputs, noncollapsed, split-clean,
useful, and not explained by a task-mean, phase-only, or action-history
shortcut.

No implementation, training, validation search, rollout, or confirmatory-test
tuning is authorized by this rebuttal. The next permitted step is mathematical
mechanism audit.

## Accepted Novelty Narrowing

Accepted. CAC-VLA already contains the main latent-action prediction and
context-gated action conditioning mechanism. RotVLA, LARA, and related
latent-action work further narrow the novelty landscape.

CALA's local contribution is restricted to:

- frozen SmolVLA integration;
- source-gated proof that no future action or privileged latent is used at
  inference;
- identity-preserving context-gate initialization;
- bounded hidden-state residual on the local SmolVLA action interface;
- matched comparison to `cac_vla_latent_action_proxy`,
  `cala_no_context_gate_ablation`, and `task_mean_latent_action_baseline`.

If the closest-prior proxy matches or beats CALA on the frozen claim axis, the
local CALA contribution is killed or archived. The proxy will remain labeled as
a faithful transparent local proxy unless official code/checkpoint/protocol
equivalence is separately verified before confirmatory testing.

## Accepted Future-Action Source Gate

Accepted. Stage 0 must prove source legality before training, validation
search, manifest freeze, or rollout.

Inference may use only:

- official deployment RGB observations;
- language instruction;
- deployment proprioception;
- frozen Base features, hidden states, action previews, or observations
  available through the local deployment path.

Inference may not use:

- future HDF5 actions;
- latent labels derived from future actions;
- future observations;
- simulator object pose or state;
- reward or success;
- reset identity;
- confirmatory-test task/reset metadata;
- hidden episode progress or demonstration timestep labels unavailable at
  deployment.

Future action segments may be used only on discovery/validation records to
construct training labels, inspect headroom, and audit source predictability.
Any source-gate failure is a pre-rollout `DATA_OR_SUPERVISION_FAILURE`,
`DESIGN_FAILURE`, or `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, not a
closed-loop scientific result.

## Accepted Trivial-Baseline Constraints

Accepted. The first serious comparison remains exactly:

1. `frozen_smolvla`
2. `cac_vla_latent_action_proxy`
3. `cala_full`
4. `cala_no_context_gate_ablation`
5. `task_mean_latent_action_baseline`

`task_mean_latent_action_baseline` remains the one mandatory simple
reviewer-killer policy through Stage A/B. Stage 0 diagnostics must also compare
latent predictability against task-mean, phase-only, action-only/action-history,
and majority/trivial predictors, but those diagnostics do not add extra
mandatory policy baselines before the five-policy comparison.

If the task-mean baseline accounts for CALA's gain, CALA is not a paper
candidate.

## Accepted Local CAC Proxy Requirements

Accepted. `cac_vla_latent_action_proxy` must be transparent and fair:

- same data partitions;
- same latent encoder where possible;
- same inference input restrictions;
- comparable inference budget;
- no official-reproduction claim unless official equivalence is documented.

The proxy/full technical difference must be documented before confirmatory
testing. A weak or unfair proxy invalidates the comparison.

## Accepted SmolVLA Interface Constraints

Accepted. CALA is not allowed to become a final-action residual wrapper or
globally destructive hidden-state perturbation.

Before rollout, CALA must show:

- exact or near-exact initial Base passthrough;
- exact SmolVLA hook and tensor shape;
- checkpoint disk reload;
- finite nonzero gradients in intended latent predictor/adaptation parameters;
- bounded translation, rotation, and gripper deltas;
- clean validation retention;
- full differs from the no-context-gate ablation;
- mechanism activation in relevant states rather than everywhere;
- no privileged inference input.

If the only available integration is a destructive final 7D action residual,
the method stops as `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.

## Accepted Latent Encoder Health Requirements

Accepted. The OAT-lite latent encoder must be mechanistically useful rather
than decorative.

Stage 0 must report:

- latent horizon and dimensionality;
- per-dimension variance and explained-variance concentration;
- high/low contrast counts or noncollapsed cluster occupancy;
- task and phase coverage;
- duplicate sample/frame counts;
- train/validation/test identity overlap;
- predictability from deployment inputs above trivial baselines;
- whether latent labels differ materially from task-mean or action-history
  prototypes.

Collapsed, split-leaking, or trivially predictable labels stop before rollout.

## Accepted Matched Ablation

Accepted. `cala_no_context_gate_ablation` must use the same latent labels,
training records, Base policy, data partitions, and comparable parameter budget
where possible. It may remove or disable the context-dependent gate, but it
must not be an intentionally weak strawman.

Before rollout report:

- full-versus-ablation latent predictions;
- gate values by task and phase;
- full-versus-ablation action L2 by translation, rotation, and gripper;
- contexts where full activates and ablation does not;
- whether full and ablation differ on validation.

If full and ablation are action-equivalent on validation, stop as exact trivial
equivalence.

## Accepted Mathematical Objective Restrictions

Accepted. The mathematical audit must not use KL between deterministic 7D
actions. It must document variables, tensor shapes, objective scales, units,
gradient paths, small-batch loss magnitudes, gradient norms, and required
ablations before implementation.

The audit must specify whether the latent encoder, latent predictor, and
adapter are:

- trained jointly;
- frozen after validation-only selection;
- generated offline from development labels;
- or nontrainable in the closest-prior proxy.

## Frozen Next Step

Proceed to mathematical mechanism audit only.

The audit must be completed and committed before any implementation, Stage 0
data construction, validation search, training, manifest freeze, or rollout.
