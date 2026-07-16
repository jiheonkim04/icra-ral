# URF-VLA Researcher A Rebuttal

Date: 2026-07-16 KST

Decision: `URF_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Proposal: `reports/urf_vla/researcher_proposal.md`

Proposal SHA-256:
`E78829E736C3F22451E72574092221904ACBE4C4BE0BDA7FA046832DABED3532`

Reviewer attack: `reports/urf_vla/reviewer_attack.md`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Researcher A accepts all Reviewer B conditions. URF remains live only under the
narrowed formulation below.

## Accepted Novelty Boundary

Researcher A accepts that URF novelty is narrowed to:

`Base-preserving uncertainty-routed bounded residual transport around an
already trained SmolVLA action chunk, initialized to exact Base passthrough and
trained from existing demonstrations without rollout-success labels.`

URF does not claim to invent heteroscedastic residual flow, residual variance
prediction, uncertainty-aware action generation, or lightweight adapter
fine-tuning. SUREFlow owns the closest positive prior for uncertainty-aware
residual flow matching. Guided Action Flow is the closest frozen-SmolVLA
action-intervention prior. URF may continue only as a conservative overlay that
keeps Base as the default action and routes bounded residual transport into
specific Base action cells.

If mathematical audit or Stage 0 cannot preserve this boundary, URF must stop
before validation search, rollout, or confirmatory testing.

## Accepted Prior And Proxy Conditions

Researcher A accepts that SUREFlow remains the closest prior and policy 2 unless
official SUREFlow assets are locally installed and verified before the first
serious comparison.

The `sureflow_uncertainty_residual_proxy` must not be a strawman. It must:

- use the same discovery/validation/test partition as URF;
- use only legal deployment-observable inference inputs;
- receive the same cached SmolVLA features, Base chunks, expert residuals, and
  comparable optimizer budget where technically valid;
- train a heteroscedastic residual-flow objective, not only plain residual MSE;
- expose a comparable uncertainty or variance output;
- use the same postprocessor and action-validity definition;
- be labeled as a transparent local proxy, not official SUREFlow, unless the
  official assets are actually installed and verified.

If official SUREFlow becomes locally available before confirmatory testing, the
campaign must compare against it or record a protocol-incompatibility reason
without using confirmatory outcomes to retune URF.

## Accepted Action-Intervention And Uncertainty Priors

Researcher A accepts that Guided Action Flow must remain listed as the closest
frozen-SmolVLA action-intervention prior, even if it is not a first-stage policy.
URF differs only by using demonstration-supervised heteroscedastic residual
routing rather than rollout-success critic gradients at inference.

Researcher A also accepts that Flow-Based VLA UQ / SAVE and perturbation-based
uncertainty are credible uncertainty-signal alternatives. URF cannot claim to be
a state-of-the-art epistemic uncertainty estimator. The uncertainty claim is
limited to an internal routing variable for bounded residual transport.

## Accepted Ablation And Simple Killer

The first serious comparison keeps:

1. `smolvla_base`;
2. `sureflow_uncertainty_residual_proxy` or official `sureflow` if installed;
3. `urf_full`;
4. `urf_no_uncertainty_route_ablation`;
5. `standard_lora`.

`urf_no_uncertainty_route_ablation` remains the key ablation. It must keep
residual capacity, residual cap, optimizer budget, clean-retention policy, and
action-validity semantics matched while removing uncertainty routing.

`standard_lora` remains the first simple reviewer-killer because URF trains
adapter/head infrastructure on the same demonstrations. If matched LoRA explains
the gain, URF is not paper-viable.

## Accepted Stage 0 Diagnostics

Stage 0 must include cheap diagnostics where feasible for:

- task/phase residual;
- residual-magnitude routing;
- stochastic-sampling disagreement;
- perturbation-disagreement routing;
- Base-to-expert residual headroom;
- SUREFlow-proxy residual headroom.

These diagnostics are development-only unless a diagnostic explains URF's
validation signal and must become a frozen comparison for a later paper package.
Offline action L2 alone cannot select a configuration or support a paper claim.

## Accepted Data And Routing Gates

Before bounded validation or any rollout, Stage 0 must show:

- residual targets are noncollapsed by task, phase, timestep, and action
  dimension;
- Base residuals are not mostly postprocessor noise;
- a heteroscedastic probe beats homoscedastic residual and task/phase residual
  baselines on validation;
- uncertainty strata are noncollapsed and monotonic with residual error on
  validation;
- URF full differs from both Base and no-uncertainty ablation after a small fit;
- route sparsity is neither all-zero nor all-one;
- gripper residuals do not dominate merely because of scale.

All-zero routing, all-one routing, collapsed uncertainty strata, missing
residual headroom, or failure to infer the mechanism from legal deployment
inputs must stop as `URF_STAGE_0_DATA_OR_SUPERVISION_FAILURE`,
`URF_STAGE_0_NO_USABLE_HEADROOM`, or `URF_STAGE_0_DESIGN_FAILURE`, not as a
closed-loop scientific result.

## Accepted Mathematical Audit Requirements

The mathematical audit must define:

- exact variables and tensor shapes for `[batch, horizon=50, action_dim=7]`;
- whether `ell_theta` parameterizes log variance, log scale, or log precision;
- how the Huber scale interacts with the variance term;
- whether uncertainty gradients can inflate variance to avoid residual learning;
- variance floor and ceiling semantics;
- how uncertainty maps to the bounded route gate;
- units and separate scaling for translation, rotation, and gripper;
- loss magnitudes and gradient norms before training;
- which gradients flow through Base, residual, variance, and route heads;
- why Huber or vector-field consistency is used instead of JS, Wasserstein,
  MMD, Mahalanobis distance, KL, or trajectory discrepancy.

No KL may be computed directly between deterministic 7D action vectors or
SmolVLA flow vectors.

## Accepted Inference And Leakage Rules

URF, SUREFlow proxy, standard LoRA, and ablations may use only legal current
images, proprioception or state already exposed to SmolVLA, task text or task
identity available to the policy, frozen Base decoded chunks, training
statistics, and learned checkpoints.

They may not use simulator reward, success, done, object pose, future
observation, expert future action at inference, confirmatory reset identity,
test-set residual statistics, or failed rollout labels collected from
confirmatory evaluation.

No confirmatory-test tuning, task/reset changes, proxy change after results,
threshold rescue, or retrospective reinterpretation is allowed. CCIF, TSC, CFR,
AMP, RAP, and VDR remain closed and may not be rescued through URF.

## Rebuttal Decision

All Reviewer B conditions are accepted. URF is not killed before implementation,
but it may proceed only to mathematical mechanism audit. No preregistration,
validation search, training, rollout, or confirmatory evaluation is allowed
until the mathematical audit freezes the objective, proxy definition,
uncertainty-routing semantics, gradient-scale checks, identity-preserving
initialization, action-validity semantics, ablation requirements, and Stage 0
stop classes.
