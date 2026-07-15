# VDR-VLA Researcher A Rebuttal

Date: 2026-07-16 KST

Proposal hash:
`0229EBC15901F4FE1EDD3839AB6B984AFA3E0E99836B5C88CF21F2C7DE2B3E72`

Reviewer decision:
`REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Rebuttal decision:
`VDR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

## Accepted Narrow Claim

Researcher A accepts that VDR is not a full FutureVLA reproduction, not a
general future-prediction architecture, not a visual reconstruction method, and
not an action-realization method. The claim is narrowed to:

`dynamic residual future-feature alignment for SmolVLA, where the residual is
the future visual-feature change unexplained by a discovery-fitted actionless
static predictor`.

## Accepted Baselines

The first serious comparison remains:

1. `smolvla_base`
2. `futurevla_latent_alignment_proxy`
3. `vdr_full`
4. `vdr_no_action_residual`
5. `standard_lora`

Standard LoRA is included because VDR updates policy weights on the same
demonstrations. It is an explanation control, not a contribution.

## Accepted Hard Gates

VDR will stop before training or rollout if:

- dynamic residual targets collapse;
- split overlap or duplicate keys are nonzero;
- the residual is not predictable from deployment-observable inputs;
- generated-action-conditioned residual prediction fails to beat actionless
  residual prediction by the preregistered margin;
- the FutureVLA proxy leaves no residual headroom;
- adapter identity, disk reload, gradient, or action validity checks fail;
- any privileged inference input is required.

These stops are development outcomes, not scientific kills.

## Accepted Source And Leakage Constraints

Future frames and frozen future visual features are training targets only.
PCA, whitening, and actionless static predictors are fitted on discovery rows
only. Validation selects only the coefficient among `{0.1,0.3,1.0}` and never
refits target construction. Confirmatory identities remain unread until the
final configuration, baselines, metrics, thresholds, and manifest are frozen.

## Accepted Mathematical Constraints

The mathematical audit will use coordinate-mean Huber on whitened residual
vectors. It will not use KL, entropy, mutual information, or any decorative
divergence. It will report target scales, objective magnitudes, gradient norms,
gradient destinations, and full-versus-ablation residual consequences before
any expensive training.

## Outcome

VDR proceeds only to mathematical mechanism audit and preregistration. No
training, validation search, rollout, confirmatory-test access, or
hyperparameter expansion has happened.
