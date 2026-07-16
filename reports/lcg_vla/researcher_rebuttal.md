# LCG-VLA Researcher A Rebuttal

Date: 2026-07-16 KST

Decision: `LCG_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Proposal: `reports/lcg_vla/researcher_proposal.md`

Proposal SHA-256:
`F0D980AA0760F143D781C723DB632BC324C1E18F390D9C33C5DA94F3A897D11E`

Reviewer attack: `reports/lcg_vla/reviewer_attack.md`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Researcher A accepts all Reviewer B conditions. LCG remains live only under the
narrowed formulation below.

## Accepted Novelty Boundary

Researcher A accepts that LCG novelty is limited to:

`A frozen-SmolVLA, Base-preserving, identity-initialized action-cell gate that
learns when deployment-observable original-versus-null language contrast
permits bounded residual edits, with exact Base passthrough when the contrast
is absent or unreliable.`

LCG does not claim to invent CAG, language-null action comparison,
counterfactual labels, or LoRA-based instruction following. Counterfactual
Action Guidance is the closest positive prior and remains policy 2 in the
first serious comparison.

## Accepted Prior And Proxy Conditions

Researcher A accepts that the local `counterfactual_action_guidance_proxy` is
only a transparent proxy unless official CAG assets are installed and verified.
The proxy must use:

- the same frozen SmolVLA Base;
- the same observation, proprioception, original instruction, and
  language-null branch;
- the same action postprocessor;
- validation-only coefficient selection;
- no confirmatory-test identities or outcomes.

If official CAG becomes locally available before confirmatory testing, the
campaign must compare against it or record a protocol-incompatibility reason
without using confirmatory outcomes to retune LCG.

## Null-Branch Definition

The null branch is not claimed to be CAG's official VA module. It is a local
development proxy that must be validated before use.

The mathematical audit must freeze `l_null` exactly. Stage 0 must verify that
`N_t = pi_base(o_t, q_t, l_null)` is finite, postprocessor-valid, noncollapsed,
and not globally destructive. If null-branch behavior is invalid or collapsed,
LCG must stop as data/proxy failure before validation search.

## Contrast Is A Gate Signal, Not A Residual Target

Researcher A accepts that `B_t - N_t` may indicate language-sensitive action
cells, but it is not treated as the correct residual direction.

The demonstration residual remains:

`R_t = E_t - B_t`.

The contrast `C_t = group_norm(B_t - N_t)` may gate or condition the residual
head. It cannot by itself count as the target residual, and a training-free CAG
coefficient must remain policy 2.

Stage 0 must test whether contrast magnitude predicts useful Base-to-expert
residual correction above task/phase baselines. If contrast is not predictive,
LCG stops before validation search.

## Accepted Policy Order

The first serious comparison remains exactly:

1. `smolvla_base`
2. `counterfactual_action_guidance_proxy`
3. `lcg_full`
4. `lcg_no_language_contrast_ablation`
5. `standard_lora`

The no-language-contrast ablation must keep the same trainable capacity,
optimizer budget, labels, clean-retention terms, and action caps while removing
`N_t` and `C_t` from the gate/residual input.

Standard LoRA remains required because LCG trains lightweight infrastructure on
demonstrations.

## Counterfactual Text And Leakage Rules

Any counterfactual instruction alternatives must come only from frozen
discovery/validation task text. They may not use confirmatory-test task text,
reset identities, labels, failed rollouts, or outcomes.

The mathematical audit or preregistration must freeze:

- discovery task identities;
- validation task identities;
- confirmatory-test task and reset identity exclusion rules;
- instruction alternative source;
- duplicate and overlap checks.

If the language alternatives cannot be generated without confirmatory leakage,
LCG must use only the language-null branch or stop before implementation.

## Stage 0 Noncollapse And Headroom Gates

Before bounded validation or rollout, Stage 0 must prove:

- Base/null contrast is finite and noncollapsed across tasks, phases, timesteps,
  and action groups;
- the language mask is neither all-zero nor all-one;
- Base-to-demonstration residual labels are noncollapsed;
- contrast magnitude predicts useful residual headroom above trivial task/phase
  baselines;
- the CAG proxy leaves residual headroom for LCG;
- LCG differs from Base, CAG proxy, no-language-contrast ablation, and standard
  LoRA after a small development fit;
- inactive gates preserve Base exactly;
- active gates are bounded and postprocessor-valid.

All-zero contrast, all-one contrast, collapsed residual labels, no CAG residual
headroom, or global action changes must stop as `DATA_OR_SUPERVISION_FAILURE`,
`NO_HEADROOM`, or `DESIGN_FAILURE`, not as a closed-loop scientific result.

## Clean Retention And Identity Preservation

LCG must initialize to exact Base passthrough and reproduce that identity after
disk reload. Clean-retention rows where language contrast is absent or
unreliable must preserve Base behavior.

Stage 0 must report:

- Base action;
- null-branch action;
- Ours action;
- residual norm;
- gate values;
- action dimensions changed;
- activation context;
- translation, rotation, and gripper deltas;
- action-validity status.

## Accepted Mathematical Audit Commitments

The mathematical audit must freeze:

- `l_null` text and tokenizer handling;
- `B_t`, `N_t`, `C_t`, `G_theta`, `Delta_theta`, and `A_t` with shapes;
- horizon, action dimension, context horizon, and action-group caps;
- initialization and disk-reload identity tolerance;
- language mask construction and noncollapse thresholds;
- clean-retention objective;
- CAG proxy formula and validation-only coefficient selection;
- loss term magnitudes and gradient norms before training;
- frozen-Base gradient checks;
- no deterministic-action KL.

If any KL term is proposed later, it must be rejected unless its arguments are
valid distributions with direction, support, estimator, gradient flow, and
alternatives justified.

## Accepted Stop Conditions

LCG must stop before validation search for:

- collapsed Base/null contrast;
- invalid or distribution-shifted null branch actions;
- collapsed language mask;
- no residual headroom beyond the CAG proxy;
- equivalence to a tuned CAG coefficient;
- equivalence to the no-language-contrast ablation;
- standard LoRA explaining the effect;
- global action changes rather than bounded cell edits;
- clean-retention failure;
- action-bound violations;
- identity or checkpoint reload failure;
- reward, success, done, object pose, future observation, or confirmatory
  record reads;
- attempted rescue of S2C or any previous closed method.

## Rebuttal Decision

All Reviewer B conditions are accepted. LCG is not killed before
implementation, but it may proceed only to mathematical mechanism audit. No
preregistration, prototype protocol, validation search, training, rollout, or
confirmatory evaluation is allowed until the mathematical audit freezes the
objective, CAG proxy boundary, null-branch handling, gradient-scale checks,
identity-preserving integration, ablation requirements, and Stage 0 stop
classes.
