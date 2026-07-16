# AFID-VLA Researcher A Rebuttal

Date: 2026-07-16 KST

Decision: `AFID_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Method: `AFID-VLA`, Action-Factor Instruction Densification for
Base-preserving SmolVLA.

Proposal: `reports/afid_vla/researcher_proposal.md`

Proposal SHA-256:
`B5D1EE12FF2D0280511452DA7FE55295740FD9942A8BE293F444C8EB157062BC`

Reviewer B attack: `reports/afid_vla/reviewer_attack.md`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

## Summary

Researcher A accepts all Reviewer B conditions. AFID proceeds only as a narrow
FineVLA-prior extension: a frozen-SmolVLA, Base-preserving residual gate driven
by deployment-observable predictions of compact action-factor labels derived
from development-only demonstrations.

No AFID implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened.

## Accepted Prior Boundary

FineVLA remains the closest prior and policy 2 in the first serious
comparison. AFID does not claim that fine-grained language supervision is new.
That result belongs to FineVLA.

AFID's novelty is narrowed to:

`development-derived action-factor labels -> deployment-observable factor
prediction -> confidence-gated Base-preserving residual edits -> exact Base
passthrough when confidence is low or the factor mask is inactive`.

AFID is not:

- FineVLA renamed;
- standard LoRA with labels;
- generic action imitation;
- a manually tuned action mask;
- or a method that reads future action factors at inference.

## Accepted Policy Order

The first serious comparison remains exactly:

1. `smolvla_base`
2. `finevla_action_factor_proxy`
3. `afid_full`
4. `afid_no_factor_ablation`
5. `standard_lora`

The FineVLA proxy must be fair, transparent, nonprivileged, and matched. If no
official compatible FineVLA assets are available, the local proxy will use the
same SmolVLA Base, development splits, derived factor labels, optimizer budget,
action postprocessor, and inference budget, but without AFID's residual-gate
mechanism.

## Accepted Factor-Label Constraints

Before Stage 0, AFID must freeze exact factor extraction rules, including:

- tensor inputs;
- thresholds and deadbands;
- phase definitions;
- tie breaks;
- class mappings;
- mask construction;
- duplicate-key checks;
- task/phase coverage reports.

Stage 0 must stop as a data or supervision failure if factor labels or
factor-conditioned masks are all-zero, all-one, dominated by a single
task/phase, or otherwise collapsed.

## Accepted Observability Constraint

Action factors may be derived from development demonstration futures only as
training labels. At inference, AFID may use only deployment-observable
RGB/proprio/language, SmolVLA features, and frozen Base chunks.

Stage 0 must verify that factor prediction beats trivial majority and
task/phase baselines on validation before any factor is allowed to drive the
gate. Unobservable factors must be removed or the method must stop before
bounded validation search.

## Accepted Proxy And Ablation Conditions

AFID cannot advance if:

- `finevla_action_factor_proxy` dominates or makes AFID redundant;
- `afid_no_factor_ablation` explains the effect;
- matched `standard_lora` explains the effect;
- gate activation is independent of predicted factors.

The no-factor ablation must keep capacity, labels, optimizer budget, residual
caps, and clean retention matched while removing predicted factors from the
gate.

## Accepted Identity And Clean-Retention Conditions

AFID must preserve Base exactly at initialization and after disk reload.
Low-confidence rows and inactive-mask rows must return exact Base. Stage 0
must report activation frequency by task, phase, factor, time index, and
action dimension, plus p95 action deltas and action-bound validity.

AFID must stop before bounded validation if it globally changes actions,
violates action bounds, fails clean retention, or loses Base identity.

## Accepted Mathematical Conditions

The mathematical audit must freeze:

- all variables and tensor shapes;
- factor-label extraction rules;
- residual caps;
- confidence and entropy passthrough rules;
- predictor/gate/residual initialization;
- objective terms, scales, coefficients, and gradient paths;
- frozen-Base no-gradient checks;
- FineVLA proxy construction;
- no-factor ablation construction;
- standard-LoRA matching rules;
- action postprocessor and validity contract.

No deterministic-action KL is allowed. If any KL term is later proposed, it
must be rejected unless valid probability distributions, supports, direction,
estimator, gradient flow, and alternatives are justified.

## Accepted Stop Conditions

Researcher A accepts all Stage 0 stop conditions from Reviewer B:

- collapsed factor labels or masks;
- factor prediction not above trivial baselines;
- insufficient task/phase/factor coverage;
- no factor-conditioned residual headroom;
- FineVLA proxy dominance or redundancy;
- no-factor ablation explanation;
- standard-LoRA explanation;
- gate activation everywhere or nowhere;
- global action changes rather than bounded factor-conditioned edits;
- clean retention failure;
- action-bound violations;
- identity or checkpoint reload failure;
- reward/success/done/object pose/future observation use;
- any confirmatory-test task, reset identity, label, or outcome read.

## Current Status

The rebuttal passes AFID to the mathematical mechanism audit. No AFID
implementation, training, validation search, rollout, simulator access, or
confirmatory-test tuning has happened.

Immediate next stage: freeze AFID mathematical mechanism audit before
preregistration, implementation, validation search, or rollout.
