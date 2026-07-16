# AFID-VLA Reviewer B Attack

Date: 2026-07-16 KST

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Method: `AFID-VLA`, Action-Factor Instruction Densification for
Base-preserving SmolVLA.

Proposal: `reports/afid_vla/researcher_proposal.md`

Proposal SHA-256:
`B5D1EE12FF2D0280511452DA7FE55295740FD9942A8BE293F444C8EB157062BC`

## Summary Judgment

Conditional pass to rebuttal.

AFID is better anchored than LCG because FineVLA already demonstrates a
positive action-control effect from fine-grained process supervision, and the
local LIBERO demonstrations plausibly contain action/proprio traces from which
compact action factors can be derived. AFID also respects the current design
constraint: LoRA is only infrastructure, and the closest prior enters the
first serious comparison.

The method may continue only if Researcher A accepts the conditions below.
The novelty is real only if AFID remains a sparse action-factor predictor plus
Base-preserving residual gate. It must not become a weak FineVLA proxy, a
hand-tuned action mask, ordinary LoRA imitation, or a future-action label leak.

## Closest Prior Boundary

Closest prior: FineVLA.

Primary source: `https://arxiv.org/html/2605.27284v1`

FineVLA is the closest prior because it already shows that fine-grained
process supervision improves steerable VLA control and can retain goal-level
task completion. AFID's allowed novelty is not the broad idea that
fine-grained language helps. That is FineVLA's prior result.

Permitted novelty boundary:

`A frozen-SmolVLA, Base-preserving residual gate driven by deployment-observable
predictions of compact action-factor labels that are derived from
development-only demonstrations, with exact Base passthrough when factor
confidence is low or the factor-conditioned mask is inactive.`

AFID novelty is not:

- converting action deltas into words and training standard LoRA;
- adding a generic auxiliary classifier;
- manually selecting action cells by residual magnitude after seeing test
  failures;
- using demonstration actions, future observations, success labels, or object
  poses at inference;
- or claiming FineVLA's process-supervision result under a new acronym.

## Required Policy Order

The first serious comparison must keep:

1. `smolvla_base`
2. `finevla_action_factor_proxy`
3. `afid_full`
4. `afid_no_factor_ablation`
5. `standard_lora`

Policy 2 must be a fair closest-prior proxy. If official FineVLA assets are
not locally compatible, the proxy must use the same SmolVLA Base, development
splits, factor labels, action postprocessor, optimizer budget, and inference
budget, but without AFID's residual-gate mechanism.

`afid_no_factor_ablation` must remove predicted factors from the gate while
keeping trainable capacity, labels, optimizer budget, residual caps, and clean
retention matched.

`standard_lora` must use matched demonstrations, optimizer budget, action
postprocessing, and clean-retention handling where applicable.

## Major Risks

### Risk 1: Factor Labels Can Be Arbitrary Or Collapsed

The proposal lists plausible factors, but their extraction rules are not yet
frozen. If thresholds are chosen after inspecting outcomes, AFID becomes
post-hoc protocol engineering. If labels are all-zero, all-one, or dominated
by one task/phase, the mechanism is untestable.

Required rebuttal: freeze exact factor extraction rules before Stage 0,
including tensor inputs, thresholds, deadbands, phase definitions, tie breaks,
and class mappings. Stage 0 must report class counts, entropy, task/phase
coverage, duplicate keys, and mask frequencies.

### Risk 2: Factor Prediction May Not Be Observable At Inference

Action factors derived from demonstration futures may be predictable only
because training sees future action chunks. At deployment, AFID has current
RGB/proprio/language/Base chunks only. If factor prediction is not better than
a majority or task/phase baseline, the gate cannot be a causal mechanism.

Required Stage 0 gate: factor prediction must beat trivial baselines on
validation for the factors used by the gate. Factors that fail observability
must be removed before any bounded validation search.

### Risk 3: FineVLA Proxy Can Be Too Weak Or Too Strong

A proxy that merely appends crude labels as text may understate FineVLA. A
proxy that directly uses action labels at inference would be privileged. Either
case invalidates the prior-first comparison.

Required rebuttal: define the FineVLA action-factor proxy as a transparent,
nonprivileged, matched-budget proxy before implementation. Stage 0 must record
whether the proxy leaves meaningful residual headroom for AFID and whether
AFID is actually distinct from it.

### Risk 4: AFID May Collapse Into Standard Imitation Or LoRA

If the factor predictor is ignored and the residual head learns generic
offline action imitation, the mechanism is not AFID. Standard LoRA is the
obvious reviewer-killer baseline.

Required Stage 0 gate: AFID cannot advance if `standard_lora` explains the
offline effect, if the no-factor ablation matches AFID, or if gate activation
does not depend on predicted factors.

### Risk 5: Factor-Conditioned Residuals May Not Help Closed Loop

Offline Huber improvement does not guarantee closed-loop success. A factor
could reduce action L2 while damaging task progress or contact timing.

Required rebuttal: Stage 0 must remain development-only and cannot be called a
scientific kill or success. The validation score must include clean retention,
action validity, mechanism activation locality, and the closest feasible
closed-loop proxy, not pure action L2.

### Risk 6: The Gate Can Become Global

LCG failed partly because the gate activated nearly everywhere. AFID has the
same risk if factor confidence is broad or masks are too permissive.

Required diagnostics: activation frequency by task, phase, factor, time index,
and action dimension; Base/Ours/residual examples for active and inactive
states; p95 action deltas; action-bound validity; exact Base passthrough when
confidence is low.

### Risk 7: Clean Behavior Can Be Damaged

The claim requires preserving Base when factor confidence is absent or low.
If AFID changes clean rows, it is not identity-preserving.

Required Stage 0 gate: initialized AFID and disk-reloaded AFID must equal Base
within tolerance; low-confidence rows must remain exact Base; clean-retention
validation rows must satisfy the frozen delta and validity requirements.

## Mathematical Audit Requirements

The mathematical audit must freeze:

- exact factor-label variables, shapes, units, thresholds, and class mappings;
- `B_t`, `E_t`, `R_t`, `Z_t`, `M_factor`, `P_theta`, `c_theta`, `G_theta`,
  `Delta_theta`, and `A_t` with tensor shapes;
- residual caps for translation, rotation, and gripper groups;
- factor-confidence rule and entropy/low-confidence passthrough rule;
- predictor, gate, and residual initialization;
- all objective terms, scales, units, coefficients, and gradient paths;
- frozen-Base no-gradient checks;
- validation-only coefficient and threshold search budget;
- FineVLA proxy construction;
- no-factor ablation construction;
- standard LoRA matching rules;
- action postprocessor and validity contract;
- no deterministic-action KL.

If a KL term is proposed, reject it unless valid probability distributions,
supports, direction, estimator, gradient flow, and alternatives are justified.
Do not compute KL directly between deterministic 7D actions.

## Required Ablations And Diagnostics

1. `finevla_action_factor_proxy`
2. `afid_no_factor_ablation`
3. matched `standard_lora`
4. factor-predictability majority and task/phase baselines
5. factor-mask noncollapse and coverage diagnostics
6. confidence-threshold passthrough diagnostics
7. clean-retention and inactive-gate exact-Base diagnostics

## Stage 0 Must Stop For

- collapsed factor labels or masks;
- factor prediction not above trivial baselines;
- insufficient task/phase/factor coverage;
- no factor-conditioned residual headroom;
- FineVLA proxy dominates or makes AFID redundant;
- no-factor ablation explains the effect;
- standard LoRA explains the effect;
- gate activation everywhere or nowhere;
- global action changes rather than bounded factor-conditioned cell edits;
- clean retention failure;
- action-bound violations;
- identity or checkpoint reload failure;
- use of reward/success/done/object pose/future observation;
- any confirmatory-test task, reset identity, label, or outcome read.

## Conditions For Researcher A

Researcher A must accept:

1. FineVLA remains the closest prior and policy 2.
2. The FineVLA proxy must be fair, transparent, nonprivileged, and matched.
3. AFID novelty is narrowed to deployment-observable action-factor prediction
   plus Base-preserving residual gating.
4. Factor extraction rules must be frozen before Stage 0.
5. Factor labels and masks must be noncollapsed across tasks and phases.
6. Factor prediction must beat trivial validation baselines before gating can
   be treated as observable.
7. The no-factor ablation and standard LoRA remain required.
8. Clean retention and exact Base passthrough are mandatory.
9. No deterministic-action KL is allowed.
10. LCG and all previous methods remain closed.

If these conditions are accepted, proceed to Researcher A rebuttal. If not,
AFID must be rejected before implementation.
