# LCG-VLA Reviewer B Attack

Date: 2026-07-16 KST

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Method: `LCG-VLA`, Language-Contrastive Guidance for Base-preserving SmolVLA
actions.

Proposal: `reports/lcg_vla/researcher_proposal.md`

Proposal SHA-256:
`F0D980AA0760F143D781C723DB632BC324C1E18F390D9C33C5DA94F3A897D11E`

## Summary Judgment

Conditional pass to rebuttal.

LCG has a credible positive prior because CAG directly targets the same failure
axis: vision shortcuts overriding language in VLA action selection. The
proposal also satisfies the current design constraint that LoRA is only
implementation infrastructure. However, the novelty is narrow and fragile.
LCG must not become a renamed CAG proxy, a counterfactual-label augmentation
method, or ordinary LoRA trained on demonstrations.

The method may continue only if Researcher A accepts the conditions below.

## Closest Prior Boundary

Closest prior: Counterfactual Action Guidance.

Primary source: `https://arxiv.org/abs/2602.17659`

CAG is the closest prior because it already proposes comparing a standard
language-conditioned VLA branch against a language-unconditioned vision-action
branch to reduce counterfactual language-following failures.

LCG novelty is not:

- feeding an empty instruction to SmolVLA;
- subtracting a language-null action from a language-conditioned action;
- training a small residual adapter;
- generating counterfactual labels;
- applying standard LoRA to language-following data;
- or claiming CAG under a new acronym.

Permitted novelty boundary:

`A frozen-SmolVLA, Base-preserving, identity-initialized action-cell gate that
learns when deployment-observable original-versus-null language contrast
permits bounded residual edits, with exact Base passthrough when the contrast
is absent or unreliable.`

## Required Policy Order

The first serious comparison must keep:

1. `smolvla_base`
2. `counterfactual_action_guidance_proxy`
3. `lcg_full`
4. `lcg_no_language_contrast_ablation`
5. `standard_lora`

The CAG proxy must not be a strawman. If official CAG code or assets are not
locally installed, policy 2 must transparently use the same SmolVLA Base, same
original and null-language chunks, same action postprocessor, and validation
only for any guidance coefficient.

`lcg_no_language_contrast_ablation` must keep the same trainable capacity and
optimizer budget while removing `N_t` and `C_t` from the gate/residual input.

## Major Risks

### Risk 1: The Null Instruction Is Not A VA Branch

CAG uses a language-unconditioned vision-action module. Feeding SmolVLA an
empty or generic instruction is only a proxy. It may produce undefined,
distribution-shifted, or task-prior actions rather than a clean vision-only
branch.

Required rebuttal: define `l_null`, justify why it is a legal local CAG proxy,
and require Stage 0 to test whether null-branch actions are finite,
postprocessor-valid, noncollapsed, and not globally destructive.

### Risk 2: Contrast Does Not Imply Correct Residual Direction

`B_t - N_t` may identify that language matters, but it does not prove the
direction from Base toward expert action. A high contrast could indicate a
spurious text sensitivity rather than useful correction.

Required rebuttal: separate language-contrast observability from residual
target quality. Stage 0 must report whether contrast magnitude predicts
Base-to-demonstration residual improvement above trivial task/phase baselines.

### Risk 3: CAG Proxy May Dominate

If training-free CAG guidance already fixes the local headroom, LCG is not
needed. Conversely, if the CAG proxy is weak because `l_null` is weak, LCG's
prior comparison is unfair.

Required Stage 0 gate: LCG cannot advance if the CAG proxy dominates, if LCG is
equivalent to a tuned CAG coefficient, or if no residual headroom remains after
the CAG proxy.

### Risk 4: Counterfactual Text Can Leak Test Identity

The proposal permits optional counterfactual instruction swaps from
development task text. This is acceptable only if the swaps never use
confirmatory task/reset identities, held-out labels, or failure inspection.

Required rebuttal: freeze the source of legal instruction alternatives, the
exact discovery/validation partitions, and a no-overlap check before any
counterfactual-language construction.

### Risk 5: Language Contrast May Collapse

Existing LIBERO tasks may not provide enough visually similar but linguistically
different states. If `B_t` and `N_t` are nearly identical everywhere, the gate
has no signal. If they differ everywhere, the gate becomes global and unsafe.

Required Stage 0 gate: Base/null contrast and the language mask must be
noncollapsed across tasks, phases, and action groups. All-zero and all-one
masks are data failures.

### Risk 6: Standard LoRA Can Explain The Gain

LCG trains a module on demonstrations. A reviewer will ask whether ordinary
LoRA with clean retention performs the same. Standard LoRA must remain policy
5 and must use matched data, optimizer budget, and action postprocessing.

### Risk 7: Clean Behavior Can Be Damaged

Instruction sensitivity is not always failure. Some states should remain Base.
The identity-preserving claim is only meaningful if inactive gates preserve
Base exactly and active gates are bounded.

Required diagnostics: gate frequency, action-group deltas, action validity,
clean-retention rows, Base reload identity, and examples of Base/Ours/residual
where the gate activates.

## Mathematical Audit Requirements

The mathematical audit must freeze:

- `l_null` text and tokenizer handling;
- original branch `B_t`, null branch `N_t`, contrast `C_t`, and their shapes;
- horizon `H`, action dimension `D`, and any context horizon;
- gate `G_theta`, residual `Delta_theta`, initialization, and exact support;
- residual caps by translation, rotation, and gripper group;
- language-contrast mask construction and noncollapse thresholds;
- clean-retention objective;
- validity loss or postprocessor contract;
- coefficient search budget;
- gradient paths and frozen-Base parameter checks;
- CAG proxy formula and validation-only coefficient selection;
- no deterministic-action KL.

If a KL term is proposed, reject it unless valid distributions, supports,
direction, estimator, gradient flow, and alternatives are justified. Do not
compute KL directly between deterministic 7D actions.

## Required Ablations And Diagnostics

1. `lcg_no_language_contrast_ablation`
2. `counterfactual_action_guidance_proxy`
3. matched `standard_lora`
4. contrast-magnitude-only gate diagnostic
5. task/phase residual baseline diagnostic
6. clean-retention and inactive-gate exact-Base diagnostics

## Stage 0 Must Stop For

- collapsed Base/null contrast;
- invalid or distribution-shifted null branch actions;
- collapsed language mask;
- no residual headroom beyond the CAG proxy;
- LCG equivalent to CAG coefficient tuning;
- no-language-contrast ablation explains the effect;
- standard LoRA explains the effect;
- global action changes rather than bounded cell edits;
- clean retention failure;
- action-bound violations;
- identity or checkpoint reload failure;
- use of reward/success/done/object pose/future observation;
- any confirmatory-test task, reset identity, label, or outcome read.

## Conditions For Researcher A

Researcher A must accept:

1. CAG remains the closest prior and policy 2.
2. The null-instruction branch is only a transparent local proxy unless
   official CAG assets are installed and verified.
3. LCG novelty is narrowed to a frozen-SmolVLA Base-preserving learned
   language-contrast action-cell gate.
4. `B_t - N_t` may gate edits but cannot by itself be treated as the correct
   residual target.
5. Counterfactual instruction alternatives must come only from
   discovery/validation text with zero confirmatory overlap.
6. Stage 0 must prove contrast, residual, and mask noncollapse.
7. CAG proxy, no-language-contrast ablation, and standard LoRA remain required.
8. Clean retention and exact Base passthrough are mandatory.
9. No deterministic-action KL is allowed.
10. S2C and all previous methods remain closed.

If these conditions are accepted, proceed to Researcher A rebuttal. If not,
LCG must be rejected before implementation.
