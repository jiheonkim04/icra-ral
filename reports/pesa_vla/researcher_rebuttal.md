# PESA-VLA Researcher A Rebuttal

Date: 2026-07-15 KST

Proposal: `reports/pesa_vla/researcher_proposal.md`

Proposal hash: `B05B1ACF7CD3514365B418E25C7E995604FCA8C117CDC0F3384F1046BAF26B63`

Reviewer attack: `reports/pesa_vla/reviewer_attack.md`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Rebuttal decision: `PESA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

## Accepted Narrow Claim

Reviewer B is correct that PESA-VLA cannot claim broad novelty for:

- preserving a pretrained VLA prior;
- using frozen prior and adaptation experts;
- fixed-rank or adaptive LoRA for VLA transfer;
- spectral low-rank capacity allocation;
- generalized/specialized expert routing;
- generic clean-retention mixing.

The claim is narrowed to the local frozen-SmolVLA 7D formulation:

> A Base-passthrough prior-expert policy whose emitted action remains an explicit frozen SmolVLA action plus a bounded spectral-capacity adaptation delta can improve closed-loop success only if the full method beats Base, a PriorVLA-style local proxy, a no-spectral/no-prior-query ablation, and one strongest standard-LoRA or clean-retention simple baseline under the frozen five-policy comparison.

If any of those comparator policies explains the result, PESA is not a paper-worthy method.

## Closest-Prior Proxy Commitment

`priorvla_style_proxy` is a faithful transparent local proxy, not an official PriorVLA reproduction.

It will be constructed to test PriorVLA's relevant claim axis locally:

- frozen SmolVLA remains an explicit read-only prior action source;
- a standard adaptation expert is trained on the same development data as PESA;
- a prior-query or retention gate may select between Base-like and adapted behavior;
- no spectral capacity allocation is included;
- inference budget and available inputs are matched as closely as local code permits.

The proxy must be labeled this way in all manifests, reports, and figures unless exact official PriorVLA equivalence is independently established. If the proxy cannot be built faithfully enough for a fair comparison, Stage 0 must stop as `DATA_OR_SUPERVISION_FAILURE`, not proceed to rollout.

## Simple-Killer Commitment

The fifth policy remains `standard_lora_or_clean_retention_baseline`.

It will be chosen before confirmatory testing using development validation only. The allowed choices are:

- a standard fixed-rank 7D LoRA or adapter baseline;
- a clean-retention LoRA mixture if validation shows it is the stronger simple explanation.

This simple killer is not decorative. If it matches or beats PESA under the frozen comparison, PESA is killed as `SIMPLE_BASELINE_EXPLAINS_METHOD`.

The MTF no-retention result is treated as live negative prior evidence: structured retention can lose to simpler adaptation. PESA must therefore prove that spectral prior-query structure contributes beyond a simple adapter or clean-retention mixture before any paper-candidate claim.

## Component-Isolation Commitment

The required five-policy comparison isolates the three claimed pieces:

1. `priorvla_style_proxy` tests whether explicit frozen-prior adaptation alone explains the result.
2. `pesa_no_spectral_no_prior_query_ablation` tests whether the full method is just ordinary adaptation without the claimed spectral/query mechanism.
3. `standard_lora_or_clean_retention_baseline` tests whether a simple adapter or retention mixture explains the result.

PESA advances only if the full method is action-distinct from all three before rollout and beats them in the frozen closed-loop comparison. If spectral activation alone explains the result, the claim collapses into LoRA-SP/VLA-GSE. If prior-query retention alone explains it, the claim collapses into a simple clean-retention method.

## Partition And Leakage Response

Discovery and validation may be used for:

- label construction;
- spectral threshold or coefficient selection;
- prior-query architecture selection;
- simple-killer selection;
- headroom checks;
- clean-retention checks;
- mechanism diagnostics;
- bounded validation search.

Confirmatory test task/reset identities may not be used for:

- spectral energy threshold selection;
- LoRA rank selection;
- retention coefficient selection;
- query label definition;
- simple-killer choice;
- policy-list changes;
- kill or success threshold changes.

If Base actions are regenerated, the exact frozen Base checkpoint, preprocessing path, and action normalization will be recorded before any training or rollout. Split overlap must be checked across sample, frame, task, episode, and reset identity keys.

## Mathematical Audit Commitments

The mathematical mechanism audit must define:

- `x_t` and its feature shape;
- `a_base_t`, `a_exp_t`, `a_adapt_t`, and `a_pesa_t` as 7D action chunks or per-step 7D actions;
- layer-wise `U_l`, `V_l`, `s_l(x_t)`, energy `E_l(k)`, active rank `k_l(x_t)`, and mask construction;
- whether `s_l(x_t)` is nonnegative by `softplus`, normalized by total energy, temperature-scaled, clipped, or otherwise bounded;
- the gradient path through imitation, prior retention, spectral concentration, optional query BCE, and delta regularization;
- term magnitudes and gradient norm ratios on a small train/validation batch before larger training.

No KL divergence may be used between deterministic 7D action vectors. The default action distances are Huber/L1 for imitation and retention, L2 or grouped L2 for action deltas, and explicit spectral-energy summaries for capacity concentration.

If action clipping is used, the audit must state whether clipping participates in training gradients or is a deployment validity safeguard. Translation, rotation, and gripper deltas must be reported separately.

## Stage 0 Stop Rules

Before rollout, Stage 0 must stop if any of the following occurs:

- fixed 7D action labels or frozen Base actions are missing;
- train/validation/reserved or reset-identity overlap is nonzero;
- standard LoRA and the PriorVLA-style proxy show no development headroom;
- prior-query labels are all-zero, all-one, task shortcuts, or not validation-predictable above majority;
- spectral activation is always off, always on, or effectively identical across tasks;
- PESA full is action-equivalent to the PriorVLA-style proxy, simple baseline, or key ablation;
- initial emitted action is not equal to Base within numerical tolerance;
- action validity is below `1.0`;
- clean validation deltas are globally destructive;
- checkpoint save/reload changes policy identity.

These stops are development classifications, not closed-loop scientific kills.

## Response To Reviewer Verdict

Researcher A accepts the conditional pass and narrows the method accordingly.

PESA proceeds to mathematical mechanism audit only under the following frozen commitments:

- closest prior is `PriorVLA`;
- secondary priors are `LoRA-SP` and `VLA-GSE`;
- `priorvla_style_proxy` is a local proxy, not an official reproduction;
- first serious comparison uses exactly five policies;
- one strongest simple LoRA or clean-retention baseline remains live;
- no deterministic-action KL is allowed;
- no confirmatory-test tuning is allowed;
- no implementation, training, rollout, or manifest freeze may occur before the mathematical audit documents variables, shapes, losses, gradient paths, scale checks, and required ablations.

Decision: `PESA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`.
