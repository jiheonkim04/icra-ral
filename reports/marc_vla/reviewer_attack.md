# MARC-VLA Reviewer B Attack

Date: 2026-07-15 KST

Reviewed frozen proposal: `reports/marc_vla/researcher_proposal.md`

Proposal hash: `D1F910465D4E415C996B3F8C7CE2B2CF47339EA94D697B06A9DCED49AC1E585A`

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

## Closest Prior Attack

Closest prior: OpenVLA-OFT, https://arxiv.org/abs/2502.19645.

OpenVLA-OFT already claims the central positive result that continuous L1-style action decoding with action chunking and parallel decoding can strongly improve VLA adaptation. MARC-VLA cannot claim that L1 continuous action prediction itself is novel. Its possible novelty is narrower:

- frozen SmolVLA preservation;
- median-anchor correction rather than action-head replacement;
- learned disagreement gate rather than always-on L1 replacement;
- direct early comparison to an OpenVLA-OFT-style local L1 proxy;
- direct early comparison to a static Base/L1 mixture.

The proposal must state that `openvla_oft_l1_proxy` is a faithful transparent local proxy, not an official OpenVLA-OFT reproduction.

## Trivial Equivalence Risks

1. Ordinary L1 adapter

If MARC's gate activates on most frames, the method collapses to a plain L1/Huber adapter. Stage 0 and validation must require full-versus-L1-proxy behavioral separation.

2. Static Base/L1 mixture

If the selected gate behaves like a constant scalar mixture, a validation-selected static mixture explains the method. This must be the mandatory simple killer baseline.

3. Generic residual correction

If the method is trained and reported only as `expert - base` residual prediction, it collapses toward prior residual routes that have repeatedly failed. The anchor loss must be documented as action-anchor prediction, and the ablation must test whether the learned disagreement gate matters.

4. MTF no-retention or standard LoRA

MTF's no-retention ablation was strong in Stage B. MARC cannot use "standard adapter training improves Base" as novelty. It must beat a plain L1/action-adapter proxy under the same first comparison.

5. DAGR route rescue

No action-group routing, gripper-transition thresholds, or DAGR residual heads may be reused as MARC's core mechanism.

## Leakage And Partition Risks

- Disagreement thresholds must be computed from training split only.
- Validation may select alpha and gate architecture only inside the preregistered bounded search.
- Confirmatory task/reset identities may not influence threshold, alpha, gate architecture, policy list, or decision thresholds.
- If existing stable prediction artifacts are reused, split overlap must be proven at sample, frame, task/reset, and reserved-test levels.
- No reward, success, simulator object pose, reset identity, or future action may be used at inference.

## Mathematical Risks

- Do not describe deterministic L1 anchor training as estimating a calibrated probability distribution unless a density model is explicitly defined.
- Do not compute KL between deterministic 7D actions.
- Define tensor shapes, clipping, action units, loss scales, and gradient flow.
- Report small-batch loss magnitudes and gradient norms before expensive training.
- If action deltas are clipped at inference, document the train/inference mismatch or train with the clipped action formula.

## Data Health Risks

Reject before rollout if:

- disagreement labels are all-zero/all-one or below minimum positive/negative counts;
- validation gate prediction does not beat the trivial majority baseline;
- the L1 proxy itself is invalid, collapsed, or worse than trivial action baselines;
- MARC full and L1 proxy receive effectively identical targets and produce indistinguishable actions;
- the no-gate ablation or static mixture matches MARC on development metrics;
- clean validation deltas are globally active or destructive.

## Required First Comparison

The first serious comparison must use exactly five policies:

1. `frozen_smolvla`
2. `openvla_oft_l1_proxy`
3. `marc_full`
4. `marc_no_disagreement_gate_ablation`
5. `static_l1_mixture_baseline`

No additional internal controls may precede this comparison unless Stage 0 exposes a concrete implementation ambiguity that would otherwise invalidate one of these five policies.

## Reviewer Verdict

Do not reject before implementation. MARC-VLA is not an exact duplicate of OpenVLA-OFT because it preserves frozen SmolVLA and uses a gated median anchor rather than replacing the action head. However, novelty is conditional and narrow. The method is viable only if Stage 0 proves noncollapsed observable disagreement labels, if MARC differs from the L1 proxy and static mixture, and if identity-preserving action deltas are bounded before rollout.
