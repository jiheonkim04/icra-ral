# EAC-VLA Reviewer B Attack

Date: 2026-07-15 KST

Reviewed proposal: `reports/eac_vla/researcher_proposal.md`

Proposal hash reviewed: `A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E`

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

## Summary Ruling

Do not kill EAC-VLA before implementation. The proposal has a real positive prior, a real local intervention surface, and a stronger identity-preserving story than recent failed adapter/residual methods.

However, novelty and validity are narrow. EAC must be treated as a local AAC extension and may not claim generic adaptive chunking novelty. It can proceed only if Researcher A accepts the constraints below and the mathematical audit/preregistration make the Stage 0 gates enforceable.

## Attack 1: Closest-Prior Duplication Risk

The closest prior is Adaptive Action Chunking. AAC already states the key idea: use action entropy to adapt chunk size at inference time. The EAC proposal is therefore not novel at the level of "adaptive chunking" or "entropy chooses chunk length."

EAC's only defensible novelty is local and conditional:

- SmolVLA-specific uncertainty-source validation;
- queue-boundary risk beyond entropy-only AAC;
- hysteresis/retention to avoid mode-jumping;
- exact 7D action-value passthrough;
- direct comparison to an AAC entropy-only proxy and fixed short replan.

Required rebuttal:

- Explicitly disclaim broad adaptive-chunking novelty.
- State that AAC entropy-only proxy dominance kills or archives EAC's local contribution.
- Keep `aac_entropy_proxy` as policy 2 in the first five-policy comparison.

## Attack 2: RCV And Fixed-Replan Equivalence

The campaign already tested RCV, a current-state queued-vs-fresh replanning method. RCV full lost to no-context and stateless first-action baselines. EAC is dangerously close to becoming another queue-flush heuristic.

The difference from RCV must be mechanistic and measured:

- RCV used learned current-state validity/replan behavior.
- EAC must use action-distribution uncertainty and queue-boundary risk.
- The fixed short-replan baseline must remain a reviewer-killer.

Required rebuttal:

- Freeze `fixed_short_replan_baseline` as the single simple killer.
- Treat EAC as invalid if its selected commitment map is equivalent to a fixed cadence.
- Report action chunks generated, policy calls per step, latency, queue flush rate, and commitment-length distribution.

## Attack 3: Entropy Validity Is Not Guaranteed

SmolVLA flow outputs are not automatically calibrated probability distributions. The proposal must not call a deterministic 7D action vector "entropy." Repeated noise samples may measure stochastic variation, but that is not automatically epistemic uncertainty or AAC-equivalent entropy.

Hard requirements:

- Define the uncertainty statistic exactly before Stage 0.
- If using multi-sample variance, call it variance or dispersion unless a valid normalized distribution and entropy estimator are defined.
- Prove the statistic is finite, noncollapsed, and task/phase-variable on development identities.
- Prove it is computed only from deployment-observable inputs and model outputs.
- Do not use target actions, oracle success, held-out outcomes, or reset identity to calibrate uncertainty.

Required rebuttal:

- Replace any casual "entropy" claim with "entropy or dispersion proxy" unless a valid distribution is defined.
- In the mathematical audit, specify support, normalization, estimator, and gradient-free status if entropy is used.

## Attack 4: Inference-Budget Confound

EAC can improve or harm success simply by changing the number of heavy policy calls per episode. A method that calls SmolVLA much more often may gain reactivity but lose deployability; a method that calls less often may gain smoothness but mask stale control.

Required rebuttal:

- Include latency, policy calls per step, action chunks generated, VRAM, and wall time in Stage A/B reports.
- Ensure the AAC proxy, EAC full, ablation, and fixed short-replan baseline all declare their expected call budget.
- Do not claim pure policy-quality improvement without reporting compute/latency trade-off.

## Attack 5: Action-Value Preservation Must Be Exact

The proposal claims exact 7D action-value passthrough. This is a strong identity-preserving claim, but it can be broken by postprocessing order, queue slicing, internal policy reset state, or wrapper bugs.

Hard requirements:

- Stage 0 must compare Base fixed-queue action values and EAC action values before scheduling.
- Any transformation, clipping, smoothing, rescaling, averaging, or low-pass filtering makes this a different method and invalidates the proposal.
- If queue flushing resets hidden policy state in a way that changes the next chunk distribution, the method must report that as part of scheduling rather than hide it as action passthrough.

Required rebuttal:

- Define the exact equality test for emitted action values.
- Treat action-value modification as `IMPLEMENTATION_FAILURE`, not a valid EAC result.

## Attack 6: Hysteresis Could Be The Whole Method

The proposal adds hysteresis or a retention band. If hysteresis alone explains the effect, the method may be a generic fixed scheduler rather than an uncertainty-calibrated AAC extension.

Required rebuttal:

- Keep `eac_no_calibration_no_hysteresis_ablation` or a sharper ablation that removes the uncertainty-specific component while preserving comparable queue mechanics.
- In Stage 0/validation, report whether commitment lengths differ between high-uncertainty and low-uncertainty states.
- If the selected configuration maps nearly all states to one commitment length, stop as exact trivial equivalence.

## Attack 7: Validation Search Must Not Become Test Tuning

Because EAC has few parameters, it is tempting to try many thresholds and commitment maps until a small rollout looks good. That is prohibited.

Hard requirements:

- At most six total configurations.
- Use discovery/validation identities only.
- Freeze the final threshold/map before Stage A.
- Save all tried configurations and negative results.
- Do not alter task/reset identities after seeing confirmatory outcomes.

## Attack 8: External Prior Proxy Must Be Honest

The AAC proxy cannot be called an official AAC reproduction unless the exact official implementation, entropy source, backbone assumptions, and benchmark protocol are reproduced.

Required wording:

`aac_entropy_proxy` is a faithful transparent local proxy, not an official AAC reproduction.

If the proxy is only variance-based and not entropy-based, label it accordingly.

## Required Five-Policy Comparison

The first serious comparison must remain exactly:

1. `frozen_smolvla_fixed_queue`
2. `aac_entropy_proxy`
3. `eac_full`
4. `eac_no_calibration_no_hysteresis_ablation`
5. `fixed_short_replan_baseline`

No extra simple baselines may be added before the first prior comparison unless the mathematical audit identifies a concrete equivalence that could change the decision and is cheaper than proceeding.

## Required Stage 0 Stop Rules

Stage 0 must stop before rollout for any of:

- queue length cannot be controlled without changing action semantics;
- uncertainty/dispersion statistic is collapsed or nonfinite;
- selected commitment map is effectively constant;
- EAC equals fixed short replan, Base fixed queue, or AAC entropy-only proxy;
- action values differ from Base before scheduling;
- latency overhead is catastrophic;
- development/validation/test identity separation fails;
- hidden confirmatory identities or outcomes are used.

## Reviewer B Decision

`REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

EAC may proceed to Researcher A rebuttal because it is externally anchored, locally feasible, identity-preserving by construction, and meaningfully different from action-residual methods. It cannot proceed to implementation, Stage 0, validation search, or rollout until Researcher A accepts or resolves the constraints above.
