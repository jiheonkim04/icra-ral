# EAC-VLA Researcher A Proposal

Date: 2026-07-15 KST

Method: `EAC-VLA`, Entropy-Calibrated Adaptive Chunking for frozen SmolVLA.

Campaign location: Epoch 4 Cycle 10.

Current prior result boundary: `PESA-VLA` stopped at Stage 0 as a pre-rollout `DESIGN_FAILURE` because prior-query observability failed. This proposal is not a PESA rescue and must not reuse PESA query labels, spectral thresholds, or adapter machinery.

## Research Claim

Frozen SmolVLA already emits a postprocessed `50 x 7` action chunk through the official LeRobot/LIBERO path. The current Base policy commits to a fixed action-queue behavior, even though the appropriate commitment length should depend on uncertainty: uncertain states need faster perception refresh, while confident states benefit from longer smooth execution.

`EAC-VLA` tests whether a deployment-observable, uncertainty-calibrated queue scheduler can improve official closed-loop LIBERO success by choosing how many actions from the current frozen SmolVLA chunk to execute before refreshing the observation.

The method does not change SmolVLA weights, the 7D action interface, postprocessing, action scaling, task/reset identities, or emitted action values. It changes only the action-queue commitment length.

## Closest Positive Prior

Closest external prior: Adaptive Action Chunking, https://arxiv.org/abs/2604.04161.

Positive result demonstrated by the prior:

- AAC identifies the fixed-chunk trade-off: long chunks reduce reactivity, while short chunks can induce mode-jumping and jerky discontinuities.
- AAC uses action entropy to adapt chunk size at inference time.
- The paper reports improved VLA manipulation performance across simulated and real-world tasks.
- Public AAC project/code links are available at https://lance-lot.github.io/adaptive-chunking.github.io/ and https://github.com/orgs/Adaptive-Action-Chunking/repositories.

Secondary priors:

- AR-VLA, https://arxiv.org/abs/2603.10126, motivates temporal action consistency and re-anchoring under refreshable vision-language prefixes.
- AC2-VLA, https://arxiv.org/abs/2601.19634, supports action-context-aware deployment decisions, though its primary axis is compute.

## Limitation Extended

AAC assumes that a useful action-entropy signal is available from the deployed VLA. Local SmolVLA may expose deterministic or weakly stochastic flow outputs, so the first scientific question is not whether an arbitrary queue wrapper helps; it is whether a legal, deployment-observable action-uncertainty signal exists and can select chunk commitments better than fixed scheduling.

EAC extends AAC locally by adding:

- a Stage 0 uncertainty-source audit for SmolVLA action chunks;
- queue-boundary risk features that do not require labels at inference;
- a hysteresis or retention band to prevent mode-jumping from frequent queue flushing;
- strict action-value passthrough so any effect is due to scheduling, not action correction;
- explicit comparison to an AAC entropy-only proxy and a fixed short-replan simple killer.

## Mechanism

Inputs available at inference:

- current RGB/proprio/language batch after official preprocessing;
- current frozen SmolVLA postprocessed action chunk `A_t in R^{50 x 7}`;
- optional repeated or stochastic chunks from the same observation if the policy supports legal noise sampling;
- previous executed action `a_{t-1} in R^7`;
- current action-queue length.

No inference input may include simulator object-state, reward, success, reset identity, future observations, target action labels, or held-out outcome information.

EAC computes a scalar queue-risk score:

`r_t = f(u_t, d_t, b_t, q_t)`

where:

- `u_t` is an action-uncertainty statistic from multi-sample chunk variance or entropy proxy;
- `d_t` is within-chunk discontinuity or curvature;
- `b_t` is boundary jump risk between `a_{t-1}` and the next proposed action;
- `q_t` is current queue context;
- `f` is a frozen, validation-selected affine or rule-based score with at most one threshold and one hysteresis/commitment-map factor.

The scheduler maps `r_t` to a commitment length:

`c_t in {1, 2, 4, 8, 16, 50}`.

The policy executes the first `c_t` actions from the current frozen chunk, then refreshes observation and predicts a new chunk. If the uncertainty source is unavailable, collapsed, nonfinite, too slow, or fails the Stage 0 audit, EAC defaults to the existing Base fixed-queue behavior and stops before rollout.

## Evidence Partitions

DISCOVERY:

- prior failed-method analysis;
- inspection of official SmolVLA action-queue semantics;
- source audit that policy chunks are `50 x 7`;
- identification of possible legal uncertainty features.

VALIDATION:

- uncertainty noncollapse and predictability checks;
- bounded threshold/hysteresis or commitment-map search;
- latency and action-validity checks;
- validation-only closed-loop proxy or small validation rollout if available under governance;
- one final frozen EAC configuration.

CONFIRMATORY_TEST:

- Stage A and Stage B official LIBERO paired manifests only after method, policy identities, baselines, ablation, metrics, task/reset identities, thresholds, and decision rules are frozen.
- Confirmatory outcomes may not tune thresholds, commitment map, entropy source, task allocation, reset seeds, or policy list.

## Required Stage 0 Audit

Before validation search, training, or rollout, run a bounded development-only Stage 0 audit:

1. Queue surface proof:
   - official policy emits a `50 x 7` chunk;
   - action queue length is observable or controllable;
   - flushing or shortening the queue can be implemented without changing action values or official postprocessing.

2. Uncertainty-source health:
   - repeated or stochastic chunk generation works on development observations;
   - uncertainty statistics are finite and noncollapsed;
   - uncertainty varies across tasks and phases;
   - uncertainty is not a hidden function of held-out success.

3. Commitment-map health:
   - candidate commitment lengths are noncollapsed;
   - high-risk states choose shorter commitments more often than low-risk states;
   - hysteresis prevents rapid oscillation;
   - fixed Base passthrough is exactly recoverable.

4. Policy disruption risk:
   - emitted 7D action values match Base exactly before queue scheduling;
   - all executed actions remain finite and in valid range;
   - action-bound violations are zero;
   - latency and extra policy calls remain within local budget.

5. Separation:
   - no confirmatory task/reset identities are used for threshold selection;
   - all discovery/validation/test identity overlap counts are zero.

Hard stops before rollout:

- `DESIGN_FAILURE`: uncertainty signal collapsed, queue cannot be controlled, or EAC is exactly equivalent to Base/fixed short replan.
- `DATA_OR_SUPERVISION_FAILURE`: development splits or identity separation fail.
- `IMPLEMENTATION_FAILURE`: queue control changes action values, corrupts postprocessing, or cannot resume safely.
- `NO_HEADROOM`: Base and simple scheduling variants saturate the condition or an oracle shows no useful queue intervention.

## Bounded Validation Search

Maximum six configurations:

- up to three values for one critical risk threshold;
- up to two commitment maps or hysteresis settings;
- no combinatorial expansion beyond six total configurations;
- no more than two lightweight seeds if a learned scalar calibration is used.

Selection score:

`validation_score = success_proxy + clean_retention + mechanism_activation + action_validity - latency_penalty - oscillation_penalty`

Exact weights must be frozen in the mathematical audit or preregistration before search. Do not select purely by offline action L2.

## First Serious Comparison

Exactly five policies:

1. `frozen_smolvla_fixed_queue`
2. `aac_entropy_proxy`
3. `eac_full`
4. `eac_no_calibration_no_hysteresis_ablation`
5. `fixed_short_replan_baseline`

`aac_entropy_proxy` is a faithful transparent local proxy, not an official AAC reproduction unless exact official equivalence is independently established.

The fixed short-replan baseline is the single mandatory simple reviewer-killer because prior RCV evidence showed that naive replanning can beat a learned queue-validity method.

## Expected Failure Modes

- SmolVLA uncertainty is unavailable or collapsed.
- Entropy-only AAC proxy matches full EAC.
- Fixed short replanning matches or beats full EAC.
- Hysteresis removes too much reactivity or too little discontinuity.
- Frequent refresh increases latency enough to harm closed-loop success.
- Queue control accidentally changes action values or official postprocessing.
- Stage A result is noncatastrophic and underpowered, requiring Stage B rather than a premature kill.

## Novelty Boundary

EAC does not claim:

- generic novelty for adaptive chunking;
- official AAC reproduction;
- new action head, adapter, or VLA training method;
- improvement from changing action values;
- paper-grade evidence from offline smoothness metrics.

The local contribution is the fixed-protocol test of an AAC-anchored, SmolVLA-specific, identity-preserving queue scheduler with an uncertainty-source audit and a direct five-policy prior-first comparison.

## Immediate Next Step

Hash this proposal. Then Reviewer B must attack novelty, leakage, equivalence to AAC or fixed replan, feasibility of local uncertainty estimation, and the scientific adequacy of the five-policy comparison before any implementation or Stage 0 audit.
