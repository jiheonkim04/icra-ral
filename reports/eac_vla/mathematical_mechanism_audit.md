# EAC-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Method: `EAC-VLA`, Entropy-Calibrated Adaptive Chunking for frozen SmolVLA.

Proposal hash: `A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Rebuttal decision: `EAC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Audit decision: `EAC_MATHEMATICAL_AUDIT_PREREGISTERED`

## Core Mechanism

EAC changes only action-queue scheduling for a frozen SmolVLA policy. It does not modify SmolVLA weights, learned adapters, action postprocessing, action scaling, or emitted 7D action values.

At an observation refresh time `t`, frozen SmolVLA emits a postprocessed action chunk:

`A_t = [a_{t,0}, ..., a_{t,H-1}] in R^{H x D}`

with:

- horizon `H = 50`;
- action dimension `D = 7`;
- action vector `a_{t,i} = (dx, dy, dz, droll, dpitch, dyaw, grip)`;
- official postprocessing and relative 7D LIBERO action semantics unchanged.

EAC computes a scalar risk score from deployment-observable chunk statistics, maps that risk to a commitment length, executes only the selected prefix of the current chunk, and then refreshes the observation.

## Variables And Shapes

Runtime observation and policy objects:

- `o_t`: official preprocessed observation batch, including RGB cameras, proprioception, and task language as supported by the frozen SmolVLA runner.
- `A_t in R^{50 x 7}`: frozen postprocessed action chunk from the current observation.
- `a_{prev} in R^7`: previously executed postprocessed action, or zeros/first action sentinel at episode start.
- `q_t in N`: current queue length before the scheduling decision.
- `K in N`: number of repeated chunks used for uncertainty/dispersion audit. Stage 0 must set `K >= 2` if a dispersion statistic is used.
- `A_t^{(k)} in R^{50 x 7}` for `k = 1..K`: repeated stochastic or repeated deterministic chunks from the same deployment observation, only if the official policy call supports legal repeated prediction.

Derived statistics:

- `U_t in R_{\ge 0}`: uncertainty or dispersion statistic.
- `D_t in R_{\ge 0}`: within-chunk discontinuity statistic.
- `B_t in R_{\ge 0}`: boundary jump statistic.
- `R_t in R`: scalar queue-risk score.
- `C_t in {1, 2, 4, 8, 16, 50}`: selected commitment length.

No variable may depend on target actions, expert labels, simulator object-state, reward, success, reset identity, future observations, held-out test outcomes, or confirmatory identities.

## Uncertainty Or Dispersion Statistic

Preferred Stage 0 path: use a dispersion proxy unless a valid predictive distribution can be established.

Dispersion proxy:

`U_t = mean_{i,d} Var_k(A_{t,i,d}^{(k)})`

where the variance is over repeated legal policy predictions from the same observation.

Alternative pairwise dispersion:

`U_t = mean_{k<l} ||A_t^{(k)} - A_t^{(l)}||_F / sqrt(H D)`

These are not called entropy unless the audit proves the samples come from a normalized predictive distribution with defined support and a stable estimator.

Valid entropy path, if used:

- define support over chunk samples or discretized commitment-relevant bins;
- define normalization;
- define estimator;
- justify sample source as predictive rather than arbitrary noise;
- report estimator stability on development identities.

No KL divergence is used. Deterministic 7D actions are not probability distributions.

## Chunk And Boundary Statistics

Within-chunk discontinuity:

`D_t = mean_{i=1..H-1} ||a_{t,i} - a_{t,i-1}||_2`

Boundary jump:

`B_t = ||a_{t,0} - a_{prev}||_2`

Optional normalized variants may divide translation, rotation, and gripper groups by development-set robust scales, but the exact scales must be frozen before validation search and reported. If scaling changes action values, it is invalid; scaling may only normalize statistics used for queue scheduling.

## Queue-Risk Score

The default score is rule-based and has no trainable parameters:

`R_t = w_u * norm(U_t) + w_d * norm(D_t) + w_b * norm(B_t)`

Default fixed weights before validation:

- `w_u = 0.5`
- `w_d = 0.25`
- `w_b = 0.25`

`norm(.)` denotes development-only robust min/max or percentile normalization, frozen before validation search. If Stage 0 shows one term collapsed, the method must either stop as `DESIGN_FAILURE` or preregister a reduced score before validation; it may not silently drop terms after seeing confirmatory outcomes.

If a learned scalar calibration is considered, it may only be a lightweight validation-stage option within the six-configuration budget, and its parameters must be trained on discovery/validation identities only. The default protocol should prefer the rule-based score.

## Commitment Map

Commitment set:

`C = {1, 2, 4, 8, 16, 50}`

The simplest monotone map:

- high risk -> shorter commitment;
- low risk -> longer commitment.

Example two-threshold map:

- `R_t >= tau_high`: `C_t = 1 or 2`;
- `tau_low <= R_t < tau_high`: `C_t = 4 or 8`;
- `R_t < tau_low`: `C_t = 16 or 50`.

However, validation search may tune at most one critical threshold and one commitment/hysteresis setting. The final map must be frozen before confirmatory Stage A.

Hysteresis:

Let `C_{t-1}` be the previous commitment length. Hysteresis may prevent oscillation by requiring a margin `h >= 0` before switching between long and short regimes. Hysteresis cannot change action values, only queue commitment.

If the selected map assigns more than `90%` of development decisions to one commitment length, EAC is effectively constant and must stop before rollout unless the preregistered audit justifies a stricter threshold.

## Action-Value Passthrough

For every refreshed observation, define:

- `A_t^base`: Base postprocessed chunk;
- `A_t^eac`: EAC pre-scheduling postprocessed chunk.

Required equality:

`max_{i,d} |A_{t,i,d}^eac - A_{t,i,d}^base| <= epsilon_eq`

with `epsilon_eq = 1e-7` for float serialization/device roundoff unless the implementation audit proves exact tensor identity is available.

Any smoothing, averaging, clipping, rescaling, learned residual, low-pass filtering, or action-value replacement violates this audit and must be classified as `IMPLEMENTATION_FAILURE`.

## Training Objective

There is no policy training objective in the default EAC method.

The method may perform validation-only selection over rule parameters. If a learned scalar risk calibration is introduced within the six-config budget, its objective must be:

`L_cal = BCE(y_commit, p_short)`

or a similarly simple calibration loss, where labels are constructed only from discovery/validation chunk-risk diagnostics and not from confirmatory success. This is disfavored unless Stage 0 proves rule-based scoring is inadequate and the labels are noncollapsed.

No objective term may use confirmatory outcomes.

## Validation Selection Score

For each of at most six configurations, compute:

`S = s_proxy + s_clean + s_active + s_valid - p_latency - p_oscillation`

where:

- `s_proxy`: validation closed-loop success if a validation rollout is preregistered, otherwise a frozen queue-quality proxy;
- `s_clean`: clean retention term requiring no action-value changes and no action-bound violations;
- `s_active`: mechanism activation score rewarding noncollapsed commitment choices;
- `s_valid`: action validity and finite-output score;
- `p_latency`: penalty for extra policy calls or wall-clock overhead;
- `p_oscillation`: penalty for rapid commitment switching.

Exact weights must be frozen in preregistration before validation search. Do not select by offline action L2.

## Gradient Path

Default EAC has no trainable policy parameters and no gradient path through SmolVLA. The frozen policy is evaluated in inference mode.

If a learned scalar calibrator is explicitly allowed later, gradients flow only through the calibrator parameters. Gradients must not update SmolVLA, action postprocessors, or policy adapters.

The Stage 0 smoke for default EAC should therefore report "no trainable parameter gradients by design" rather than fabricating gradient norms.

## Required Ablation

Key ablation:

`eac_no_calibration_no_hysteresis_ablation`

This ablation preserves queue mechanics but removes the EAC-specific calibration/hysteresis component. If the full method does not beat this ablation in the frozen paired comparison, the EAC-specific mechanism is not supported.

Closest prior proxy:

`aac_entropy_proxy`

This uses the closest available AAC-style entropy or dispersion-only commitment rule and is explicitly a faithful transparent local proxy, not an official AAC reproduction.

Simple reviewer-killer:

`fixed_short_replan_baseline`

This uses a fixed short commitment or fixed periodic queue flush. It remains mandatory because prior RCV evidence showed naive replanning can beat a learned queue-validity method.

## Stage 0 Audit Requirements

Stage 0 must verify:

- official Base chunk shape is `[50, 7]` after postprocessing;
- queue length is observable or controllable;
- queue control does not alter action values;
- repeated/stochastic chunks are available if `U_t` uses repeated samples;
- `U_t`, `D_t`, and `B_t` are finite and noncollapsed;
- commitment choices are noncollapsed and task/phase variable;
- Base passthrough equality holds within `epsilon_eq`;
- action bounds and finite checks pass;
- latency/policy-call overhead is measured;
- discovery/validation/confirmatory identity overlap is zero;
- no privileged inference input is used.

Stage 0 hard stops:

- `DESIGN_FAILURE`: collapsed uncertainty/dispersion, constant commitment map, exact equivalence to Base/AAC proxy/fixed replan, or no usable scheduling surface.
- `DATA_OR_SUPERVISION_FAILURE`: identity split or source health fails.
- `IMPLEMENTATION_FAILURE`: queue control changes action values, breaks postprocessing, corrupts action shape, or cannot resume missing keys safely.
- `NO_HEADROOM`: diagnostics show queue scheduling cannot plausibly affect the claimed failure condition.

These are not closed-loop scientific kills.

## First Serious Comparison

Exactly five policies:

1. `frozen_smolvla_fixed_queue`
2. `aac_entropy_proxy`
3. `eac_full`
4. `eac_no_calibration_no_hysteresis_ablation`
5. `fixed_short_replan_baseline`

Primary metric:

- task-balanced official closed-loop success.

Secondary metrics:

- paired wins/losses/ties;
- per-task success;
- commitment-length distribution;
- queue flush rate;
- action chunks generated;
- policy calls per step;
- latency;
- VRAM;
- finite/action-bound validity;
- smoothness and boundary jump statistics.

## Identity-Preserving Integration Audit

EAC starts from exact Base behavior:

- if commitment length is `50` throughout, behavior equals Base fixed queue;
- emitted action values are Base actions;
- scheduler can default to Base when uncertainty is unavailable.

Disruption risk is limited to timing of observation refresh, not action-value modification. This risk is still real and must be measured through latency, policy calls, smoothness, and closed-loop success.

## Mathematical Decision

`EAC_MATHEMATICAL_AUDIT_PREREGISTERED`

The method is mathematically coherent if and only if the implementation treats uncertainty as a defined entropy estimator or honest dispersion proxy, preserves action values exactly, and bounds validation search. The audit permits preregistration and prototype protocol drafting. It does not permit implementation, Stage 0, validation search, or rollout until those documents are frozen.
