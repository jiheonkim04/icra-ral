# AFID-VLA Mathematical Mechanism Audit

Date: 2026-07-16 KST

Decision: `AFID_MATHEMATICAL_AUDIT_PREREGISTERED`

Proposal: `reports/afid_vla/researcher_proposal.md`

Proposal SHA-256:
`B5D1EE12FF2D0280511452DA7FE55295740FD9942A8BE293F444C8EB157062BC`

Reviewer attack: `reports/afid_vla/reviewer_attack.md`

Researcher rebuttal: `reports/afid_vla/researcher_rebuttal.md`

This audit freezes AFID-VLA's mathematical mechanism before preregistration,
prototype protocol, implementation, training, validation search, rollout,
simulator access, or confirmatory-test access.

## Scope

AFID's allowed novelty is only:

`A frozen-SmolVLA, Base-preserving residual gate driven by
deployment-observable predictions of compact action-factor labels that are
derived from development-only demonstrations, with exact Base passthrough when
factor confidence is low or the factor-conditioned mask is inactive.`

AFID is not FineVLA renamed, not ordinary LoRA, not a new VLA backbone, not a
hand-tuned residual mask, not action-label access at inference, and not a
rescue or reinterpretation of LCG or any previous closed method.

## Constants

- `H = 50`: SmolVLA action chunk horizon.
- `D = 7`: official LIBERO action dimension.
- `B`: batch size.
- `d_trans = {0,1,2}`: translation coordinates.
- `d_rot = {3,4,5}`: rotation coordinates.
- `d_grip = {6}`: gripper coordinate.
- `eps = 1e-6`: numerical floor.

Default residual/action caps in normalized action units:

- `rho_trans = 0.02`;
- `rho_rot = 0.05`;
- `rho_grip = 0.25`.

Fixed factor extraction thresholds:

- `tau_axis_motion = 0.03`;
- `tau_dir = 0.01`;
- `tau_rot = 0.02`;
- `tau_grip_event = 0.20`;
- `tau_settle = 0.015`;
- `tau_residual_mask = 0.50` after residual normalization;
- `tau_conf = 0.60`;
- `tau_entropy = 0.75` normalized entropy.

These constants may not be changed after inspecting Stage 0 outcomes. Any
alternative value requires a new method cycle.

## Variables And Shapes

For each legal development row:

| Symbol | Shape | Source | Gradient path | Meaning |
| --- | --- | --- | --- | --- |
| `o_t` | tuple | legal current observation | frozen Base only unless adapter hooks are declared | current visual input |
| `q_t` | vector | legal current proprioception/state exposed to SmolVLA | frozen Base only unless adapter hooks are declared | current robot state |
| `l_t` | string | legal task instruction | tokenizer only | goal instruction |
| `B_t` | `[B,H,D]` | frozen SmolVLA with `l_t` | stopgrad | Base action chunk |
| `E_t` | `[B,H,D]` | demonstration action chunk | target only | expert action chunk |
| `V_t` | `[B,H,1]` | action-valid row mask | no gradient | valid horizon cells |
| `R_t` | `[B,H,D]` | `E_t - B_t` | target only | Base-to-demonstration residual |
| `S_d` | `[D]` | discovery-only `p95(abs(R_t[:,:,d]))` | no gradient | residual normalization scale |
| `R_norm` | `[B,H,D]` | `abs(R_t) / S_d` | no gradient | normalized residual magnitude |
| `Z_axis` | `[B]` | factor extraction from `E_t` | target only | dominant translation axis |
| `Z_dir` | `[B,3]` | factor extraction from `E_t` | target only | translation signs |
| `Z_grip_type` | `[B]` | factor extraction from gripper trace | target only | gripper event type |
| `Z_grip_bin` | `[B]` | factor extraction from gripper trace | target only | event timing bin |
| `Z_rot` | `[B,3]` | factor extraction from `E_t` | target only | rotation signs |
| `Z_term` | `[B]` | factor extraction from `E_t` and gripper trace | target only | terminal motion class |
| `M_factor` | `[B,H,D]` | discovery-only residual/factor mask | no gradient | factor-conditioned editable cells |
| `X_t` | implementation-defined | legal deployment inputs, SmolVLA features, `B_t` | trainable only through AFID modules | factor/gate features |
| `P_axis` | `[B,4]` | factor predictor | trainable | axis probabilities |
| `P_dir` | `[B,3,3]` | factor predictor | trainable | per-axis sign probabilities |
| `P_grip_type` | `[B,4]` | factor predictor | trainable | gripper type probabilities |
| `P_grip_bin` | `[B,4]` | factor predictor | trainable | gripper timing probabilities |
| `P_rot` | `[B,3,3]` | factor predictor | trainable | per-rotation sign probabilities |
| `P_term` | `[B,6]` | factor predictor | trainable | terminal class probabilities |
| `c_theta` | `[B,1]` | factor predictor | trainable | scalar confidence |
| `Delta_theta` | `[B,H,D]` | residual head | trainable | proposed bounded residual edit |
| `G_theta` | `[B,H,D]` | gate head | trainable | action-cell intervention gate |
| `A_AFID` | `[B,H,D]` | edited chunk | trainable through AFID only | AFID output chunk |

No future observation, object pose, reward, success flag, done flag, expert
future action at inference, confirmatory identity, or confirmatory outcome may
enter any variable above.

## Factor Extraction

Factor labels are generated only on discovery/validation development rows.
They are never available at inference.

Let `T = E_t[:,:,0:3]`, `W = E_t[:,:,3:6]`, and `g = E_t[:,:,6]`.

For each row:

### Dominant Translation Axis

Compute:

`m_j = sum_h abs(T[h,j]) / H` for `j in {0,1,2}`.

If `max_j m_j < tau_axis_motion`, set `Z_axis = none`. Otherwise set
`Z_axis = argmax_j m_j`, with fixed tie break `x > y > z`.

### Translation Direction Signs

For each translation dimension:

`u_j = sum_h T[h,j] / H`.

Set:

- `Z_dir[j] = +1` if `u_j > tau_dir`;
- `Z_dir[j] = -1` if `u_j < -tau_dir`;
- `Z_dir[j] = 0` otherwise.

### Gripper Type And Timing

Let `dg[h] = g[h] - g[h-1]` for `h > 0`.

Let `h_star = argmax_h abs(dg[h])`, tie break to earliest `h`.

If `max_h abs(dg[h]) < tau_grip_event`, set:

- `Z_grip_type = hold`;
- `Z_grip_bin = none`.

Otherwise:

- `Z_grip_type = close` if `dg[h_star] < 0`;
- `Z_grip_type = open` if `dg[h_star] > 0`;
- `Z_grip_bin = early` if `h_star < 17`;
- `Z_grip_bin = mid` if `17 <= h_star < 34`;
- `Z_grip_bin = late` if `h_star >= 34`.

If the gripper trace is missing or constant-invalid, set both labels to
`none` and mark the row as not usable for gripper-factor training.

### Rotation Signs

For each rotation dimension:

`v_j = sum_h W[h,j] / H`.

Set:

- `Z_rot[j] = +1` if `v_j > tau_rot`;
- `Z_rot[j] = -1` if `v_j < -tau_rot`;
- `Z_rot[j] = 0` otherwise.

### Terminal Motion Class

Let:

- `m_first = mean_{h<17} ||T[h,:]||_1`;
- `m_mid = mean_{17<=h<34} ||T[h,:]||_1`;
- `m_last = mean_{h>=34} ||T[h,:]||_1`.

Set:

- `grasp` if `Z_grip_type = close`;
- `release` if `Z_grip_type = open`;
- `settle` if `m_last < tau_settle` and `m_first + m_mid >= 2*tau_settle`;
- `approach` if `m_first >= m_mid` and `m_first >= m_last`;
- `transport` if `m_mid > m_first` and `m_mid >= m_last`;
- `align` otherwise.

The terminal class is a compact process label, not a task-success label.

## Factor Health Gates

Stage 0 must report all factor counts by split, task, phase, and action group.

A factor is usable only if all are true on discovery and validation:

- at least two nonempty classes after removing `none`;
- largest class fraction `<= 0.90`;
- smallest nonempty class count `>= 8` in discovery and `>= 2` in validation;
- no single task contributes more than `0.60` of nonempty positives;
- duplicate `(split, task_suite, task_id, demo_id, window_start)` keys are `0`;
- train/validation/test overlap keys are `0`.

If no factor is usable, Stage 0 stops as
`AFID_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

Unusable individual factors must not drive `G_theta`.

## Factor-Conditioned Mask

Compute discovery-only residual scales:

`S_d = clamp(p95_discovery(abs(R_t[:,:,d])), 1e-4, 10.0)`.

Validation and later partitions use frozen discovery scales.

The normalized residual magnitude is:

`R_norm[:,:,d] = abs(R_t[:,:,d]) / S_d[d]`.

For each usable factor `f`, compute discovery-only editable-cell frequencies:

`p_f[h,d] = mean_{rows with nonempty f} 1[R_norm[h,d] >= tau_residual_mask]`.

The factor-conditioned mask for a row is:

`M_factor[h,d] = 1[p_{predicted_or_label_factor}(h,d) >= 0.20]`.

During supervised training, labels may define the factor key. At inference and
for no-leak validation, the predicted factor key and confidence rule must
define the mask.

Mask stop gates:

- global positive fraction must be in `[0.02, 0.80]`;
- every validation task must have positive fraction in `[0.01, 0.90]`;
- at least one translation or rotation group must be active;
- inactive rows must exist for clean-retention checks.

Collapsed masks stop as `AFID_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

## Identity-Preserving Gate

The residual head is bounded by action group:

`Delta_theta = clip_group(raw_Delta_theta, rho_trans, rho_rot, rho_grip)`.

The factor confidence is:

`c_theta = max_factor_confidence * (1 - normalized_entropy_factor)`.

The gate is:

`G_theta = I[c_theta >= tau_conf] * I[entropy <= tau_entropy] * M_factor *
eta * sigmoid(Z_theta(X_t, P_theta))`.

where:

- `Z_theta` has shape `[B,H,D]`;
- `eta in [0,1]` is a scalar or per-group multiplier initialized exactly to
  `0`;
- any parameterization of `eta` must reproduce exact Base within `1e-6` before
  training and after disk reload.

The output chunk is:

`A_AFID = B_t + G_theta * Delta_theta`.

All multiplication is elementwise. Gradients flow only into AFID trainable
parameters. No gradient flows into frozen SmolVLA Base, `B_t`, `E_t`, `R_t`,
`S_d`, factor labels, or hard masks.

If `M_factor = 0`, `eta = 0`, `c_theta < tau_conf`, or entropy exceeds
`tau_entropy`, AFID must output exact Base.

## FineVLA Action-Factor Proxy

The closest-prior proxy is:

`finevla_action_factor_proxy`.

If compatible official FineVLA assets are unavailable, the local proxy must:

- use the same frozen SmolVLA Base;
- use the same development splits and factor labels;
- use no future action labels at inference;
- use no AFID residual gate;
- use matched optimizer, parameter budget, action postprocessor, and inference
  budget;
- encode factors as fine-grained instruction text or training metadata only.

The proxy must be reported as a transparent local proxy, not official FineVLA.
If it dominates AFID or leaves no residual headroom, AFID stops before bounded
validation.

## Objective Terms

All objective terms are computed on development partitions only. Reductions
are masked by `V_t` and are coordinate means unless stated.

### 1. Factor Prediction Loss

Variables:

- `P_axis [B,4]`, `Z_axis [B]`;
- `P_dir [B,3,3]`, `Z_dir [B,3]`;
- `P_grip_type [B,4]`, `Z_grip_type [B]`;
- `P_grip_bin [B,4]`, `Z_grip_bin [B]`;
- `P_rot [B,3,3]`, `Z_rot [B,3]`;
- `P_term [B,6]`, `Z_term [B]`.

Formula:

`L_factor = CE(P_axis,Z_axis) + mean_j CE(P_dir[:,j],Z_dir[:,j]) +
CE(P_grip_type,Z_grip_type) + CE(P_grip_bin,Z_grip_bin) +
mean_j CE(P_rot[:,j],Z_rot[:,j]) + CE(P_term,Z_term)`.

Rows with unusable or missing labels for a factor are masked out of that
factor's term.

Scale: dimensionless negative log likelihood.

Gradient path: factor predictor parameters only.

Intended effect: make action factors observable from deployment inputs.

Simpler alternative: majority and task/phase factor baselines.

Required diagnostic: validation factor accuracy, macro-F1, and margin over
trivial baselines for each factor used by the gate.

### 2. Factor-Masked Residual Huber

Variables: `A_AFID`, `E_t`, `M_factor`, each `[B,H,D]`.

Formula:

`L_res = mean V_t * M_factor * Huber_delta((A_AFID - E_t) / sigma_d)`.

Default scales:

- `sigma_trans = 0.02`;
- `sigma_rot = 0.05`;
- `sigma_grip = 1.0`;
- `delta = 1.0`.

Units: dimensionless normalized action error.

Gradient path: `A_AFID -> G_theta, Delta_theta -> theta`.

Intended effect: learn bounded residual edits only in factor-relevant cells.

Simpler alternative: matched standard LoRA on demonstrations.

Required ablation: `standard_lora`.

### 3. Clean Retention

Variables: `A_AFID`, `B_t`, `M_factor`.

Formula:

`L_clean = mean V_t * (1 - M_factor) * Huber_delta((A_AFID - B_t) / sigma_d)`.

Scale and units: same as `L_res`.

Gradient path: `A_AFID -> G_theta, Delta_theta -> theta`.

Intended effect: preserve Base where factor confidence or mask is absent.

Simpler alternative: exact Base passthrough.

Required diagnostic: inactive-gate exact-Base report.

### 4. Gate Sparsity

Variable: `G_theta`, shape `[B,H,D]`.

Formula:

`L_gate = mean V_t * G_theta`.

Scale: dimensionless.

Gradient path: `G_theta -> theta`.

Intended effect: prevent global action edits.

Simpler alternative: fixed factor mask without learned gate.

Required ablation: `afid_no_factor_ablation`.

### 5. Action Validity Penalty

Let `post(A)` be the official SmolVLA/LIBERO postprocessor. Let `invalid(A)`
be zero for finite in-bound actions and positive for NaN, inf, or
postprocessor-bound violations.

Formula:

`L_valid = mean invalid(post(A_AFID))`.

Scale: normalized action-bound violation.

Gradient path: implementation-dependent only through `A_AFID`; if the
postprocessor is nondifferentiable, this term is reported as a smoke metric
and not used for gradient updates.

Required diagnostic: action-validity report by action group.

## Total Objective

Default Stage 0 small-fit objective:

`L_total = 1.0 L_factor + 1.0 L_res + 1.0 L_clean + 0.01 L_gate + 1.0 L_valid`.

Before training, Stage 0 must estimate term magnitudes and gradient norms on a
small development batch. If any weighted gradient norm is more than `20x` the
median weighted term norm, the run must stop as
`AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE` unless a bounded
validation-only search is preregistered.

No KL divergence is used. Deterministic 7D actions and SmolVLA flow vectors
are not probability distributions.

## Observability And Headroom Diagnostics

Stage 0 must prove that factors are observable and useful before any bounded
validation search.

Required diagnostics:

- factor accuracy and macro-F1 versus majority baseline;
- factor accuracy and macro-F1 versus task/phase baseline;
- per-factor validation margin over the best trivial baseline;
- factor-conditioned oracle Huber using `M_factor` and clipped `R_t`;
- validation Huber of `finevla_action_factor_proxy` versus Base;
- validation Huber of AFID small-fit versus Base, FineVLA proxy,
  no-factor ablation, and standard LoRA;
- clean-retention delta on inactive/low-confidence rows;
- gate activation by task, phase, factor, timestep, and action group.

Pass to bounded validation is disallowed if:

- every usable factor fails to beat the best trivial baseline by at least
  `0.05` macro-F1 or `0.05` accuracy;
- factor-conditioned oracle reduction over Base is below `2%`;
- FineVLA proxy leaves no measurable residual headroom;
- AFID small-fit is explained by FineVLA proxy, no-factor ablation, or
  standard LoRA;
- gate activation is below `0.02` or above `0.80` globally;
- clean-retention max absolute error exceeds `1e-6` on required exact-Base
  rows.

The oracle is diagnostic only and is not an inference method.

## Stage 0 Pass And Stop Gates

Stage 0 may pass to bounded validation only if all are true:

- proposal hash matches
  `B5D1EE12FF2D0280511452DA7FE55295740FD9942A8BE293F444C8EB157062BC`;
- no reward, success, done, simulator result, object pose, future observation,
  or confirmatory identity is read;
- `B_t`, `E_t`, `R_t`, factor labels, `M_factor`, and predictor outputs parse
  with expected shapes;
- factor labels and masks are noncollapsed;
- factor prediction beats trivial validation baselines;
- factor-conditioned residual headroom exists;
- FineVLA proxy leaves residual headroom;
- initialized and disk-reloaded AFID equals Base within `1e-6`;
- expected AFID parameters receive finite nonzero gradients;
- frozen SmolVLA Base parameters receive no gradients;
- AFID after a small fit differs from Base, FineVLA proxy, no-factor
  ablation, and standard LoRA;
- action deltas respect group caps;
- action postprocessing remains valid;
- clean retention passes on inactive/low-confidence rows.

Stop classes:

- `AFID_STAGE_0_DATA_OR_SUPERVISION_FAILURE`: collapsed factor labels,
  collapsed masks, insufficient coverage, duplicate/split-overlap keys, or no
  usable factors.
- `AFID_STAGE_0_NO_USABLE_HEADROOM`: no factor-conditioned residual headroom
  or FineVLA proxy leaves no measurable headroom.
- `AFID_STAGE_0_DESIGN_FAILURE`: factors are not observable from deployment
  inputs, AFID equals FineVLA proxy, no-factor ablation explains the effect,
  standard LoRA explains the effect, or gate activation is global/nonacting.
- `AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE`: identity reload,
  gradient, shape, objective-scale, action-validity, or exact-Base passthrough
  failures.
- `AFID_STAGE_0_PASS_TO_BOUNDED_VALIDATION`: all Stage 0 gates pass.

Stage 0 is development-only and is not a closed-loop scientific result.

## Required Ablations

1. `finevla_action_factor_proxy`
2. `afid_no_factor_ablation`
3. `standard_lora`
4. majority factor baseline
5. task/phase factor baseline
6. inactive-gate exact-Base diagnostic
7. factor-mask noncollapse diagnostic

## Validation Search Envelope

If Stage 0 passes, bounded validation search may consider at most six
configurations total. The only tunable factors allowed are:

- `tau_conf` from `{0.50, 0.60, 0.70}`;
- one clean-retention coefficient from `{0.5, 1.0, 2.0}`;
- one residual/gate capacity choice from `{small, medium}`.

No combinatorial grid over all factors is allowed. One final configuration
must be selected on validation only before confirmatory testing. The selection
score must include validation proxy improvement, clean retention, factor
predictability, mechanism activation locality, action validity, and compute
overhead.

## Current Status

No AFID implementation, training, validation search, rollout, simulator
access, or confirmatory-test tuning has happened before this audit.

Immediate next stage: preregistration before prototype protocol,
implementation, validation search, training, or rollout.
