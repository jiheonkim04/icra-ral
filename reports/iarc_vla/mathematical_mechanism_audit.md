# IARC-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Decision: `IARC_MATHEMATICAL_AUDIT_PREREGISTERED`

Proposal hash:
`A1B0CF8BCBCF6A88F27B31EF5E38BAF408A3E62BB34206A1AC9F051EA6B57408`.

Authoritative constraints: the frozen proposal, Reviewer B attack, and
Researcher A rebuttal. Where the proposal's raw-gradient-plus-AdamW statement
conflicts with the rebuttal, this audit freezes the repaired Stage II projected
SGD rule.

## 1. Variables, Shapes, Units, And Sources

For one logical batch:

| Symbol | Meaning | Shape | Dtype for audit | Units/source |
| --- | --- | --- | --- | --- |
| `o_c^k` | clean raw RGB camera `k` | `[1, 3, H_k, W_k]` | float32 | raw `[0,1]`, official dataset |
| `o_r^k` | allowlisted perturbed RGB camera `k` | `[1, 3, H_k, W_k]` | float32 | raw `[0,1]` |
| `s` | clean proprioceptive state | official preprocessor shape, measured | policy dtype | official dataset units |
| `l_c` | original instruction | one string | UTF-8 | official task text |
| `l_r` | allowlisted perturbed instruction | one string | UTF-8 | exact repetition/wrapper contract |
| `a*` | demonstration action chunk before native padding | `[1, 50, 7]` | float32 | official normalized action units |
| `z` | shared flow noise | `[1, 50, 32]` | model forward dtype; float32 hash source | native SmolVLA action latent |
| `t` | shared flow time | `[1]` | float32 | unit interval |
| `L_c`, `L_r` | scalar SmolVLA action-flow loss | `[]` | float32 reduction | dimensionless training loss |
| `theta_j` | trainable rank-4 LoRA tensor `j` | runtime-resolved | model dtype | adapter parameter |
| `g_c^j`, `g_r^j` | accumulated parameter gradients | shape of `theta_j` | float32 | loss per parameter unit |
| `g_c`, `g_r` | flattened gradients in frozen order | `[P]` | float32 | concatenated adapter gradient |
| `d` | dot product `<g_c,g_r>` | `[]` | float32 accumulation | squared gradient units |
| `r` | robust squared norm `||g_r||^2` | `[]` | float32 accumulation | squared gradient units |
| `g_IARC` | projected update gradient | `[P]` | float32 | adapter gradient |
| `eta` | Stage II learning rate | `[]` | float | parameter-step scale |
| `Delta theta` | actual SGD parameter delta | `[P]` | parameter dtype after float32 calculation | adapter parameter units |

`H_k`, `W_k`, processed image shapes, token shapes, state shape, action target
shape after native padding, `P`, target-module names, and trainable parameter
count are runtime facts. Stage 0 must record them before any update. A mismatch
from `[1,50,7]` demonstration chunks or `[1,50,32]` native flow noise is an
implementation failure, not an opportunity to rewrite the method.

No future observation, simulator state, privileged label, test identity, or
future action beyond the normal demonstration action chunk is an inference
input. Training perturbations alter only current images or current instruction.

## 2. Clean And Robust Action Objectives

Let `F(theta; o,s,l,a*,z,t)` be the scalar native SmolVLA flow-matching action
loss returned by the official policy forward path. Define

`L_c(theta) = F(theta; o_c,s,l_c,a*,z,t)`

`L_r(theta) = F(theta; o_r,s,l_r,a*,z,t)`.

For every pair, `a*`, `s`, `z`, and `t` are byte-identical. Exactly one
allowlisted modality transform differs. Clean and perturbed calls use the same
loss reduction and autocast scope.

These are the only scientific objective terms. There is no KL, JS,
Wasserstein, MMD, Mahalanobis, L2 action-retention penalty, consistency head,
gate loss, or auxiliary representation loss.

Gradient paths:

- `L_c -> official SmolVLA forward -> trainable LoRA tensors`;
- `L_r -> official SmolVLA forward -> the same trainable LoRA tensors`;
- Base parameters are frozen and must receive no update;
- raw images, text tokens, action targets, noise, and time are not optimized.

## 3. Frozen Gradient Vectorization

At policy construction, collect all trainable parameters sorted by full name.
Freeze a manifest containing each name, shape, numel, dtype, and target module.

For each objective:

1. zero all parameter gradients;
2. run the shared-draw forward/backward;
3. unscale gradients if any mixed-precision scaler is active;
4. copy each gradient to float32;
5. replace `None` with an exact float32 zero tensor of the parameter shape;
6. verify finite values;
7. flatten in manifest order;
8. concatenate to `[P]`.

Stage 0 and full Stage II use physical and logical batch size `1`; no gradient
accumulation is used. This removes an unnecessary ambiguity. Projection occurs
once for each optimizer step.

Flatten/unflatten must be exact by shape and name. A reconstructed synthetic
vector must match bitwise in float32. Gradient clipping is forbidden.

## 4. IARC Update

Let

`d = sum_i g_c[i] * g_r[i]`

`r = sum_i g_r[i]^2`.

The frozen robust squared-norm floor is

`nu = 1e-12`.

The update is

`g_IARC = g_c`, if `d >= 0` and `r >= nu`;

`g_IARC = g_c - (d/r) g_r`, if `d < 0` and `r >= nu`.

If `r < nu`, no valid robust reference direction exists. Do not update under
the name IARC for that pair; record `robust_gradient_below_floor` and apply the
classification rules.

Stage II uses SGD with momentum `0`, weight decay `0`, and no clipping:

`Delta theta = -eta * g_IARC`.

For a conflict row with `r >= nu`:

`<g_r,g_IARC> = d - (d/r) r = 0`.

Therefore the first-order robust-loss change is

`Delta L_r = <g_r,Delta theta> + O(eta^2) = O(eta^2)`.

This is a local first-order statement. It is not a finite-step monotonicity,
trajectory-success, or general robustness theorem.

Numerical constraint tolerance:

`<g_r,g_IARC> >= -1e-6 * max(1, ||g_r|| ||g_IARC||)`.

The projection coefficient `-d/r` must be finite and nonnegative on conflict
rows. On agreeing rows the coefficient is exactly zero and `g_IARC` must equal
`g_c` bitwise before parameter casting.

## 5. Comparator Updates

### Transparent STRONG Proxy

Stage I matches the frozen perturbation curriculum. Stage II uses

`g_prior = g_c`.

It may evaluate `g_r` and discard it for compute diagnostics. The robust
gradient cannot affect parameters or optimizer state.

### Unprojected Joint-Replay Ablation

Discovery computes

`q = median(||g_c||) / median(||g_r||)`

over healthy fixed audit pairs.

Freeze

`beta = 1`, if `q in [0.25,4.0]`;

otherwise freeze `beta = q`.

`beta` is a discovery-only scale correction, not a validation configuration.
The ablation uses

`g_joint = (g_c + beta g_r) / 2`.

IARC may use `beta g_r` in its implementation because the exact projection is
invariant to positive scalar rescaling of the reference gradient. The robust
norm floor is always checked on unscaled `g_r`.

Required scale audit:

- clean and robust loss median, IQR, p05, p95;
- clean and robust gradient norm median, IQR, p05, p95;
- `q` and frozen `beta`;
- per-module norm contributions;
- fraction of rows in which either gradient is below its floor.

### Standard LoRA

Standard LoRA uses only clean action loss:

`g_standard = g_c`.

It receives the same rank, target modules, demonstrations, total steps, Stage I
AdamW schedule length, Stage II SGD schedule length, and validation rule. It
does not receive a robustness projection. The joint-replay ablation is the
direct control for extra perturbed data.

## 6. Perturbation Operators

All visual inputs are raw float RGB in `[0,1]` before the official processor.

### Gaussian Sensor Noise

For camera `k`:

`o_r^k = clip(o_c^k + sigma xi_k, 0, 1)`,

where `xi_k` is deterministic seeded standard normal noise and
`sigma in {0.02,0.05,0.10}`.

### Image Translation

Translate both streams in one deterministic cardinal direction by
`p in {4,8,16}` pixels using edge-replication padding. Shape and raw range are
preserved.

### Instruction Repetition

For severity `m in {1,2,3}`, append `m` additional exact copies of the original
instruction separated by ` ; `.

### Context Wrapper

For severity `m in {1,2,3}`, prepend `m` exact copies of
`Context note: the workspace contains several objects. Task:` separated by one
space, followed by the exact original instruction once.

The visual and text families have no common numeric unit; severity is an ordinal
family-local level only. No arithmetic comparison across families is made.

## 7. Frozen Training Schedule

### Stage 0 Micro Stage I

- optimizer: AdamW;
- learning rate: `1e-4`;
- weight decay: official wrapper default recorded at runtime, with no search;
- steps: `20`;
- seed: `1601`;
- batch size: `1`;
- rank: `4`.

### Full Stage I

- optimizer: AdamW;
- learning rate: `1e-4`;
- steps: `60`;
- seeds: `1601`, `1602`;
- batch size: `1`;
- steps `0-29`: text families alternating;
- steps `30-59`: visual families alternating;
- within each 30-step modality block, severity phases are `0-9`, `10-19`, and
  `20-29` for low, middle, and high severity;
- perturbation probabilities by phase: `0.25`, `0.50`, `0.75`;
- nonperturbed draws remain clean with the same action objective.

Task rows are deterministically balanced and permuted by seed. The schedule is
the transparent local STRONG proxy; it is not claimed to reproduce unpublished
code.

### Full Stage II

- optimizer: SGD;
- momentum: `0`;
- weight decay: `0`;
- steps: `40`;
- batch size: `1`;
- learning-rate candidates: `{5e-5,1e-4,2e-4}`;
- seeds: `{1601,1602}`;
- total trials: `6`.

Train two Stage I checkpoints, then branch each into the three Stage II learning
rates. Save every trial. Select learning rate by mean validation score across
both seeds. The designated final seed is always `1601`; never select the better
seed post hoc.

Standard LoRA receives `60` clean AdamW steps followed by `40` clean SGD steps.

## 8. Validation Score

Discovery freezes component scales before trial outcomes. Each validation
component is clipped to `[0,1]`:

- `R`: robust-loss improvement against the same trial's Stage I checkpoint,
  scaled by the discovery Base perturbation-loss gap;
- `C`: clean retention, `1 - normalized clean-loss degradation versus Base`;
- `S`: fraction of conflict rows satisfying the projection tolerance;
- `A`: finite in-range postprocessed action fraction;
- `M`: conflict-row activation fraction scaled to `1` at `0.25`.

Validation score:

`V = 0.40 R + 0.25 C + 0.15 S + 0.10 A + 0.10 M`.

Hard eligibility before score:

- `A = 1.0`;
- checkpoint reload passes;
- test decode count `0`;
- no target-changing transform;
- finite gradients and outputs;
- clean success proxy degradation does not exceed the frozen retention limit.

Closed-loop validation may replace `R` and `C` only if its manifest and scales
are frozen before all six trials. Offline action L2 alone cannot select.

## 9. Stage 0 Mechanism Tests

Pure tensor cases:

1. agreeing: `g_c=[1,0]`, `g_r=[1,0]` -> unchanged;
2. conflicting: `g_c=[-1,0]`, `g_r=[1,0]` -> zero;
3. partly conflicting: `g_c=[-1,1]`, `g_r=[1,0]` -> `[0,1]`;
4. orthogonal: `g_c=[0,1]`, `g_r=[1,0]` -> unchanged;
5. below-floor robust gradient -> explicit below-floor status;
6. nonfinite input -> exception and no update;
7. flatten/unflatten -> exact round trip.

Real-batch requirements:

- `40` independent clean/perturbed gradient pairs;
- shared flow noise/time hash equality `40 / 40`;
- action target/state/nonperturbed input hash equality `40 / 40`;
- conflict count at least `4 / 40` for direct pass;
- at least two families with conflict;
- projection tolerance pass on every conflict row;
- no projection on agreeing rows;
- finite difference or realized tiny-step robust loss does not increase beyond
  numerical tolerance on at least `90%` of conflict rows at a frozen diagnostic
  step size `1e-6`;
- IARC differs from clean and joint gradients on all conflict rows.

The tiny-step functional check uses a temporary parameter snapshot and restores
it exactly; it is diagnostic and does not train the checkpoint.

## 10. Identity, Capacity, And Action Checks

Before training:

- zero-effect LoRA action equals Base with maximum absolute error `<= 1e-6` on
  native and postprocessed actions using shared noise;
- only LoRA tensors require gradients;
- Base parameter hash is unchanged;
- rank-4 adapter receives finite nonzero gradients;
- a fixed `8`-row subset can reduce action loss after the `20` micro steps;
- checkpoint persists and reload output error is `<= 1e-6`;
- all postprocessed actions are finite and within dataset action bounds;
- translation, rotation, and gripper deltas are separately reported;
- no privileged inference field is present.

Failure of subset fit or nonzero gradients is
`LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT` or implementation failure, not a
scientific method kill.

## 11. Development Closed-Loop Headroom

Primary screen:

- ten task/reset pairs frozen in the rebuttal;
- clean and perturbed Base condition;
- `20` synchronous episodes;
- reset `20261601`;
- no timeout, exception, duplicate, action modification, or off-manifest key.

Headroom passes when clean-minus-perturbed success is at least `0.10` or at
least two clean successes become perturbed failures.

If unresolved, one fixed expansion is allowed:

- same ten tasks;
- reset `20261602`;
- another `20` episodes;
- aggregate pass when clean-minus-perturbed success is at least `0.10` or at
  least four clean successes become perturbed failures across `20` pairs.

No second expansion. Decisive equality with a narrow paired bootstrap interval
excluding `0.10` is `NO_HEADROOM`. Wide/mixed evidence is
`UNDERPOWERED_OR_UNRESOLVED` and cannot be mislabeled a scientific kill.

## 12. Required Ablation And Simpler Alternative

Required key ablation: unprojected, norm-audited joint replay under the same
Stage II batches and SGD optimizer.

Strongest simple alternative: standard clean-only LoRA under matched capacity,
data source, steps, optimizer schedule, and selection rule.

Closest prior: transparent STRONG curriculum plus clean refinement.

These three comparisons answer distinct questions. No extra projection method,
rank, gate, head, memory, or divergence is authorized.

## 13. Resource Evidence

`reports/resource_contention_intervals.json` is an exclusion registry. Timing,
throughput, wall-clock, and resource-utilization evidence with unknown or
positive overlap is diagnostic only and cannot enter final paper claims.
Task-success and action rows require synchronous execution, zero exceptions,
unchanged action semantics, frozen identities, and zero duplicate/off-manifest
keys.

## Audit Decision

`IARC_MATHEMATICAL_AUDIT_PREREGISTERED`.

The method now has one exact constrained Stage II update, one primary action
objective evaluated on clean and perturbed inputs, one key joint-replay
ablation, one standard-LoRA simple alternative, explicit tensor and gradient
contracts, bounded validation search, and no decorative mathematics. Proceed to
preregistration and prototype protocol only.

