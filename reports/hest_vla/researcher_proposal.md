# HEST-VLA Researcher A Proposal

Date: 2026-07-15 KST

Method: `HEST-VLA`, Hybrid Event-Spline Trajectories for VLA policies.

Decision: `HEST_RESEARCHER_PROPOSAL_FROZEN_FOR_REVIEW`

## 1. Claim Boundary

HEST tests whether a hybrid continuous-discrete trajectory interface can
improve closed-loop manipulation while preserving the exact event semantics of
a pretrained 7D VLA action chunk.

The narrow paper claim is:

> A 7D manipulation action chunk should not be represented as seven
> homogeneous smooth coordinates. Modeling six arm coordinates as a bounded,
> endpoint-constrained spline while preserving the gripper as a discrete event
> stream can improve SmolVLA task success over Base, an all-channel spline
> proxy, an endpoint-free ablation, and a simple arm smoother.

This is not a claim that splines are new, that smoothing is always beneficial,
or that offline reconstruction predicts task success. Closed-loop task success
is mandatory.

## 2. Positive External Prior

Closest prior: Spline Policy: A Structured Representation for Robot Policies,
https://arxiv.org/abs/2606.07386.

Spline Policy replaces fixed-resolution action chunks with spline parameters,
supports flow-matching and VLA-style backbones, exposes temporal and geometric
structure, and demonstrates compatible simulated and real-robot execution.

The local prior arm is a transparent analytic proxy because no official public
code or checkpoint was verified. It applies the same endpoint-constrained
spline construction to all seven SmolVLA action coordinates. It must never be
labeled an official Spline Policy reproduction.

## 3. Scientific Method

### 3.1 Input And Output

At each ordinary SmolVLA queue refill, let the postprocessed Base chunk be:

`A = [A_arm, A_grip] in R^(50 x 7)`,

where:

- `A_arm in R^(50 x 6)` contains the controller-facing translation and
  rotation increments;
- `A_grip in R^50` contains the controller-facing gripper command.

HEST returns another `50 x 7` chunk. It does not change the observation,
instruction, task, reset identity, action count, queue length, postprocessor,
or environment step call.

### 3.2 Cumulative Arm Path

Define `P in R^(50 x 6)` by cumulative summation:

`P_i = sum_(j=0)^i A_arm,j`.

For each arm dimension independently, HEST solves:

`Q* = argmin_Q ||Q - P||_F^2 + lambda ||D2 Q||_F^2`

subject to:

`Q*_0 = P_0` and `Q*_49 = P_49`.

`D2 in R^(48 x 50)` is the second-difference matrix. The fixed curvature
coefficient is `lambda = 4.0`. It is not searched.

Decode spline arm increments:

`S_arm,0 = Q*_0`,

`S_arm,i = Q*_i - Q*_(i-1)` for `i > 0`.

The cumulative arm endpoint is therefore preserved analytically.

### 3.3 Bounded Blend

For one validation-selected `alpha in {0.25, 0.50, 1.00}`:

`H_arm = (1 - alpha) A_arm + alpha S_arm`.

The gripper stream is copied exactly:

`H_grip = A_grip`.

The HEST chunk is `H = [H_arm, H_grip]`.

### 3.4 Whole-Chunk Fallback

HEST returns Base exactly if any of the following holds:

- input shape is not exactly `50 x 7`;
- any input or output value is nonfinite;
- cumulative arm endpoint error exceeds `1e-8` in float64 reference code or
  `1e-6` in float32 runtime code;
- any gripper value differs bit-for-bit from Base;
- any transformed arm coordinate falls outside the discovery action support
  expanded by the frozen `1%` per-dimension range tolerance;
- the runtime implementation and disk-reloaded implementation disagree above
  `1e-7` maximum absolute difference.

Fallback is all-or-nothing. Per-coordinate clipping is forbidden.

## 4. Mechanism Chain

Problem condition:

`50 x 7` action chunks combine smooth arm motion and abrupt gripper events.

Intermediate failure mechanism:

a homogeneous chunk representation either leaves high-frequency arm variation
unstructured or shifts a discontinuous gripper command when it is regularized.

Action consequence:

arm jerk, cumulative endpoint drift, or gripper-event timing error changes the
physical approach, grasp, transport, or release sequence.

Closed-loop consequence:

small controller-facing errors compound into missed contacts, unstable grasps,
or premature/late release.

HEST intervention:

regularize the cumulative six-dimensional arm path, preserve its first point
and endpoint, and pass the gripper event stream unchanged.

Intended internal change:

lower second-difference energy in cumulative arm motion without changing the
cumulative endpoint or gripper event indices.

Intended action behavior:

bounded arm deltas with lower jerk and exact gripper commands.

Expected closed-loop result:

higher matched task success than Base and all controls, with no material clean
retention loss.

## 5. Scientific Method Versus Parameterization

### Scientific Method

The contribution is the hybrid action object:

- a continuous endpoint-constrained arm spline;
- a discrete exact gripper event stream;
- a whole-chunk identity fallback.

### Low-Compute Parameterization

The method is a deterministic analytic wrapper around frozen SmolVLA. It has no
trainable parameters, no LoRA, no QLoRA, no auxiliary model, and no extra model
checkpoint.

Standard LoRA is omitted because generic weight adaptation does not test the
hybrid representation claim. The all-channel spline, endpoint-free ablation,
and moving-average control isolate the relevant mechanism more directly.

## 6. Evidence Partitions

### Discovery

Tasks:

- `libero_spatial/task_3`;
- `libero_object/task_3`;
- `libero_goal/task_5`;
- `libero_10/task_5`.

Official HDF5 demonstrations `0..7` per task are discovery-only. Four
deterministic valid 50-step windows per demonstration are selected from the
lexicographically ordered feasible start frames. Discovery may define action
support and diagnose mechanism behavior.

### Validation

The same four tasks use demonstrations `8..9`, four deterministic windows per
demonstration, for direct exact-state replay and controller-fidelity evaluation.

Closed-loop validation reset identities are `20262101..20262110`, allocated by
the frozen balanced task schedule in the prototype protocol. They may select
one `alpha` from `{0.25, 0.50, 1.00}`.

### Confirmatory Test

Stage A reset identities are `20262111..20262120`. Stage B adds
`20262121..20262150` so the paired Stage B set has `40` identities per key
policy. No confirmatory reset identity, reward, success, done flag, or video may
be read before method, comparator, alpha, task schedule, thresholds, and
checkpoint-free implementation are frozen.

Confirmatory outcomes may not retune HEST. A redesign after test is a new
method cycle.

## 7. Pre-Experiment Headroom And Data Audit

Stage 0A is CPU-only and reads discovery/validation demonstration actions, not
confirmatory resets or task outcomes.

It must establish:

1. all source chunks have shape `50 x 7` and finite actions;
2. split overlap and duplicate window keys are zero;
3. discovery action support is noncollapsed in every arm dimension;
4. at least `8` validation chunks contain a gripper command transition;
5. HEST endpoint and gripper invariants pass exactly;
6. HEST acts on at least `80%` of validation chunks at `alpha = 1`;
7. median cumulative-arm second-difference energy falls by at least `10%` at
   `alpha = 1`;
8. HEST and all controls preserve action validity without clipping;
9. HEST is not exactly equivalent to the all-channel proxy, endpoint-free
   ablation, or moving-average control;
10. no timing, latency, or resource metric is used as paper evidence.

Failure is classified as `DATA_FAILURE`, `NO_HEADROOM`,
`IMPLEMENTATION_FAILURE`, or `DESIGN_FAILURE`, not a closed-loop scientific
kill.

Stage 0B is allowed only after Stage 0A passes. It replays the fixed validation
windows from exact simulator states and measures terminal controller-state and
object-state deviation relative to the original expert chunk. It must show that
HEST offers a nontrivial jerk/fidelity tradeoff and is not explained by the
all-channel proxy or moving-average control before closed-loop validation.

The original expert chunk is a diagnostic reference only, never an inference
method.

## 8. Bounded Validation Search

The entire search is exactly three configurations:

- `alpha = 0.25`;
- `alpha = 0.50`;
- `alpha = 1.00`.

`lambda = 4.0`, tasks, windows, reset identities, support tolerance, and all
other choices are fixed. No architecture or seed search exists.

The preregistered validation score is:

`0.60 * paired_validation_success_delta`

`+ 0.15 * controller_fidelity_score`

`+ 0.10 * normalized_jerk_reduction`

`+ 0.10 * clean_retention_score`

`+ 0.05 * action_validity_score`.

Ties within `1e-12` choose the smaller alpha. After selection the alpha and
implementation hash are frozen. All three outcomes are retained.

## 9. First Serious Policy Comparison

The first comparison contains exactly five policies:

1. `Base`: frozen SmolVLA, unchanged `50 x 7` queue;
2. `SplineProxy`: transparent all-seven-coordinate endpoint-constrained spline;
3. `HEST`: selected hybrid arm-spline/event-stream method;
4. `NoEndpoint`: arm regularization without the cumulative endpoint constraint,
   exact gripper;
5. `MovingAverage`: fixed three-tap arm moving average, exact gripper.

Baseline rationale:

| Comparison | Scientific question |
| --- | --- |
| Base vs HEST | Does the hybrid action interface improve frozen SmolVLA? |
| SplineProxy vs HEST | Does continuous/discrete factorization improve on the closest spline prior mechanism? |
| NoEndpoint vs HEST | Is exact cumulative endpoint preservation necessary? |
| MovingAverage vs HEST | Is any gain explained by ordinary arm smoothing? |

No baseline is included merely by template.

## 10. Stage A And Stage B

Stage A uses approximately `10` paired episodes per policy under the fixed task
and reset manifest. It may permanently stop only for mechanism invalidity, no
headroom, catastrophic degradation, exact equivalence, or clear dominance by
the closest prior, ablation, or simple control. Small or unresolved differences
advance.

Stage B uses at least `40` paired episodes per key policy. Report:

- success count and rate;
- paired wins, losses, and ties;
- paired bootstrap confidence interval;
- absolute effect and relative failure-rate reduction;
- task breakdown;
- intervention/fallback frequency;
- endpoint and gripper invariants;
- clean retention;
- action validity.

One expansion to `80` is allowed only if the frozen Stage B rule classifies the
comparison as genuinely unresolved. No second expansion is allowed.

## 11. Paper-Candidate Gate

HEST becomes a prototype GO only if:

- HEST beats Base on matched closed-loop task success;
- HEST beats SplineProxy;
- HEST beats NoEndpoint;
- MovingAverage does not explain the gain;
- clean behavior is retained;
- fallback is not so frequent that HEST is effectively Base;
- endpoint and gripper invariants hold during real rollout;
- the hybrid representation remains novel after a final literature audit.

After GO, continue immediately to larger SmolVLA confirmation, Quantized
OpenVLA-OFT INT4 versus Quantized OpenVLA-OFT INT4 plus HEST, one
claim-specific second condition, recent baselines, and figure/table-ready
evidence.

## 12. Kill And Failure Taxonomy

- `FATAL_PREIMPLEMENTATION`: near-exact duplication, mathematical invalidity,
  exact trivial equivalence, unavailable essential resource, or non-falsifiable
  mechanism.
- `DATA_FAILURE`: malformed/collapsed actions, inadequate event coverage, split
  overlap, duplicate keys, or invalid source mapping.
- `IMPLEMENTATION_FAILURE`: invariant, reload, queue, action validity, or
  simulator integration defect.
- `NO_HEADROOM`: valid direct development evidence shows no useful
  jerk/fidelity or closed-loop opportunity.
- `DESIGN_FAILURE`: the valid hybrid representation cannot produce the intended
  action effect or is explained exactly by a simple control.
- `LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT`: not applicable unless a later
  backbone integration demonstrates a representation bottleneck without
  changing the scientific method.
- `PROTOTYPE_GO`: all paper-candidate gates pass.
- `GENUINE_METHOD_KILL`: valid data and implementation, acting mechanism, and a
  complete matched confirmatory result clearly fail against Base, prior,
  ablation, or simple control.
- `SIMPLE_BASELINE_EXPLAINS_METHOD`: MovingAverage matches or beats HEST under
  the frozen decision rule.
- `KEY_COMPONENT_NOT_USEFUL`: SplineProxy or NoEndpoint matches or beats HEST
  under the frozen decision rule.

No Stage 0 result is a closed-loop scientific kill.
