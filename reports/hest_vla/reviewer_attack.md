# HEST-VLA Reviewer B Attack

Date: 2026-07-15 KST

Proposal hash:
`E56B4717BDF949E1A4371457058DFC662E0D79C70D9E2FBEF35A5415FD0F0527`.

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

## Strongest Fair Interpretation

HEST is not presented as a learned action code or as a generic smoothing trick.
Its strongest fair interpretation is an analytic hybrid action interface: six
continuous controller-coordinate increments are represented by an
endpoint-constrained cumulative spline, while the gripper command remains an
exact discrete event stream and any invalid transformation falls back to Base.

The narrowest publishable claim is that this factorization improves matched
closed-loop success and controller fidelity over Base, a transparent
all-channel spline proxy, an endpoint-free ablation, and an ordinary arm
smoother.

## Major Attack 1: Novelty May Collapse To Filtering

The method solves a quadratic smoothing problem and blends the result with the
original action. A reviewer can reasonably call this a low-pass filter with
boundary conditions. Preserving the gripper channel may be sound engineering
rather than a research contribution.

Required answer:

- MovingAverage must enter the first serious comparison;
- HEST must beat it on closed-loop success, not merely jerk;
- NoEndpoint must isolate the endpoint constraint;
- SplineProxy must isolate the hybrid event factorization;
- exact or near-exact equivalence is a fatal design result.

## Major Attack 2: The Closest-Prior Proxy Is Incomplete

Spline Policy trains a policy to predict spline parameters and also develops a
state-dependent flow-field construction. A post-hoc projection of a pretrained
chunk is not an official reproduction. If the paper is described as the prior
method itself, the comparison is misleading.

Required answer:

- always label the comparator `transparent analytic SplineProxy`;
- report the missing trained spline head and flow-field differences;
- use the paper only as a positive mechanism prior;
- do not claim superiority over official Spline Policy unless an official
  matched implementation later becomes available.

## Major Attack 3: Six Controller Coordinates Are Not A Euclidean Pose

Elementwise cumulative sums of rotation increments do not generally form a
valid `SE(3)` trajectory. A spline over Euler-like or axis-angle controller
coordinates can create geometrically misleading interpolation.

Required answer:

- call `P` a cumulative controller-coordinate path, not an end-effector pose;
- make no `SE(3)` or task-level stability claim;
- require finite, support, and exact-state simulator replay checks;
- report translation and rotation effects separately;
- reject HEST if rotation interpolation is invalid or dominates replay error.

## Major Attack 4: Endpoint Equality Is Not Effect Equality

Two chunks with the same cumulative arm endpoint can produce different contact,
collision, grasp, and object trajectories. Exact endpoint preservation is not a
physical guarantee.

Required answer:

- direct exact-state replay must measure robot and object-state trajectories;
- no offline endpoint or jerk metric may authorize confirmatory rollout alone;
- Stage 0B must establish a useful fidelity/smoothness tradeoff;
- contact-sensitive failures must be reported, not averaged away.

## Major Attack 5: Demonstrations May Have No Headroom

Expert LIBERO actions may already be smooth. SmolVLA errors may be semantic or
geometric rather than trajectory-frequency errors. The wrapper could act
everywhere and erase useful corrective motion.

Required answer:

- quantify Base arm second-difference energy and task/phase variability;
- require nontrivial smoothing headroom before simulator work;
- measure action delta and fallback frequency;
- require clean closed-loop retention during validation;
- classify no useful tradeoff as `NO_HEADROOM`, not as evidence against all
  structured action representations.

## Major Attack 6: Gripper Preservation Could Be Trivial

If most windows contain no gripper transition, exact gripper copying does not
test the proposed hybrid mechanism. Conversely, if transformed arm motion
changes when contact occurs, unchanged gripper commands may still be mistimed
relative to the physical trajectory.

Required answer:

- require at least eight fixed validation windows with a gripper transition;
- report transition and nontransition windows separately;
- measure state/object fidelity around transition windows;
- do not infer physical event alignment from bitwise command equality alone.

## Major Attack 7: Search And Confirmatory Leakage

Three alpha values are legitimate validation search only if all outcomes are
saved and confirmatory identities remain sealed. Choosing tasks, phases,
windows, or smoothing strength after examining success would be test tuning.

Required answer:

- freeze deterministic window selection and split identities before execution;
- persist all three alpha results;
- use the stated validation score and smaller-alpha tie rule;
- hash the selected implementation before Stage A;
- never retune after confirmatory outcomes.

## Major Attack 8: Queue Integration Can Silently Change The Protocol

The wrapper must transform the postprocessed `50 x 7` chunk exactly once at
queue refill. Transforming actions one by one, replanning more frequently, or
operating before the official postprocessor would change action semantics and
confound the claim.

Required answer:

- test queue-refill count and queue length against Base;
- hash Base input and transformed output chunks;
- preserve action order and one environment step per queued action;
- fail on any hidden queue or postprocessor mismatch.

## Baseline Classification

Essential paper evidence:

- Base;
- transparent SplineProxy;
- HEST;
- NoEndpoint;
- MovingAverage.

Useful diagnostics:

- expert-chunk exact-state replay;
- per-dimension support and delta summaries;
- transition-window trajectory fidelity.

Optional supplementary evidence:

- alternative curvature coefficients;
- additional smoothing kernels;
- timing measurements outside resource-contention intervals.

Irrelevant to the current mechanism:

- standard LoRA;
- QLoRA rank sweeps;
- a generic confidence head;
- adaptive action-chunk length.

## False-Positive And False-Negative Calibration

False-positive risk: high if HEST only lowers jerk, if the closest-prior proxy is
misrepresented, or if a moving average explains the result.

False-negative risk: moderate because a bounded post-hoc wrapper is weaker than
a policy trained to emit splines, but this does not justify changing the frozen
method after test.

Confidence before data: moderate.

Minimum evidence for permanent scientific kill: valid implementation and data,
acting bounded mechanism, complete matched Stage B or valid catastrophic Stage
A, and clear failure against Base, prior proxy, ablation, or simple control.

No Stage 0 result is a scientific kill.
