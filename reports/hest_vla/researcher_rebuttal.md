# HEST-VLA Researcher A Rebuttal

Date: 2026-07-15 KST

Proposal hash:
`E56B4717BDF949E1A4371457058DFC662E0D79C70D9E2FBEF35A5415FD0F0527`.

Decision: `HEST_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

## 1. Novelty Is Conditional On Baseline-Resistant Utility

Researcher A accepts that the quadratic solve alone is a filter. The method
claim therefore depends on the complete hybrid construction and its
closed-loop consequence:

- cumulative arm coordinates rather than raw independent action samples;
- exact first-point and endpoint constraints;
- exact gripper-event passthrough;
- whole-chunk Base fallback;
- a matched win over ordinary moving-average smoothing.

If MovingAverage matches or beats HEST, classify
`SIMPLE_BASELINE_EXPLAINS_METHOD`. If NoEndpoint matches or beats HEST,
classify `KEY_COMPONENT_NOT_USEFUL`.

## 2. Prior Status Is Narrowed

The comparator is named `SplineProxy` in every artifact. It is a transparent
analytic approximation of the structured-output idea, not an official
Spline Policy implementation. The proposal does not claim to reproduce the
prior's learned spline head, state-dependent flow field, or published robot
system.

The paper claim may be "improves on a matched analytic spline proxy" unless a
future official matched implementation is available.

## 3. Controller Coordinates, Not SE(3)

`P` is explicitly a cumulative controller-coordinate path. HEST does not claim
that elementwise rotation integration is a group-correct pose trajectory.
Translation and rotation deltas, support, and simulator trajectory effects are
reported separately. A rotation-validity or replay-fidelity failure blocks the
method before confirmatory rollout.

## 4. Direct Replay Is Mandatory

Endpoint equality is only an algebraic invariant. Stage 0B must replay all
fixed validation windows from exact simulator states and compare robot and
object-state trajectories to the original expert action chunk. HEST cannot
advance on jerk or endpoint metrics alone.

The expert chunk remains a diagnostic reference and is never available to the
deployed method.

## 5. Headroom And Acting Gates

Stage 0A requires at least `10%` median cumulative-arm second-difference energy
reduction, acting on at least `80%` of validation chunks at `alpha = 1`, full
action validity, and non-equivalence to every control. Stage 0B then requires a
nontrivial replay fidelity/smoothness tradeoff and clean validation retention.

Failure is reported honestly as data, implementation, no-headroom, or design
failure. No task, window, threshold, or coefficient is changed to rescue HEST.

## 6. Event Coverage And Physical Alignment

Bitwise gripper equality is necessary but not sufficient. At least eight
validation chunks must contain a transition. Replay results are stratified by
transition status and include a local trajectory window around each event. If
arm smoothing changes the physical approach enough to invalidate event timing,
the exact command invariant does not excuse the failure.

## 7. Search And Leakage Controls

Only `alpha` is searched, over exactly three values. Discovery/validation
windows and reset identities are frozen. All configurations and negative
results are retained. Confirmatory reset identities, reward, success, done,
and videos remain unread until alpha and implementation hash are frozen.

## 8. Queue Identity

HEST wraps the already postprocessed `50 x 7` chunk once per ordinary queue
refill. It does not call the VLA more often or alter queue order. Unit and live
smoke tests record queue lengths, input/output hashes, first queued action, and
refill counts. Any mismatch is `IMPLEMENTATION_FAILURE`.

## Rebuttal Conclusion

The proposal survives review only as a narrow empirical hypothesis. Its
mathematical invariants are testable, its first decisive checks are bounded,
and its strongest simple explanation is live from the beginning. Proceed to a
formal mathematical and execution audit without changing the frozen proposal.
