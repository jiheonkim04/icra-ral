# Simple Baseline Failure Patterns

Lesson: a method that only beats no-method is not enough. The route must beat the strongest trivial baseline for its claimed failure mode.

## Baseline-Kill Table

| Route | Claimed failure mode | Baseline that killed or weakened it | Pattern |
| --- | --- | --- | --- |
| Target-Prior TCA-Map | target-conditioned action grounding | mean-action | action decoder weaker than a trivial action prior |
| CSS-Shield | semantic wrong-target safety | safety-only | semantic component did not add value beyond generic safety intervention |
| ExecSpec-Repair | executable-spec mismatch | diagonal affine | broad repair collapsed to simple calibration |
| AMP-GD | active disambiguating probes | informative-probe heuristic, random-probe, safety-only | active probe logic did not beat simple probe/safety behavior |
| ResetSpec-Retarget | object-pose/reset retargeting | global scale | state-dependent retargeting lost to action-only scaling |

## Mandatory Early Baselines

Every new topic must predeclare and test the relevant subset of:
- mean-action,
- no-action/no-method,
- clipping-only,
- safety-only,
- nearest-target,
- random-probe,
- informative-probe heuristic,
- diagonal affine,
- global scale,
- nearest-demo,
- exact-init expert replay upper bound,
- oracle/replay-leakage upper bound clearly labeled as invalid method evidence.

## Anti-Pattern

Do not start from a clever method and add baselines later. Start from the strongest simple baseline, then ask whether a method can plausibly beat it within 48 to 72 hours.
