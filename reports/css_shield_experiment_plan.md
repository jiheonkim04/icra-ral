# CSS-Shield Experiment Plan

## State 1: Minimal Rollout-First Safety Diagnostic

Run the smallest available LIBERO/RoboSuite diagnostic first. Do not start with offline-only metrics.

Required variants:

1. no shield,
2. action clipping only,
3. safety-only shield,
4. semantic target-only shield,
5. full counterfactual semantic safety shield.

Required metrics:

- wrong-target action rate,
- unsafe action rate,
- intervention rate,
- false positive intervention rate,
- false negative unsafe rate,
- action modification magnitude,
- target-directed movement score when available,
- collision or near-collision proxy when available,
- utility preservation,
- reward/success when available,
- runtime overhead.

## Continue Criteria

Continue only if the full shield reduces wrong-target or unsafe actions while utility degradation remains bounded and it beats clipping-only or safety-only.

## Kill Criteria

Kill quickly if no simulator/rollout metric can be produced, the shield only stops everything, clipping-only matches the full shield, or realistic failures cannot be generated.

