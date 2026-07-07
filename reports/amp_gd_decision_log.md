# AMP-GD Decision Log

## State 0 Initialization

Decision: initialize Active Micro-Probe Goal Disambiguation as the next rollout-first, baseline-first route.

Reason: the topic tournament selected AMP-GD because it tests active evidence gathering before target commitment and can fail quickly against no-probe, random-probe, nearest-target, and safety-only baselines.

Consequence: do not continue Target-Prior TCA-Map, CSS-Shield, or ExecSpec-Repair as the main route. The first executable result must be a rollout/control metric, not an offline-only proxy or broad planning package.

## State 1 Minimal Probe Diagnostic

Decision: continue to State 2 scale diagnostic, with the explicit limitation that the first metric is toy control evidence.

Reason: on 60 seeded point-world trials, AMP-GD reached wrong-target rate `0.0` and success `1.0`, while no-probe and safety-only had wrong-target rate `0.5`, random-probe had `0.466666667`, and nearest-target had `0.483333333`. AMP-GD probe cost was `0.12`, extra path length versus no-probe was `0.318929988`, and unsafe/collision rate was `0.0`.

Consequence: the route is not killed at State 1. State 2 should scale the diagnostic and start a narrow LIBERO/RoboSuite object-observable port. Do not make paper-grade claims; kill if simple baselines catch up or if no credible LIBERO/RoboSuite path remains.
