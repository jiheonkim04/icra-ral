# CSS-Shield Kill Summary

## Original Hypothesis

CSS-Shield could provide a lightweight inference-time counterfactual semantic/safety layer that reduces wrong-target and unsafe VLA actions while preserving task utility.

## Strongest Positive Evidence

- State 1.5 and State 2 diagnostics showed wrong-target metric computability under controlled proposal settings.
- State 4 scaled randomized diagnostic reported full vs safety-only wrong-target delta `0.58`.
- False positive rate in the controlled diagnostic package was `0.0`.
- Native SmolVLA CPU inference and LIBERO/RoboSuite rollout plumbing worked.

## Decisive Negative Evidence

Phase 2 native-action diagnostic hit the kill/reframe gate:

- full vs safety-only wrong-target delta: `0.0`.
- full vs clipping-only wrong-target delta: `0.0`.
- full vs clipping-only unsafe delta: `0.85`.
- full intervention rate: `1.0`.
- reward/success under full shield: `0.0 / false`.

## Kill Criterion Triggered

Full CSS-Shield failed to beat safety-only on native-action wrong-target reduction and behaved like a full-intervention shield.

## Reusable Artifacts

- CSS-Shield variants: no shield, clipping-only, safety-only, semantic-only, full shield.
- Bounded WSL LIBERO/RoboSuite diagnostic wrappers.
- Native SmolVLA CPU inference path.
- Object inventory and wrong-target metric tools.
- Phase 2 state machine and kill/reframe package.

## Why It Should Not Continue As RA-L-Stable

The native-action evidence supports unsafe-action damping against clipping-only, not a semantic shielding contribution beyond safety-only. That is insufficient for a main RA-L route.

