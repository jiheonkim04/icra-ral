# Phase-Locked Retiming Kill Summary

## Original Phase-Locked Retiming Hypothesis

Event-locked retiming should recover replay/control success when action chunks are temporally out of phase. The intended method would align gripper, translation, rotation, and object-interaction timing to task events such as approach, gripper close, object motion, lift, and place/contact.

## Strongest Positive Evidence

- Exact-init expert replay succeeded with reward/success/done `1.0 / true / 260`.
- All nine temporal phase perturbations degraded replay.
- The runner produced a real bounded LIBERO/RoboSuite replay/control table with `82` variants and `22248` simulator steps.
- The perturbation suite covered gripper close delay/advance, lift delay/advance, forward/backward chunk shift, time stretch/compression, and chunk-boundary offset.

## Decisive Negative Evidence

- Event-locked retiming recovered over raw perturbed replay on `0 / 9` perturbations.
- Event-locked retiming beat the best simple baseline on `0 / 9` perturbations.
- Simple baselines matched or beat Event-Locked Retiming on `3 / 9` perturbations.
- Gripper-only timing correction recovered both gripper timing perturbations.
- Fixed time shift recovered chunk-shifted-backward.
- Linear time warp recovered time-compression.

## Exact Kill Criterion Triggered

Kill criterion: Event-Locked Retiming must recover reward, success, done index, or meaningful progress over raw perturbed replay and must beat fixed shift, gripper-only correction, linear warp, global scale, diagonal affine, and nearest-progress demo on at least one meaningful replay/control metric.

Triggered because Event-Locked Retiming improved neither raw replay/progress nor best-simple-baseline performance.

## Which Simple Baselines Killed It

- `gripper_only_timing_correction` killed the gripper phase novelty.
- `fixed_time_shift` killed the global chunk-shift recovery novelty.
- `linear_time_warp` killed the time-compression recovery novelty.
- `repeat_last_hold` and `diagonal_affine` matched or weakened Event-Locked Retiming on other perturbation families.
- `nearest_progress_demo` remained a required strong baseline even though it did not win in this first task.

## Reusable Artifacts

- `tca_map.phase_locked.retiming`
- `scripts\180_phase_locked_retiming_diagnostic.ps1`
- phase perturbation generator,
- event-anchor extraction,
- exact-init replay/control harness,
- per-perturbation baseline table,
- event timing, gripper timing, trajectory drift, EEF-object distance, object movement, controller-valid action, and clip-rate metrics,
- result report: `reports\phase_locked_retiming_state1_result.md`.

## Why Not Continue As RA-L-Stable

The route found a real failure mode, but the proposed method did not recover from it. Worse, separate obvious simple baselines solved or matched the relevant sub-failures. A route whose targeted failure families are each handled by separate trivial heuristics does not have sufficient method novelty for RA-L-stable continuation.
