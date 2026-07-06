# ExecSpec-Repair Experiment Plan

## STATE 0: Initialize

Create concise project docs, branch state, kill criteria, and a first diagnostic script. Do not spend another branch on report-only scaffolding.

## STATE 1: Reproduce Executable Mismatch

Use local LIBERO HDF5 expert actions first. If safe, add exact-init replay in a later bounded state.

Perturbations:

1. correct 7D expert actions,
2. global action scale mismatch,
3. per-dimension scale mismatch,
4. gripper sign flip,
5. gripper threshold/convention mismatch,
6. translation scale mismatch,
7. rotation scale mismatch,
8. clipping-only,
9. 6D/7D zero-gripper bridge stress.

Metrics:

- action L2 drift,
- translation drift,
- rotation drift,
- gripper mismatch,
- clip rate,
- controller-valid action rate,
- HDF5 done/reward index metadata when available,
- action-integral trajectory proxy,
- supervised calibration recovery versus identity, clipping-only, and naive global affine baselines.

Continue if at least one plausible mismatch causes substantial drift or supervised calibration recovers drift beyond simple baselines. Kill if local assets cannot reproduce mismatch.

## STATE 1 Result

STATE 1 produced both HDF5 action-drift evidence and bounded exact-init replay degradation on the first local LIBERO demo.

- strongest HDF5 mismatch: `gripper_sign_flip`, action L2 mean `2.0`, gripper mismatch rate `1.0`.
- other substantial HDF5 mismatches: global scale, per-dimension scale, gripper threshold 0/1, translation scale, and 6D/7D zero-gripper bridge.
- supervised diagonal calibration beat identity, clipping-only, and naive global affine baselines on seven mismatch variants.
- exact-init expert replay succeeded: reward `1.0`, success `true`, first reward/done index `260`.
- exact-init gripper-sign-flip replay failed: reward `0.0`, success `false`.
- exact-init translation-scale replay failed: reward `0.0`, success `false`.

Next state: replay a minimal calibrated repair for the strongest degraded mismatch under the same exact-init boundary.
