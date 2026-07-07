# ResetSpec-Retarget Kill Summary

## Original ResetSpec Hypothesis

Object-relative and EEF-relative retargeting should recover replay success under initial-state and object-pose mismatch better than raw replay, diagonal affine calibration, global scaling, clipping, and nearest-demo replay.

## Strongest Positive Evidence

- Exact-init expert replay succeeded with reward/success `1.0 / true`, first done `260`.
- Default-reset raw replay failed with reward/success `0.0 / false`.
- Object poses and EEF poses were available from non-leaking observation keys.
- Object-relative translation retargeting improved EEF-object progress and object movement.
- Object-relative retargeting reduced shifted-trajectory drift to about `0.002` mean L2, better than raw default replay.

## Decisive Negative Evidence

- Object-relative translation retargeting failed reward/success: `0.0 / false`.
- Object-relative translation plus gripper-phase retargeting also failed reward/success: `0.0 / false`.
- Fixed global-scale replay from default reset succeeded with reward/success `1.0 / true`, first done `257`.

## Exact Kill Criterion Triggered

Kill criterion: object-relative retargeting must beat diagonal-affine, global-scale, clipping, and feasible nearest-demo baselines on success, reward, done index, or meaningful progress.

Triggered because fixed global scale beat object-relative retargeting on the primary replay metrics.

## Why Global Scale Kills The Novelty

The target failure mode was supposed to require state-dependent object/EEF retargeting. If a task-agnostic fixed action scaling recovers default-reset success while object-relative retargeting does not, the observed gap is not strong evidence for object-relative executable retargeting. The result is better explained as an action magnitude or controller interaction issue than as a novel state-dependent retargeting contribution.

## Reusable Infrastructure

- `tca_map.resetspec.retarget`
- `scripts\170_resetspec_retarget_diagnostic.ps1`
- exact-init versus default-reset replay comparison,
- object/EEF pose capture from simulator observations,
- object-shifted EEF trajectory drift metric,
- translation, rotation, gripper timing, clip, and controller-valid action metrics,
- non-leaking instruction-text plus visible-object-key target resolver,
- concise result report: `reports\resetspec_state1_result.md`.

## Why Not Continue As RA-L-Stable

ResetSpec-Retarget found a real reset-mismatch gap, but it did not show method novelty beyond a trivial global-scale baseline. Continuing would risk building a paper around a failure mode already explained by action-only scaling. The route should be archived as useful infrastructure, not scaled as the main RA-L route.
