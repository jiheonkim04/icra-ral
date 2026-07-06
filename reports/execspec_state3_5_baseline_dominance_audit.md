# ExecSpec STATE 3.5 Baseline Dominance Audit

This is a report-only reframe audit over the existing STATE 3 replay results. It performs no new replay, training, downloads, GPU work, OpenVLA-OFT, or paper-grade claim.

- decision: `kill`
- reason: best single simple baseline matches full repair within 5 percentage points on success and action recovery
- degraded replay cases analyzed: `19`
- full repair success recovery: `0.894736842`
- best single simple baseline: `diagonal_affine_calibration`
- best single simple baseline success recovery: `0.894736842`
- full minus best single simple baseline: `0.0`
- best trivial baseline: `gripper_only_calibration`
- full minus best trivial baseline: `0.578947368`
- simple baselines explain result: `True`
- repair selector/routing meaningful: `False`
- next state: `archive_execspec_repair_or_select_new_rollout_first_route`

## Method Aggregates

| method | success recovered | success rate | reward rate | done rate | action recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity_no_repair | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| clipping_only | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| global_affine_calibration | 4 | 0.210526 | 0.210526 | 0.210526 | 0.361162 |
| gripper_only_calibration | 6 | 0.315789 | 0.315789 | 0.315789 | 0.315789 |
| diagonal_affine_calibration | 17 | 0.894737 | 0.894737 | 0.894737 | 1.0 |
| full_execspec_repair | 17 | 0.894737 | 0.894737 | 0.894737 | 1.0 |
| mismatch_aware_selector | 17 | 0.894737 | 0.894737 | 0.894737 | 1.0 |
| oracle_best_per_case | 17 | 0.894737 | 0.894737 | 0.894737 | 1.0 |

## Four Simple-Baseline Matched Cases

| demo | task | mismatch | matched methods | full reward/success | global reward/success | diagonal reward/success |
| --- | --- | --- | --- | --- | --- | --- |
| LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | global_action_scale_mismatch | global_affine_calibration | 1.0/true | 1.0/true | 1.0/true |
| LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | range_clipping_mismatch | global_affine_calibration | 1.0/true | 1.0/true | 1.0/true |
| LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | global_action_scale_mismatch | global_affine_calibration | 1.0/true | 1.0/true | 1.0/true |
| LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | range_clipping_mismatch | global_affine_calibration | 1.0/true | 1.0/true | 1.0/true |

## Per-Mismatch Recovery

| mismatch | degraded cases | best fixed repair | selector rule | sufficient repairs vs full | failing repairs |
| --- | ---: | --- | --- | --- | --- |
| global_action_scale_mismatch | 3 | global_affine_calibration | global_affine_calibration | global_affine_calibration, diagonal_affine_calibration, full_execspec_repair | identity_no_repair, clipping_only, gripper_only_calibration |
| gripper_sign_flip | 3 | gripper_only_calibration | gripper_only_calibration | gripper_only_calibration, diagonal_affine_calibration, full_execspec_repair | identity_no_repair, clipping_only, global_affine_calibration |
| gripper_threshold_0_1_mismatch | 3 | gripper_only_calibration | gripper_only_calibration | gripper_only_calibration, diagonal_affine_calibration, full_execspec_repair | identity_no_repair, clipping_only, global_affine_calibration |
| per_dimension_scale_mismatch | 3 | diagonal_affine_calibration | diagonal_affine_calibration | diagonal_affine_calibration, full_execspec_repair | identity_no_repair, clipping_only, global_affine_calibration, gripper_only_calibration |
| range_clipping_mismatch | 3 | global_affine_calibration | global_affine_calibration | global_affine_calibration, diagonal_affine_calibration, full_execspec_repair | identity_no_repair, clipping_only, gripper_only_calibration |
| rotation_scale_mismatch | 1 | diagonal_affine_calibration | diagonal_affine_calibration | diagonal_affine_calibration, full_execspec_repair | identity_no_repair, clipping_only, global_affine_calibration, gripper_only_calibration |
| translation_scale_mismatch | 3 | diagonal_affine_calibration | diagonal_affine_calibration | diagonal_affine_calibration, full_execspec_repair | identity_no_repair, clipping_only, global_affine_calibration, gripper_only_calibration |

## Interpretation

- Global affine explains the four simple-baseline matched replay cases, but it does not explain the full result.
- Gripper-only repairs the gripper convention cases but fails translation, per-dimension, global/range, and degraded rotation cases.
- Per-dimension diagonal affine matches full ExecSpec-Repair on both replay recovery and action recovery in this STATE 3 evidence.
- The mismatch-aware selector also matches full repair, but it does not beat diagonal affine; routing is therefore not enough to rescue the broad claim.