# ExecSpec STATE 3 Replay Validation

This is bounded diagnostic evidence only. It is not benchmark success or paper-grade evidence.

- decision: `kill_or_reframe`
- calibration demos/action samples: `5` / `1403`
- held-out eval demos/action samples: `3` / `805`
- task count: `8`
- eval leakage detected: `False`
- replay/rollout happened: `True`
- best repair method: `diagonal_affine_calibration`
- full beats identity/clipping/global: `True` / `True` / `True`
- full repair mean recovery fraction: `1.0`
- success recovery rate: `0.894736842`
- reward recovery rate: `0.894736842`
- next state: `kill/reframe`

## Split

- calibration paths: `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5; KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5; KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5; LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5; LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5`
- eval paths: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5; LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5; LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5`
- tasks: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it; KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it; KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it; KITCHEN_SCENE8_put_both_moka_pots_on_the_stove; LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket; LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket; LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket; LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`
- suite coverage: `{"libero_10": 8}`

## Held-Out Action Aggregate

| mismatch | wrong L2 | full L2 | full recovery | full beats id/clip/global |
| --- | ---: | ---: | ---: | --- |
| gripper_sign_flip | 2.0 | 0.0 | 1.0 | true/true/true |
| translation_scale_mismatch | 0.379381 | 0.0 | 1.0 | true/true/true |
| rotation_scale_mismatch | 0.047587 | 0.0 | 1.0 | true/true/true |
| global_action_scale_mismatch | 0.233874 | 0.0 | 1.0 | true/true/false |
| per_dimension_scale_mismatch | 0.211681 | 0.0 | 1.0 | true/true/true |
| gripper_threshold_0_1_mismatch | 0.504084 | 0.0 | 1.0 | true/true/true |
| range_clipping_mismatch | 0.50646 | 0.0 | 1.0 | true/true/false |

## Exact-Init Replay Aggregate

- cases: `21`
- degraded cases: `19`
- success recovered: `17`
- reward recovered: `17`
- done-index recovered: `17`
- simple baseline match count: `4`
- failure count: `8`
- failure reasons: `clipping_or_global_matches_full_replay_recovery, full_repair_did_not_recover_replay, wrong_spec_did_not_degrade`

| mismatch | cases | degraded | success recovered | reward recovered | done recovered |
| --- | ---: | ---: | ---: | ---: | ---: |
| gripper_sign_flip | 3 | 3 | 3 | 3 | 3 |
| translation_scale_mismatch | 3 | 3 | 3 | 3 | 3 |
| rotation_scale_mismatch | 3 | 1 | 1 | 1 | 1 |
| global_action_scale_mismatch | 3 | 3 | 2 | 2 | 2 |
| per_dimension_scale_mismatch | 3 | 3 | 3 | 3 | 3 |
| gripper_threshold_0_1_mismatch | 3 | 3 | 3 | 3 | 3 |
| range_clipping_mismatch | 3 | 3 | 2 | 2 | 2 |

## Replay Cases

| demo | mismatch | expert | wrong | clipping | global | diagonal | gripper | full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo | gripper_sign_flip | 1.0/true | 0.0/false | 0.0/false | 0.0/false | 1.0/true | 1.0/true | 1.0/true |
| KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo | translation_scale_mismatch | 1.0/true | 0.0/false | 0.0/false | 0.0/false | 1.0/true | 0.0/false | 1.0/true |
| KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo | rotation_scale_mismatch | 1.0/true | 0.0/false | 0.0/false | 0.0/false | 1.0/true | 0.0/false | 1.0/true |
| KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo | global_action_scale_mismatch | 1.0/true | 0.0/false | 0.0/false | 0.0/false | 0.0/false | 0.0/false | 0.0/false |
| KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo | per_dimension_scale_mismatch | 1.0/true | 0.0/false | 0.0/false | 0.0/false | 1.0/true | 0.0/false | 1.0/true |
| KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo | gripper_threshold_0_1_mismatch | 1.0/true | 0.0/false | 0.0/false | 0.0/false | 1.0/true | 1.0/true | 1.0/true |
| KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo | range_clipping_mismatch | 1.0/true | 0.0/false | 0.0/false | 0.0/false | 0.0/false | 0.0/false | 0.0/false |
| LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo | gripper_sign_flip | 1.0/true | 0.0/false | 0.0/false | 0.0/false | 1.0/true | 1.0/true | 1.0/true |
| LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo | translation_scale_mismatch | 1.0/true | 0.0/false | 0.0/false | 0.0/false | 1.0/true | 0.0/false | 1.0/true |
| LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo | rotation_scale_mismatch | 1.0/true | 1.0/true | 1.0/true | 1.0/true | 1.0/true | 1.0/true | 1.0/true |
| LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo | global_action_scale_mismatch | 1.0/true | 0.0/false | 0.0/false | 1.0/true | 1.0/true | 0.0/false | 1.0/true |
| LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo | per_dimension_scale_mismatch | 1.0/true | 0.0/false | 0.0/false | 0.0/false | 1.0/true | 0.0/false | 1.0/true |
| LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo | gripper_threshold_0_1_mismatch | 1.0/true | 0.0/false | 0.0/false | 0.0/false | 1.0/true | 1.0/true | 1.0/true |
| LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo | range_clipping_mismatch | 1.0/true | 0.0/false | 0.0/false | 1.0/true | 1.0/true | 0.0/false | 1.0/true |
| LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo | gripper_sign_flip | 1.0/true | 0.0/false | 0.0/false | 0.0/false | 1.0/true | 1.0/true | 1.0/true |
| LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo | translation_scale_mismatch | 1.0/true | 0.0/false | 0.0/false | 0.0/false | 1.0/true | 0.0/false | 1.0/true |
| LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo | rotation_scale_mismatch | 1.0/true | 1.0/true | 1.0/true | 1.0/true | 1.0/true | 1.0/true | 1.0/true |
| LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo | global_action_scale_mismatch | 1.0/true | 0.0/false | 0.0/false | 1.0/true | 1.0/true | 0.0/false | 1.0/true |
| LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo | per_dimension_scale_mismatch | 1.0/true | 0.0/false | 0.0/false | 0.0/false | 1.0/true | 0.0/false | 1.0/true |
| LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo | gripper_threshold_0_1_mismatch | 1.0/true | 0.0/false | 0.0/false | 0.0/false | 1.0/true | 1.0/true | 1.0/true |
| LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo | range_clipping_mismatch | 1.0/true | 0.0/false | 0.0/false | 1.0/true | 1.0/true | 0.0/false | 1.0/true |

## Calibration Data-Size Sensitivity

| calibration demos | samples | full L2 | full recovery | success replay evaluated |
| ---: | ---: | ---: | ---: | --- |
| 1 | 261 | 0.0 | 1.0 | false |
| 3 | 861 | 0.0 | 1.0 | false |
| 5 | 1403 | 0.0 | 1.0 | false |

## Exact-Init vs Default Reset

- default-reset sanity scope: `one demo and one mismatch only; non-primary sanity check`
- default-reset expert succeeded: `False`
- default-reset full repair succeeded: `False`
- primary claim boundary: exact-init executable-spec repair under matched replay conditions.