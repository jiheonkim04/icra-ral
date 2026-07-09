# SmolVLA 7D Offline-To-Control Gap Diagnosis

Final decision: `FEATURE_PATH_MISMATCH`

This is an evaluation and interface diagnosis, not a new RA-L method.

## Summary

- eligible demos used: `['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_8', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_30', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_31', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_5', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_7', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_8']`
- feature path audit result: `FEATURE_PATH_MISMATCH_FOR_TRUE_CLOSED_LOOP`
- live feature orientation L2 mean: `2.141995`
- adapter teacher-forced action L2: `0.867437`
- adapter teacher-forced gripper error: `0.628795`
- mean/ridge/adapter open-loop progress: `0.038336` / `0.040788` / `-0.059671`
- closed-loop divergence: `{'executed': False, 'reason': 'Skipped as a model-quality measurement because STATE 1 found live closed-loop feature mismatch.', 'blocked_by': 'FEATURE_PATH_MISMATCH_FOR_TRUE_CLOSED_LOOP', 'required_before_rerun': 'Provide live env features matching HDF5 ee_states (ee_pos + ee_ori) or retrain/evaluate with the live observation schema.'}`
- oracle diagnostics: `{'adapter_motion_error_first6_mean': 0.494638, 'adapter_rotation_l2_mean': 0.089128, 'adapter_gripper_error_mean': 0.628795, 'gripper_oracle_alone_unlikely_to_fix_motion': True}`
- failure category: `FEATURE_PATH_MISMATCH`

Exact next step: Fix the live closed-loop feature schema so replay uses HDF5-compatible ee_states features, then rerun teacher-forced and replay diagnostics before any method work.
