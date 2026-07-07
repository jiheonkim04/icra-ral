# TL-ChunkRepair STATE 1 Result

Bounded replay/control diagnostic only. This is not benchmark success, paper-grade evidence, or a policy rollout claim.

- decision: `kill`
- reason: TL-ChunkRepair did not beat the best single simple baseline.
- replay happened: `True`
- training happened: `False`
- loss computed: `False`
- GPU/download/OpenVLA-OFT: `False` / `False` / `False`
- demos/tasks: `1 / 1`
- temporal properties tested: `grasp_before_lift, keep_grasp_until_placement, do_not_release_before_target_region, do_not_move_object_while_gripper_open, avoid_forbidden_contact_before_safe_phase, mechanism_action_onset_order`
- perturbations tested: `8`
- baselines tested: `no_repair, clipping_only, safety_only_one_step_filter, gripper_only_timing_fix, fixed_delay_shift, linear_time_warp, abort_to_stop, repeat_last_hold, tl_chunkrepair`
- perturbations degraded replay: `7`
- TL violation reductions: `8`
- TL safe-success count: `0`
- best single simple baseline: `no_repair`
- TL beats best single baseline: `False`
- TL beats best per-failure baseline: `False`
- next state: `archive_or_reframe_tl_chunkrepair`

## Case

- task: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`
- instruction: turn on the stove and put the moka pot on it
- selected horizon: `272`
- HDF5 first reward/done/signal: `271` / `271` / `271`
- HDF5 EEF source: `ee_pos`
- HDF5 object source: `None`
- event anchors: `{'approach_index': 0, 'demo_eef_object_distance': {'available': True, 'final': 0.278242, 'min': 0.0, 'min_index': 0, 'start': 0.0}, 'gripper_close_index': 62, 'horizon': 272, 'lift_index': 119, 'object_motion_onset_index': None, 'place_or_contact_index': None, 'safe_release_index': 262}`

## Perturbation Summary

| perturbation | raw degraded | raw violations | TL violations | TL safe success | best simple | best simple safe success | TL beats best simple |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| early_gripper_release | true | 4 | 0 | false | fixed_delay_shift | false | false |
| delayed_gripper_close | true | 3 | 0 | false | fixed_delay_shift | false | false |
| lift_before_grasp | true | 3 | 0 | false | fixed_delay_shift | false | false |
| transport_with_gripper_open | true | 4 | 0 | false | fixed_delay_shift | false | false |
| premature_place_release | true | 4 | 0 | false | fixed_delay_shift | false | false |
| chunk_truncation | false | 4 | 0 | false | no_repair | false | false |
| phase_skip | true | 4 | 0 | false | fixed_delay_shift | false | false |
| inserted_unsafe_contact_action | true | 5 | 0 | false | fixed_delay_shift | false | false |

## Method Totals

| method | safe success | success | reward sum | mean edit L1 |
| --- | ---: | ---: | ---: | ---: |
| no_repair | 0 | 1 | 1.0 | 0.0 |
| clipping_only | 0 | 1 | 1.0 | 0.0 |
| safety_only_one_step_filter | 0 | 1 | 1.0 | 0.000394 |
| gripper_only_timing_fix | 0 | 0 | 0.0 | 0.074449 |
| fixed_delay_shift | 0 | 0 | 0.0 | 0.200975 |
| linear_time_warp | 0 | 0 | 0.0 | 0.15693 |
| abort_to_stop | 0 | 0 | 0.0 | 0.065348 |
| repeat_last_hold | 0 | 1 | 1.0 | 0.016178 |
| tl_chunkrepair | 0 | 0 | 0.0 | 0.075254 |

## Non-Leakage Notes

- Predicate source is action chunk timing plus HDF5/visible EEF state where available.
- The diagnostic does not use reward labels, success labels, task ids, BDDL target fields, or dataset target labels to select repair actions.
- Exact-init replay is a bounded local control diagnostic, not a benchmark or paper-grade claim.
