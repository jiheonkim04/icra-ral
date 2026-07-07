# Phase-Locked Retiming STATE 1 Result

Bounded replay/control diagnostic only. This is not benchmark success, paper-grade evidence, or a policy rollout claim.

- decision: `kill`
- reason: Event-locked retiming did not improve replay/progress over raw perturbed replay.
- replay happened: `True`
- training happened: `False`
- loss computed: `False`
- GPU/download/OpenVLA-OFT: `False` / `False` / `False`
- demos/tasks: `1 / 1`
- perturbations tested: `9`
- baselines tested: `raw_perturbed_replay, fixed_time_shift, repeat_last_hold, gripper_only_timing_correction, global_scale, diagonal_affine, linear_time_warp, nearest_progress_demo, event_locked_retiming`
- phase mismatch degraded replay count: `9`
- event-locked beats best simple count: `0`
- simple baseline matches/beats event count: `3`
- next state: `archive_or_reframe_phase_locked_retiming`

## Case

- task: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`
- instruction: turn on the stove and put the moka pot on it
- selected horizon: `272`
- HDF5 first reward/done/signal: `271` / `271` / `271`
- HDF5 EEF source: `ee_pos`
- HDF5 object source: `None`
- event anchors: `{'approach_index': 0, 'gripper_close_index': 62, 'object_motion_onset_index': None, 'lift_index': 119, 'place_or_contact_index': None, 'horizon': 272, 'demo_eef_object_distance': {'available': True, 'min': 0.0, 'min_index': 0, 'start': 0.0, 'final': 0.278242}}`

## Exact Expert Replay

| reward | success | first done | steps | dist change | object move | traj drift |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1.0 | true | 260 | 261 | -0.267427 | 0.217868 | 0.015495 |

## Perturbation Summary

| perturbation | raw degraded | raw success | event success | best simple | best simple success | event beats best simple | simple matches/beats event |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gripper_close_delayed | true | false | false | gripper_only_timing_correction | true | false | false |
| gripper_close_early | true | false | false | gripper_only_timing_correction | true | false | false |
| lift_phase_delayed | true | false | false | fixed_time_shift | false | false | false |
| lift_phase_early | true | false | false | repeat_last_hold | false | false | false |
| chunk_shifted_forward | true | false | false | repeat_last_hold | false | false | true |
| chunk_shifted_backward | true | false | false | fixed_time_shift | true | false | false |
| time_stretch | true | false | false | linear_time_warp | false | false | true |
| time_compression | true | false | false | linear_time_warp | true | false | false |
| chunk_boundary_offset | true | false | false | diagonal_affine | false | false | true |

## Replay Metrics

| perturbation | baseline | reward | success | first done | steps | dist change | object move | event err | grip err | traj drift | clip step |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gripper_close_delayed | raw_perturbed_replay | 0.0 | false | n/a | 272 | -0.199703 | 0.003387 | 21.0 | 18 | 0.031362 | 0.0 |
| gripper_close_delayed | fixed_time_shift | 0.0 | false | n/a | 272 | -0.220436 | 0.004683 | 14.666667 | 0 | 0.112795 | 0.0 |
| gripper_close_delayed | repeat_last_hold | 0.0 | false | n/a | 272 | -0.186875 | 0.000302 | 22.0 | 19 | 0.02928 | 0.0 |
| gripper_close_delayed | gripper_only_timing_correction | 1.0 | true | 260 | 261 | -0.267427 | 0.217868 | 47.333333 | 0 | 0.015495 | 0.0 |
| gripper_close_delayed | global_scale | 0.0 | false | n/a | 272 | -0.168408 | 0.008096 | 22.0 | 18 | 0.048418 | 0.0 |
| gripper_close_delayed | diagonal_affine | 0.0 | false | n/a | 272 | -0.199703 | 0.003387 | 21.0 | 18 | 0.031362 | 0.0 |
| gripper_close_delayed | linear_time_warp | 0.0 | false | n/a | 272 | -0.199703 | 0.003387 | 21.0 | 18 | 0.031362 | 0.0 |
| gripper_close_delayed | nearest_progress_demo | 0.0 | false | n/a | 272 | 0.037228 | 0.215318 | 124.5 | n/a | 0.419144 | 0.0 |
| gripper_close_delayed | event_locked_retiming | 0.0 | false | n/a | 272 | 0.630111 | 0.000523 | 56.666667 | 62 | 0.465561 | 0.0 |
| gripper_close_early | raw_perturbed_replay | 0.0 | false | n/a | 272 | -0.174277 | 4e-06 | 23.0 | 18 | 0.052753 | 0.0 |
| gripper_close_early | fixed_time_shift | 0.0 | false | n/a | 272 | -0.103278 | 0.000519 | 29.333333 | 0 | 0.0954 | 0.0 |
| gripper_close_early | repeat_last_hold | 0.0 | false | n/a | 272 | -0.173284 | 4e-06 | 23.0 | 17 | 0.052604 | 0.0 |
| gripper_close_early | gripper_only_timing_correction | 1.0 | true | 260 | 261 | -0.267427 | 0.217868 | 47.333333 | 0 | 0.015495 | 0.0 |
| gripper_close_early | global_scale | 0.0 | false | n/a | 272 | -0.182017 | 0.00073 | 25.0 | 18 | 0.040894 | 0.0 |
| gripper_close_early | diagonal_affine | 0.0 | false | n/a | 272 | -0.174277 | 4e-06 | 23.0 | 18 | 0.052753 | 0.0 |
| gripper_close_early | linear_time_warp | 0.0 | false | n/a | 272 | -0.174277 | 4e-06 | 23.0 | 18 | 0.052753 | 0.0 |
| gripper_close_early | nearest_progress_demo | 0.0 | false | n/a | 272 | 0.037228 | 0.215318 | 124.5 | n/a | 0.419144 | 0.0 |
| gripper_close_early | event_locked_retiming | 0.0 | false | n/a | 272 | 0.630111 | 0.000523 | 56.666667 | 62 | 0.465561 | 0.0 |
| lift_phase_delayed | raw_perturbed_replay | 0.0 | false | n/a | 272 | -0.064025 | 4e-06 | 14.666667 | 0 | 0.094858 | 0.0 |
| lift_phase_delayed | fixed_time_shift | 0.0 | false | n/a | 272 | -0.150754 | 4e-06 | 20.0 | 18 | 0.142703 | 0.0 |
| lift_phase_delayed | repeat_last_hold | 0.0 | false | n/a | 272 | -0.060668 | 4e-06 | 15.666667 | 1 | 0.097994 | 0.0 |
| lift_phase_delayed | gripper_only_timing_correction | 0.0 | false | n/a | 272 | -0.064025 | 4e-06 | 14.666667 | 0 | 0.094858 | 0.0 |
| lift_phase_delayed | global_scale | 0.0 | false | n/a | 272 | -0.077899 | 4e-06 | 16.333333 | 0 | 0.088637 | 0.0 |
| lift_phase_delayed | diagonal_affine | 0.0 | false | n/a | 272 | -0.064025 | 4e-06 | 14.666667 | 0 | 0.094858 | 0.0 |
| lift_phase_delayed | linear_time_warp | 0.0 | false | n/a | 272 | -0.064025 | 4e-06 | 14.666667 | 0 | 0.094858 | 0.0 |
| lift_phase_delayed | nearest_progress_demo | 0.0 | false | n/a | 272 | 0.037228 | 0.215318 | 124.5 | n/a | 0.419144 | 0.0 |
| lift_phase_delayed | event_locked_retiming | 0.0 | false | n/a | 272 | 0.630111 | 0.000523 | 56.666667 | 62 | 0.465561 | 0.0 |
| lift_phase_early | raw_perturbed_replay | 0.0 | false | n/a | 272 | -0.054661 | 4e-06 | 14.666667 | 0 | 0.107406 | 0.0 |
| lift_phase_early | fixed_time_shift | 0.0 | false | n/a | 272 | -0.05124 | 4e-06 | 32.333333 | 18 | 0.123895 | 0.0 |
| lift_phase_early | repeat_last_hold | 0.0 | false | n/a | 272 | -0.05906 | 4e-06 | 15.666667 | 1 | 0.10652 | 0.0 |
| lift_phase_early | gripper_only_timing_correction | 0.0 | false | n/a | 272 | -0.054661 | 4e-06 | 14.666667 | 0 | 0.107406 | 0.0 |
| lift_phase_early | global_scale | 0.0 | false | n/a | 272 | -0.056308 | 4e-06 | 18.0 | 0 | 0.119059 | 0.0 |
| lift_phase_early | diagonal_affine | 0.0 | false | n/a | 272 | -0.054661 | 4e-06 | 14.666667 | 0 | 0.107406 | 0.0 |
| lift_phase_early | linear_time_warp | 0.0 | false | n/a | 272 | -0.054661 | 4e-06 | 14.666667 | 0 | 0.107406 | 0.0 |
| lift_phase_early | nearest_progress_demo | 0.0 | false | n/a | 272 | 0.037228 | 0.215318 | 124.5 | n/a | 0.419144 | 0.0 |
| lift_phase_early | event_locked_retiming | 0.0 | false | n/a | 272 | 0.630111 | 0.000523 | 56.666667 | 62 | 0.465561 | 0.0 |
| chunk_shifted_forward | raw_perturbed_replay | 0.0 | false | n/a | 272 | -0.226487 | 4e-06 | 20.0 | 18 | 0.113523 | 0.0 |
| chunk_shifted_forward | fixed_time_shift | 0.0 | false | n/a | 272 | -0.109815 | 0.000465 | 3.0 | 0 | 0.060489 | 0.0 |
| chunk_shifted_forward | repeat_last_hold | 0.0 | false | n/a | 272 | -0.23015 | 0.001435 | 19.333333 | 17 | 0.112317 | 0.0 |
| chunk_shifted_forward | gripper_only_timing_correction | 0.0 | false | n/a | 272 | -0.220437 | 0.004683 | 14.666667 | 0 | 0.112795 | 0.0 |
| chunk_shifted_forward | global_scale | 0.0 | false | n/a | 272 | -0.16867 | 0.000634 | 16.666667 | 18 | 0.107478 | 0.0 |
| chunk_shifted_forward | diagonal_affine | 0.0 | false | n/a | 272 | -0.226487 | 4e-06 | 20.0 | 18 | 0.113523 | 0.0 |
| chunk_shifted_forward | linear_time_warp | 0.0 | false | n/a | 272 | -0.226487 | 4e-06 | 20.0 | 18 | 0.113523 | 0.0 |
| chunk_shifted_forward | nearest_progress_demo | 0.0 | false | n/a | 272 | 0.037228 | 0.215318 | 124.5 | n/a | 0.419144 | 0.0 |
| chunk_shifted_forward | event_locked_retiming | 0.0 | false | n/a | 272 | 0.630111 | 0.000523 | 56.666667 | 62 | 0.465561 | 0.0 |
| chunk_shifted_backward | raw_perturbed_replay | 0.0 | false | n/a | 272 | -0.266411 | 0.216839 | 65.333333 | 18 | 0.087515 | 0.0 |
| chunk_shifted_backward | fixed_time_shift | 1.0 | true | 260 | 261 | -0.267998 | 0.22705 | 47.333333 | 0 | 0.015473 | 0.0 |
| chunk_shifted_backward | repeat_last_hold | 0.0 | false | n/a | 272 | -0.268705 | 0.223005 | 66.0 | 19 | 0.0908 | 0.0 |
| chunk_shifted_backward | gripper_only_timing_correction | 0.0 | false | n/a | 272 | -0.103278 | 0.000519 | 29.333333 | 0 | 0.0954 | 0.0 |
| chunk_shifted_backward | global_scale | 0.0 | false | n/a | 272 | -0.133298 | 0.002172 | 34.666667 | 18 | 0.093773 | 0.0 |
| chunk_shifted_backward | diagonal_affine | 0.0 | false | n/a | 272 | -0.266411 | 0.216839 | 65.333333 | 18 | 0.087515 | 0.0 |
| chunk_shifted_backward | linear_time_warp | 0.0 | false | n/a | 272 | -0.266411 | 0.216839 | 65.333333 | 18 | 0.087515 | 0.0 |
| chunk_shifted_backward | nearest_progress_demo | 0.0 | false | n/a | 272 | 0.037228 | 0.215318 | 124.5 | n/a | 0.419144 | 0.0 |
| chunk_shifted_backward | event_locked_retiming | 0.0 | false | n/a | 272 | 0.630111 | 0.000523 | 56.666667 | 62 | 0.465561 | 0.0 |
| time_stretch | raw_perturbed_replay | 0.0 | false | n/a | 272 | -0.128942 | 4e-06 | 25.0 | 10 | 0.103925 | 0.0 |
| time_stretch | fixed_time_shift | 0.0 | false | n/a | 272 | -0.128942 | 4e-06 | 25.0 | 10 | 0.103925 | 0.0 |
| time_stretch | repeat_last_hold | 0.0 | false | n/a | 272 | -0.128991 | 4e-06 | 26.0 | 11 | 0.106502 | 0.0 |
| time_stretch | gripper_only_timing_correction | 0.0 | false | n/a | 272 | -0.141195 | 5e-06 | 22.333333 | 0 | 0.10043 | 0.0 |
| time_stretch | global_scale | 0.0 | false | n/a | 272 | -0.158028 | 0.004556 | 26.666667 | 10 | 0.088259 | 0.0 |
| time_stretch | diagonal_affine | 0.0 | false | n/a | 272 | -0.128942 | 4e-06 | 25.0 | 10 | 0.103925 | 0.0 |
| time_stretch | linear_time_warp | 0.0 | false | n/a | 272 | -0.250292 | 0.369555 | 49.0 | 1 | 0.02796 | 0.0 |
| time_stretch | nearest_progress_demo | 0.0 | false | n/a | 272 | 0.037228 | 0.215318 | 124.5 | n/a | 0.419144 | 0.0 |
| time_stretch | event_locked_retiming | 0.0 | false | n/a | 272 | 0.630111 | 0.000523 | 56.666667 | 62 | 0.465561 | 0.0 |
| time_compression | raw_perturbed_replay | 0.0 | false | n/a | 272 | -0.156444 | 0.002099 | 18.666667 | 9 | 0.077465 | 0.0 |
| time_compression | fixed_time_shift | 0.0 | false | n/a | 272 | -0.156444 | 0.002099 | 18.666667 | 9 | 0.077465 | 0.0 |
| time_compression | repeat_last_hold | 0.0 | false | n/a | 272 | -0.156251 | 0.002016 | 18.333333 | 8 | 0.075523 | 0.0 |
| time_compression | gripper_only_timing_correction | 0.0 | false | n/a | 272 | -0.162289 | 0.003969 | 16.666667 | 0 | 0.090476 | 0.0 |
| time_compression | global_scale | 0.0 | false | n/a | 272 | -0.182201 | 0.005913 | 17.0 | 9 | 0.096586 | 0.0 |
| time_compression | diagonal_affine | 0.0 | false | n/a | 272 | -0.156444 | 0.002099 | 18.666667 | 9 | 0.077465 | 0.0 |
| time_compression | linear_time_warp | 1.0 | true | 259 | 260 | -0.260573 | 0.211714 | 49.0 | 1 | 0.019787 | 0.0 |
| time_compression | nearest_progress_demo | 0.0 | false | n/a | 272 | 0.037228 | 0.215318 | 124.5 | n/a | 0.419144 | 0.0 |
| time_compression | event_locked_retiming | 0.0 | false | n/a | 272 | 0.630111 | 0.000523 | 56.666667 | 62 | 0.465561 | 0.0 |
| chunk_boundary_offset | raw_perturbed_replay | 0.0 | false | n/a | 272 | -0.253681 | 0.175259 | 54.0 | 0 | 0.064728 | 0.0 |
| chunk_boundary_offset | fixed_time_shift | 0.0 | false | n/a | 272 | -0.209453 | 0.001552 | 15.333333 | 18 | 0.084931 | 0.0 |
| chunk_boundary_offset | repeat_last_hold | 0.0 | false | n/a | 272 | -0.23904 | 0.22444 | 54.0 | 1 | 0.071217 | 0.0 |
| chunk_boundary_offset | gripper_only_timing_correction | 0.0 | false | n/a | 272 | -0.122317 | 0.006393 | 20.333333 | 0 | 0.063616 | 0.0 |
| chunk_boundary_offset | global_scale | 0.0 | false | n/a | 272 | -0.134759 | 0.001363 | 22.666667 | 0 | 0.071431 | 0.0 |
| chunk_boundary_offset | diagonal_affine | 0.0 | false | n/a | 272 | -0.253681 | 0.175259 | 54.0 | 0 | 0.064728 | 0.0 |
| chunk_boundary_offset | linear_time_warp | 0.0 | false | n/a | 272 | -0.253681 | 0.175259 | 54.0 | 0 | 0.064728 | 0.0 |
| chunk_boundary_offset | nearest_progress_demo | 0.0 | false | n/a | 272 | 0.037228 | 0.215318 | 124.5 | n/a | 0.419144 | 0.0 |
| chunk_boundary_offset | event_locked_retiming | 0.0 | false | n/a | 272 | 0.630111 | 0.000523 | 56.666667 | 62 | 0.465561 | 0.0 |

## Non-Leakage Notes

- Target object key is resolved from natural-language instruction text plus visible observation object keys.
- Event-locked retiming uses the demonstration chunk being retimed and current observation progress; it does not use reward labels or success labels for action selection.
- Nearest-progress demo is reported as a strong simple baseline, not as method novelty.
