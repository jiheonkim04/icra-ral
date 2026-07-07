# ResetSpec-Retarget STATE 1 Result

Bounded replay/retarget diagnostic only. This is not benchmark success, paper-grade evidence, or a policy rollout claim.

- decision: `kill`
- reason: Object-relative retargeting did not beat the simple action-only baselines.
- replay happened: `True`
- training happened: `False`
- loss computed: `False`
- exact-init expert replay success: `True`
- default-reset raw replay success: `False`
- object poses available: `True`
- best retarget variant: `object_relative_translation_gripper_phase_retarget`
- best simple baseline: `default_reset_global_scale_replay`
- object-relative beats simple baselines: `False`
- next state: `archive_or_reframe_resetspec_retarget`

## Case

- task: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`
- instruction: turn on the stove and put the moka pot on it
- selected horizon: `272`
- HDF5 first reward/done/signal: `271` / `271` / `271`
- HDF5 EEF trajectory source: `ee_pos`
- translation unit source: `median_hdf5_eef_delta_norm_per_translation_action_norm`

## Replay Metrics

| variant | init | reward | success | first done | steps | dist change | object move | traj drift | clip step | trans err | rot err | grip timing err |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hdf5_expert_replay_exact_init | exact | 1.0 | true | 260 | 261 | -0.267427 | 0.217868 | 0.015495 | 0.0 | 0.0 | 0.0 | 0 |
| hdf5_expert_replay_default_reset | default | 0.0 | false | n/a | 272 | -0.124403 | 0.006482 | 0.039519 | 0.0 | 0.0 | 0.0 | 0 |
| default_reset_diagonal_affine_replay | default | 0.0 | false | n/a | 272 | -0.124403 | 0.006482 | 0.039519 | 0.0 | 0.0 | 0.0 | 0 |
| default_reset_global_scale_replay | default | 1.0 | true | 257 | 258 | -0.236694 | 0.189037 | 0.035747 | 0.0 | 0.067315 | 0.013029 | 0 |
| default_reset_clipping_replay | default | 0.0 | false | n/a | 272 | -0.124403 | 0.006482 | 0.039519 | 0.0 | 0.0 | 0.0 | 0 |
| object_relative_translation_retarget | default | 0.0 | false | n/a | 272 | -0.232446 | 0.231257 | 0.002036 | 0.007353 | 0.153467 | 0.0 | 0 |
| object_relative_translation_gripper_phase_retarget | default | 0.0 | false | n/a | 272 | -0.247037 | 0.21788 | 0.00201 | 0.007353 | 0.15355 | 0.0 | 1 |

## Skipped Conditions

- perturbed-init raw replay: `not_run_no_task_generic_safe_state_perturbation_helper`
- nearest-demo replay: `not_run_no_nonleaking_nearest_demo_selector_with_object_pose_cache`

## Non-Leakage Notes

- Target object key is resolved from natural-language instruction text plus visible observation object keys.
- The runner does not use BDDL target metadata, eval labels, dataset target labels, task IDs, filenames, or manifest target fields as inference-time target proxies.
- Retargeted actions use demonstration EEF trajectory and current object/EEF state as replay diagnostics, not as an online policy-performance claim.
