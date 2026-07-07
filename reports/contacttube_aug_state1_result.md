# ContactTube-Aug STATE 1 Result

Bounded replay/control diagnostic only. This is not benchmark success, paper-grade evidence, or a policy-training result.

- decision: `kill`
- reason: ContactTube-Aug trajectory was not controller-valid/replay-valid.
- replay happened: `True`
- training happened: `False`
- loss computed: `False`
- GPU/download/OpenVLA-OFT: `False` / `False` / `False`
- demos/tasks: `1 / 1`
- contact-tube extraction success: `True`
- HDF5 object pose available: `False`
- runtime object pose available: `True`
- augmentation validity: `False`
- simple object-relative matches/beats ContactTube-Aug: `True`
- ContactTube-Aug beats simple object-relative: `False`
- next state: `archive_or_reframe_contacttube_aug_before_training`

## Case

- task: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`
- instruction: turn on the stove and put the moka pot on it
- selected horizon: `272`
- HDF5 first reward/done/signal: `271` / `271` / `271`
- HDF5 EEF source: `ee_pos`
- HDF5 object source: `None`
- translation unit source: `median_hdf5_eef_delta_norm_per_translation_action_norm`

## Replay Metrics

| variant | init | reward | success | first done | steps | tube score | dist MAE | motion err | lift err | place err | grip close err | valid rate | clip step |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_init_noop_upper_bound | exact | 1.0 | true | 260 | 261 | n/a | n/a | n/a | n/a | n/a | 0 | 1.0 | 0.0 |
| raw_demo_replay | default_reset | 0.0 | false | n/a | 272 | 0.169342 | 0.030056 | 0 | 99 | 14 | 0 | 1.0 | 0.0 |
| random_pose_jitter | default_reset | 0.0 | false | n/a | 272 | 0.166599 | 0.029383 | 1 | 99 | 14 | 0 | 1.0 | 0.0 |
| simple_object_relative_translation_retarget | default_reset | 0.0 | false | n/a | 272 | 0.009154 | 0.002585 | 0 | 0 | 1 | 0 | 0.988971 | 0.011029 |
| random_action_jitter | default_reset | 0.0 | false | n/a | 272 | 0.176102 | 0.03064 | 10 | 99 | 14 | 0 | 1.0 | 0.0 |
| contacttube_aug | default_reset | 0.0 | false | n/a | 272 | 0.015226 | 0.002393 | 0 | 2 | 4 | 0 | 0.849265 | 0.150735 |

## Baseline Gate

- baselines tested: `raw_demo_replay, random_pose_jitter, simple_object_relative_translation_retarget, random_action_jitter`
- method variant: `contacttube_aug`
- ContactTube-Aug beats random action jitter: `True`
- ContactTube-Aug beats random pose jitter: `True`
- ContactTube-Aug beats simple object-relative: `False`

## Feasibility Notes

- Object pose shift is represented by the simulator default-reset object start relative to exact HDF5 init when available.
- Reset pose shift uses default reset versus exact HDF5 init; no task-generic init-state object editor is assumed.
- Distractor insertion/relabeling and camera perturbation are logged as not feasible in this smoke because no training/render augmentation is run.
- Exact-init no-op replay is the upper bound/control; default-reset variants are diagnostics for augmentation validity.
- The runner does not use reward labels or success labels to choose ContactTube-Aug actions.
