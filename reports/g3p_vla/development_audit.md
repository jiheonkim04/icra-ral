# G3P-VLA Development Audit

Date: `2026-07-15`

Proposal hash: `BEE3822D8F54EFBD09C1CA47A9BF126EBE694B7B6219002FF770C5794ED7AA71`

Final decision: `DATA_OR_SUPERVISION_FAILURE`

- closed-loop experiment happened: `False`
- training happened: `False`
- validation search happened: `False`
- confirmatory-test tuning happened: `False`
- scoreable development records: `2800`
- train records: `1200`
- validation records: `400`
- reserved records not used: `1200`
- selected task count: `40`
- duplicate sample keys: `0`
- duplicate frame keys: `0`
- source gate passed: `True`
- RGB video available in dataset: `True`
- privileged object/pose feature available: `False`
- train valid point fraction: `0.9333333333333333`
- validation valid point fraction: `0.9`
- validation material point fraction: `1.0`
- point predictability margin: `0.2136890612067978`
- best trivial baseline: `base_action_only`
- oracle action headroom L2 validation: `0.08630366897708504`
- initial action delta p95: `0.0`
- base action validity: `1.0`
- point gradient norm: `0.17689797539443197`
- adapter surrogate gradient norm: `0.06257652460706827`

Source gate manifest:

```json
{
  "dataset_feature_names": [
    "action",
    "episode_index",
    "frame_index",
    "index",
    "observation.images.image",
    "observation.images.image2",
    "observation.state",
    "task_index",
    "timestamp"
  ],
  "forbidden_inference_keys": [
    "dataset_global_index",
    "episode_index",
    "frame_index",
    "future_action",
    "future_observation",
    "identity",
    "object_pose",
    "object_state",
    "oracle_help_label",
    "placement_pose",
    "reward",
    "success",
    "target_action"
  ],
  "future_waypoint_labels_used_for_training_only": true,
  "legal_inference_features": [
    "observation.images.image",
    "observation.images.image2",
    "observation.state",
    "language_or_task_instruction",
    "base_action"
  ],
  "object_or_pose_feature_names": [],
  "oracle_geometry_used_at_inference": false,
  "privileged_object_pose_available_as_dataset_feature": false,
  "rgb_video_available_in_dataset": true,
  "source_gate_passed": true,
  "state_available_in_dataset": true,
  "used_inference_features_for_stage_0_probe": [
    "observation.state",
    "base_action",
    "language_or_task_instruction_proxy"
  ]
}
```

Point label manifest:

```json
{
  "future_phase_offset": 0.15,
  "inference_uses_future_waypoint": false,
  "label_source": "future_eef_waypoint_from_official_development_state",
  "material_displacement_threshold": 0.005,
  "train_label_summary": {
    "coordinate_variance": [
      0.012193284964659984,
      0.02692965347513754,
      0.1402154804658926
    ],
    "coordinate_variance_nonzero_dims": 3,
    "displacement_norm_mean": 0.17385331887701555,
    "displacement_norm_p95": 0.34218248868894335,
    "invalid_point_count": 80,
    "material_point_count": 1118,
    "material_point_fraction_of_valid": 0.9982142857142857,
    "max_task_valid_share": 0.025,
    "source": "future_eef_waypoint_from_official_development_state",
    "task_count": 40,
    "total_records": 1200,
    "valid_point_count": 1120,
    "valid_point_fraction": 0.9333333333333333
  },
  "validation_label_summary": {
    "coordinate_variance": [
      0.011992391719559414,
      0.027109811602601796,
      0.1420020146150658
    ],
    "coordinate_variance_nonzero_dims": 3,
    "displacement_norm_mean": 0.1787342452815633,
    "displacement_norm_p95": 0.3417490838916082,
    "invalid_point_count": 40,
    "material_point_count": 360,
    "material_point_fraction_of_valid": 1.0,
    "max_task_valid_share": 0.025,
    "source": "future_eef_waypoint_from_official_development_state",
    "task_count": 40,
    "total_records": 400,
    "valid_point_count": 360,
    "valid_point_fraction": 0.9
  }
}
```

Predictability summary:

```json
{
  "accuracy_margin": 0.2136890612067978,
  "baseline_rmses": {
    "base_action_only": 0.16144493230608728,
    "phase_only": 0.19260205892360766,
    "state_only": 0.18059173401488288,
    "task_only": 0.19096333501453763,
    "train_mean": 0.19704668150704166,
    "zero_displacement": 0.20368746687465547
  },
  "best_trivial_baseline": "base_action_only",
  "best_trivial_rmse": 0.16144493230608728,
  "full_probe_features": [
    "observation.state",
    "base_action",
    "language_or_task_instruction_proxy"
  ],
  "full_probe_rmse": 0.11933821192093266,
  "full_probe_score": 0.3943657867860718,
  "full_probe_uses_only_deployment_observable_features": true,
  "label_scale_rmse": 0.19704668150704166,
  "valid": true
}
```

Gradient audit:

```json
{
  "adapter_surrogate_gradient_norm": 0.06257652460706827,
  "batch_size": 64,
  "largest_to_smallest_nonzero_ratio": 2.826906359936305,
  "point_probe_gradient_norm": 0.17689797539443197,
  "valid": true
}
```

Split manifest:

```json
{
  "duplicate_frame_keys": 0,
  "duplicate_sample_keys": 0,
  "reserved_records_not_used": 1200,
  "split_overlap": {
    "train_reserved": 0,
    "train_validation": 0,
    "validation_reserved": 0
  },
  "train_records": 1200,
  "validation_records": 400
}
```

Hard stop reasons:
- `train material point fraction collapsed: 0.998214`
- `validation material point fraction collapsed: 1.000000`

Next step: Record pre-rollout Stage 0 stop and continue to the next method cycle without rescuing G3P.
