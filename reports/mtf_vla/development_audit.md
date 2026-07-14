# MTF-VLA Development Audit

Date: `2026-07-14`

Proposal hash: `11DC94A2B75CD8605577AB044E5743DFDA4131A4FA7F6C6A7390519B9F995B31`

Final decision: `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`

- closed-loop experiment happened: `False`
- training happened: `False`
- scoreable development records: `1600`
- raw prediction records: `2800`
- train records: `1200`
- validation records: `400`
- reserved records not used: `1200`
- selected task count: `40`
- duplicate sample keys: `0`
- duplicate frame keys: `0`
- high milestone count: `400`
- retention frame count: `400`
- high milestone fraction: `0.25`
- retention frame fraction: `0.25`
- high-low score gap: `0.5857019297500093`
- gripper transition fraction: `0.341875`
- state joined fraction: `1.0`
- uniform overlap fraction: `0.2175`
- adapter init action delta p95: `0.0`

Base headroom:

```json
{
  "available": true,
  "frozen_base_task_balanced_success_rate": 0.74,
  "max_allowed": 0.9,
  "min_allowed": 0.05,
  "passes": true
}
```

Frame score summary:

```json
{
  "count": 1600,
  "max": 1.0,
  "median": 0.21491683325962457,
  "min": 0.0,
  "p10": 0.0,
  "p90": 0.8256337603707409
}
```

Base-retention target manifest:

```json
{
  "reloadable": true,
  "sha256": "3312B604E5F60EC0C0F909792C1410AF17FAD7ED50FB4861E6602056E88A5ACE",
  "source": "prediction_artifact.base_action",
  "target_count": 400
}
```

Split manifest:

```json
{
  "confirmatory_reserved_splits": [
    "test"
  ],
  "reserved_record_count": 1200,
  "split_overlap": {
    "train_reserved": 0,
    "train_validation": 0,
    "validation_reserved": 0
  },
  "train_record_count": 1200,
  "train_splits": [
    "train"
  ],
  "validation_record_count": 400,
  "validation_splits": [
    "val"
  ]
}
```

FrameSkip proxy:

```json
{
  "components": [
    "action_variation",
    "gripper_transition_preservation",
    "task_progress_phase_coverage",
    "kinematic_turning_point"
  ],
  "constructible": true,
  "omission_reason": "Stage 0 avoids video decoding; state/action coherence is used as the transparent local proxy.",
  "omitted_components": [
    "visual_action_coherence"
  ]
}
```

Hard stop reasons:
- none

Next step: Run the bounded six-config MTF validation search.
