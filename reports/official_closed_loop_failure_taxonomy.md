# Official Closed-Loop Failure Taxonomy

Date: 2026-07-11 KST

## Category Counts

```json
{
  "ambiguous_or_unclassified": 118
}
```

Automatic phase labels are intentionally conservative. Episodes without visual/semantic evidence are marked `ambiguous_or_unclassified` and placed in the bounded review queue.

## Review Queue Priority

The failure evidence is task-structured but not yet phase-structured. The bounded review queue should start from repeated all-policy task/reset failures, especially:

| Priority | Task/reset pair | Rationale |
| ---: | --- | --- |
| `1` | `libero_10/task_4/seed_20260713` | all four policies failed; task has the lowest aggregate success slice |
| `2` | `libero_10/task_4/seed_20260715` | all four policies failed; repeated on the same weak task |
| `3` | `libero_spatial/task_4/seed_20260712` | all four policies failed; task failed on multiple reset seeds |
| `4` | `libero_spatial/task_4/seed_20260713` | all four policies failed; repeated same task |
| `5` | `libero_spatial/task_4/seed_20260714` | all four policies failed; repeated same task |

## Interpretation

No `target_or_object_selection`, `grasp_approach`, `gripper_timing_or_contact`, `object_transport`, `placement_or_release`, or `action_chunk_drift` label is asserted from this run. The rollout traces prove success-critical failures exist, but without videos or semantic state traces they do not yet identify a mechanism suitable for a new method.
