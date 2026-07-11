# Closed-Loop Failure Video Inventory

Date: 2026-07-11 KST

## Scope

- selected review episodes: `24`
- maximum allowed episodes: `24`
- full 400-episode sweep rerun: `false`
- policy training: `false`
- policy set: `frozen_base`, `rank4_lora_seed_11`, `rank4_lora_seed_22`, `rank4_lora_seed_33`
- selected hard tasks: `libero_spatial/task_4`, `libero_10/task_4`
- completed video reruns: `24`
- rerun errors: `0`

## Identity And Outcome Check

- all rows preserve the original scaleup suite/task/policy/reset identity: `true`
- original-vs-rerun success matches: `16/24`
- original-vs-rerun success flips: `8/24`
- visual evidence status: bounded same-identity rerun evidence, not exact original-frame replay
- contact sheets: `runs/closed_loop_failure_visual_review/contact_sheets`

Outcome flips:

- `libero_spatial/task_4/seed_20260712`: original all-policy failure, rerun all-policy success
- `libero_10/task_4/seed_20260713`: original all-policy failure, rerun success for `rank4_lora_seed_22` and `rank4_lora_seed_33`
- `libero_10/task_4/seed_20260712`: original `frozen_base` success, rerun failure
- `libero_10/task_4/seed_20260714`: original `rank4_lora_seed_11` success, rerun failure

## Existing Video Search

No tracked failure videos were present in the prior 400-episode scaleup. This bounded rerun writes videos under the gitignored `runs/closed_loop_failure_visual_review/videos` directory.

## Selected Episodes

| Episode | Role | Original success | Rerun success | Video |
| --- | --- | ---: | ---: | --- |
| `frozen_base|libero_spatial|task_4|seed_20260712` | `spatial_all_policy_failure` | `False` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/frozen_base/libero_spatial/task_4_seed_20260712.mp4` |
| `rank4_lora_seed_11|libero_spatial|task_4|seed_20260712` | `spatial_all_policy_failure` | `False` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_11/libero_spatial/task_4_seed_20260712.mp4` |
| `rank4_lora_seed_22|libero_spatial|task_4|seed_20260712` | `spatial_all_policy_failure` | `False` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_22/libero_spatial/task_4_seed_20260712.mp4` |
| `rank4_lora_seed_33|libero_spatial|task_4|seed_20260712` | `spatial_all_policy_failure` | `False` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_33/libero_spatial/task_4_seed_20260712.mp4` |
| `frozen_base|libero_spatial|task_4|seed_20260713` | `spatial_all_policy_failure` | `False` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/frozen_base/libero_spatial/task_4_seed_20260713.mp4` |
| `rank4_lora_seed_11|libero_spatial|task_4|seed_20260713` | `spatial_all_policy_failure` | `False` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_11/libero_spatial/task_4_seed_20260713.mp4` |
| `rank4_lora_seed_22|libero_spatial|task_4|seed_20260713` | `spatial_all_policy_failure` | `False` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_22/libero_spatial/task_4_seed_20260713.mp4` |
| `rank4_lora_seed_33|libero_spatial|task_4|seed_20260713` | `spatial_all_policy_failure` | `False` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_33/libero_spatial/task_4_seed_20260713.mp4` |
| `frozen_base|libero_spatial|task_4|seed_20260714` | `spatial_all_policy_failure` | `False` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/frozen_base/libero_spatial/task_4_seed_20260714.mp4` |
| `rank4_lora_seed_11|libero_spatial|task_4|seed_20260714` | `spatial_all_policy_failure` | `False` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_11/libero_spatial/task_4_seed_20260714.mp4` |
| `rank4_lora_seed_22|libero_spatial|task_4|seed_20260714` | `spatial_all_policy_failure` | `False` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_22/libero_spatial/task_4_seed_20260714.mp4` |
| `rank4_lora_seed_33|libero_spatial|task_4|seed_20260714` | `spatial_all_policy_failure` | `False` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_33/libero_spatial/task_4_seed_20260714.mp4` |
| `frozen_base|libero_spatial|task_4|seed_20260711` | `spatial_matched_success` | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/frozen_base/libero_spatial/task_4_seed_20260711.mp4` |
| `rank4_lora_seed_11|libero_spatial|task_4|seed_20260715` | `spatial_matched_success` | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_11/libero_spatial/task_4_seed_20260715.mp4` |
| `frozen_base|libero_10|task_4|seed_20260713` | `libero10_all_policy_failure` | `False` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/frozen_base/libero_10/task_4_seed_20260713.mp4` |
| `rank4_lora_seed_11|libero_10|task_4|seed_20260713` | `libero10_all_policy_failure` | `False` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_11/libero_10/task_4_seed_20260713.mp4` |
| `rank4_lora_seed_22|libero_10|task_4|seed_20260713` | `libero10_all_policy_failure` | `False` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_22/libero_10/task_4_seed_20260713.mp4` |
| `rank4_lora_seed_33|libero_10|task_4|seed_20260713` | `libero10_all_policy_failure` | `False` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_33/libero_10/task_4_seed_20260713.mp4` |
| `frozen_base|libero_10|task_4|seed_20260715` | `libero10_all_policy_failure` | `False` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/frozen_base/libero_10/task_4_seed_20260715.mp4` |
| `rank4_lora_seed_11|libero_10|task_4|seed_20260715` | `libero10_all_policy_failure` | `False` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_11/libero_10/task_4_seed_20260715.mp4` |
| `rank4_lora_seed_22|libero_10|task_4|seed_20260715` | `libero10_all_policy_failure` | `False` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_22/libero_10/task_4_seed_20260715.mp4` |
| `rank4_lora_seed_33|libero_10|task_4|seed_20260715` | `libero10_all_policy_failure` | `False` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_33/libero_10/task_4_seed_20260715.mp4` |
| `frozen_base|libero_10|task_4|seed_20260712` | `libero10_matched_success` | `True` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/frozen_base/libero_10/task_4_seed_20260712.mp4` |
| `rank4_lora_seed_11|libero_10|task_4|seed_20260714` | `libero10_matched_success` | `True` | `False` | `/mnt/c/Users/jiheo/tca_map/runs/closed_loop_failure_visual_review/videos/rank4_lora_seed_11/libero_10/task_4_seed_20260714.mp4` |
