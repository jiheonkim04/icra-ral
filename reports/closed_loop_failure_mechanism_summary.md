# Closed-Loop Failure Mechanism Summary

Date: 2026-07-11 KST

Final visual source: bounded same-identity rerun videos under `runs/closed_loop_failure_visual_review/videos`.

## Scope

- videos reviewed: `24`
- maximum allowed rerun episodes: `24`
- training/tuning: `false`
- full 400-episode rerun: `false`
- policies: `frozen_base`, `rank4_lora_seed_11`, `rank4_lora_seed_22`, `rank4_lora_seed_33`
- hard slices reviewed: `libero_spatial/task_4`, `libero_10/task_4`
- rerun errors: `0`
- same suite/task/policy/reset identity preserved: `true`
- original-vs-rerun success matches: `16/24`
- original-vs-rerun success flips: `8/24`

The success flips matter: these videos are bounded same-identity reruns, not exact original-frame replays. They can support visible mechanism review only for rows that reran as failures.

## Mechanism A: Spatial Drawer Bowl Stable-Grasp Failure

Task: `libero_spatial/task_4`

Instruction: `pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate`

Visible failure phase: `stable_grasp`, with `contact_transition` as the adjacent physical phase.

Evidence:

- rerun failures on reset seeds `20260713` and `20260714`
- all four policies failed on those two reset seeds
- target/object selection appears correct: the EEF approaches the top drawer and black bowl
- the bowl remains in the drawer after approach/contact; no stable extracted grasp is established
- transport and placement never start
- matched rerun successes on `frozen_base/20260711` and `rank4_lora_seed_11/20260715` show the same task can succeed when the bowl is extracted cleanly

Gate status: mechanism is visually strong inside one task, but it has only two independent rerun-failure reset seeds. Original seed `20260712` failed for all policies in the 400-episode scaleup but flipped to success in all four video reruns, so it is not counted as visual failure evidence.

## Mechanism B: Libero-10 Multi-Object Long-Horizon Failure

Task: `libero_10/task_4`

Instruction: `put the white mug on the left plate and put the yellow and white mug on the right plate`

Visible failure phase: `long_horizon_compounding`, with incomplete `placement_alignment` / `release_transition` evidence.

Evidence:

- reset seed `20260715` reran as failure for all four policies
- reset seed `20260713` reran as failure for `frozen_base` and `rank4_lora_seed_11`, while `rank4_lora_seed_22` and `rank4_lora_seed_33` flipped to success
- two matched-success selections, `frozen_base/20260712` and `rank4_lora_seed_11/20260714`, flipped to failure on rerun
- the visible behavior is a partial multi-object sequence: the EEF reaches/manipulates mugs, but the final two-plate arrangement is not achieved before timeout

Gate status: this is not the same physical failure as the drawer-bowl stable-grasp problem. It is also less phase-specific because the contact sheets do not isolate one contact or release event as the decisive first divergence.

## Cross-Task Mechanism Gate

The hard mechanism gate is not passed.

- Same mechanism in at least two tasks: `false`
- Same visually supported mechanism in at least three independent reset seeds within a hard task: `false`
- Material effect on success: `true` for both observed mechanisms, but not enough for method selection

The strongest mechanism is the spatial drawer/bowl stable-grasp/extraction failure. It is still not method-ready because the visual rerun evidence has only two independent failure reset seeds and no second-task/backbone support.

## Consequence

No method should be implemented from this review. The review converts the previous `ambiguous_or_unclassified` failures into two plausible failure families, but it does not identify a single repeated, success-critical, literature-surviving mechanism suitable for an RA-L method.
