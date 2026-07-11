# RA-L Method Kill Criteria

Date: 2026-07-11 KST

Current candidate: `NONE`

## Active Kill Results

| Criterion | Status | Evidence |
| --- | --- | --- |
| No single repeated visual mechanism | `triggered` | Spatial drawer/bowl stable-grasp failure and `libero_10` multi-object long-horizon failure are different. |
| Fewer than three independent reset seeds for one-task-only mechanism | `triggered` | Spatial visual rerun failures occur on seeds `20260713` and `20260714`; seed `20260712` failed originally but reran success. |
| No second task with same mechanism | `triggered` | `libero_10/task_4` does not show drawer/bowl extraction. |
| Rerun identity instability weakens causal evidence | `triggered` | `8/24` same-identity reruns changed success status. |
| Obvious method is killed by recent VLA work | `triggered` | Confidence, verification, corrective replanning, adaptive chunking, progress, failure-negative learning, and adapter-routing routes are already occupied. |
| Second backbone not shown | `triggered` | No OpenVLA-OFT or other second-backbone evidence exists. |
| Second benchmark not shown | `triggered` | No LIBERO-Plus, LIBERO-Occ, RoboTwin, or CALVIN evidence exists. |
| Simple baseline not ruled out | `triggered` | Frequent replanning, smaller fixed chunks, and verification/correction proxies were not tested. |

## What Would Reopen The Gate

All of the following would be required:

- video evidence for one mechanism in at least two tasks or three independent reset seeds;
- rerun stability high enough that original failure identity is not ambiguous;
- a method idea that is not confidence, verification, progress, chunking, failure-negative learning, routing, or generic replanning;
- a predeclared second-backbone plan;
- a predeclared second-benchmark plan;
- a simple-baseline plan that can kill the method cheaply.

## Current Kill Decision

The RA-L method route is killed for this pass. No implementation prompt is authorized.
