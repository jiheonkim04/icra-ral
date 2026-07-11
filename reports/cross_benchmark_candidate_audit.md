# Cross-Benchmark Candidate Audit

Date: 2026-07-11 KST

Objective: select one second benchmark or perturbation suite for mechanism generality after a second backbone reproduces a mechanism. No benchmark assets were downloaded.

## Candidate Order

The requested order was followed:

1. LIBERO-PRO
2. LIBERO-Occ only if occlusion is visually relevant
3. RoboTwin 2.0 or CALVIN only if the first two are infeasible

## LIBERO-PRO

| Field | Audit |
| --- | --- |
| Official source | `https://github.com/Zxy-MLlab/LIBERO-PRO` |
| Paper | `https://arxiv.org/abs/2510.03827` |
| Dataset | `https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro` |
| License/access | MIT, public, non-gated |
| Asset size | Hugging Face API reports `1,090,523` bytes for uploaded BDDL/init files |
| Runtime | Developed on original LIBERO and uses the same LIBERO runtime environment |
| Supported perturbations | object, position/swap, semantic/language, task, environment |
| Task success metric | binary task success under LIBERO-style rollout |
| SmolVLA compatibility | Likely compatible if BDDL/init files are placed under the existing LIBERO structure; no action-semantics change expected |
| OpenVLA-OFT compatibility | Supported by LIBERO-PRO README through modifications to OpenVLA-OFT `run_libero_eval.py`; requires protocol integration |
| Stable-grasp testability | Yes, especially position/object perturbations around drawer/bowl manipulation |
| Long-horizon testability | Yes, especially task/language/environment or position perturbations on LIBERO-10 |
| Setup cost | Small asset download plus repo install/integration; no large model download beyond selected second VLA |

LIBERO-PRO is selected because it can test both observed mechanisms without changing action semantics and because it is directly tied to LIBERO/OpenVLA-style evaluation.

## LIBERO-Occ

Not selected. Occlusion was not the visually identified failure driver in the reviewed videos. The spatial failure was a drawer/bowl extraction and stable-grasp issue, not an occlusion-specific issue.

## RoboTwin 2.0 / CALVIN

Not selected. LIBERO-PRO is feasible enough and better aligned with the current tasks, reset semantics, and both selected backbones.

## State 2 Decision

Selected second benchmark: `LIBERO-PRO`

Decision: `SECOND_BENCHMARK_READY_AFTER_SECOND_BACKBONE`

The benchmark itself is not the active blocker. The active blocker is the selected second-backbone checkpoint download/hardware path.
