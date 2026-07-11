# ECHO-VLA Full RA-L Experiment Matrix

Date: 2026-07-11 KST

This is a future plan only. Nothing was run.

## Backbones

1. Official SmolVLA-LIBERO.
2. Quantized OpenVLA-OFT INT4.

Quantization, SmolVLA, and OpenVLA-OFT are experimental backbones only, not contributions.

## Benchmarks / Conditions

1. Standard LIBERO suites most relevant to effect credit:
   - `libero_spatial`
   - `libero_object`
   - `libero_goal`
   - selected `libero_10` tasks for multi-object phase effects

2. Controlled execution perturbation condition:
   - small action noise,
   - one-step latency/delay,
   - perturbed initial object poses within official reset bounds when available,
   - same perturbation protocol for all policies.

Alternative second condition if implementation support is cleaner: LIBERO-Plus initial-state/action-relevant perturbations. LIBERO-Occ is not first choice because ECHO's primary claim is action-effect credit, not perception completion.

## Primary Metric

Closed-loop task success.

## Secondary Metrics

- task-balanced success,
- predicate-effect success rate,
- phase-required effect F1,
- recovery from controlled perturbation,
- success degradation slope by perturbation level,
- number of VLA forward passes,
- number of ECHO forward passes,
- latency,
- VRAM,
- calibration of predicted effect probabilities,
- per-task results with confidence intervals.

## Required Comparisons

- frozen backbone,
- standard backbone adaptation if relevant,
- simple heuristic predicate-progress baseline,
- progress/value head baseline,
- Pre-VLA-style validity/advantage head baseline,
- verifier-style reranker baseline,
- ECHO without counterfactual ranking,
- ECHO without phase conditioning,
- ECHO without predicate distillation,
- ECHO full.

## Statistical Protocol

- predeclare task IDs, initial states, rollout seeds, and perturbation levels,
- use multiple rollout seeds,
- report confidence intervals,
- report per-task and task-balanced results,
- no best-seed selection,
- no dropping negative cases,
- report all failed tasks and failure categories.

## Intended SOTA Axis

The intended SOTA axis is not standard LIBERO average. It is:

`closed-loop task success under controlled execution perturbation with calibrated phase-required action-effect prediction`.

The method should claim a robustness-success and effect-calibration axis only if it beats the strongest recent feasible baselines on both backbones.

## Full Matrix

| Factor | Values |
| --- | --- |
| Backbones | SmolVLA, OpenVLA-OFT INT4 |
| Conditions | standard LIBERO, controlled execution perturbation |
| Tasks | balanced subsets from spatial/object/goal/10, expanded after first gate |
| Policies | frozen, standard adaptation, heuristic, progress/value, validity/advantage, verifier-rerank, ECHO ablations, ECHO full |
| Metrics | success, effect F1, calibration, perturbation degradation, latency, VRAM, forward passes |
| Seeds | multiple rollout seeds, fixed before run |
| Reporting | per-task, aggregate, confidence intervals, negative cases |

## Full-Paper Success Bar

ECHO-VLA is paper-credible if:

- first prototype passes,
- both backbones show positive closed-loop success gains,
- controlled perturbation condition shows at least `5` absolute task-balanced success points over the strongest simple baseline,
- effect prediction is calibrated enough to explain the gains,
- latency/VRAM overhead is modest and reported.
