# ECHO-VLA Paper Contribution Outline

Date: 2026-07-11 KST

This is a provisional outline only, not a complete paper.

## Provisional Title

`ECHO-VLA: Counterfactual Action-Effect Credit for Closed-Loop Vision-Language-Action Manipulation`

## Contributions

1. Analysis contribution: identify the action-objective versus closed-loop-success mismatch shared across recent VLA action generation, progress, confidence, verification, and robustness methods.
2. Technical method contribution: introduce a phase-conditioned counterfactual effect objective estimating `P(effect | do(action_chunk), observation, instruction, phase)` and use it to train/guide VLA action chunks.
3. Experimental validation contribution: show improved closed-loop success under standard LIBERO and controlled execution perturbations on SmolVLA and OpenVLA-OFT INT4, with effect calibration, latency, VRAM, and ablations.

## Abstract Skeleton

Vision-language-action models have improved rapidly, but their training objectives still largely reward matching expert actions rather than causing task-relevant physical effects. We propose ECHO-VLA, a counterfactual effect-credit method that learns phase-conditioned predicate effects of action chunks from privileged training labels and deploys without privileged state. ECHO-VLA augments existing VLAs with an effect model trained on matched counterfactual chunks and uses predicted effect advantage to guide action selection. Experiments will evaluate closed-loop task success under standard and perturbed LIBERO conditions across SmolVLA and OpenVLA-OFT INT4, with comparisons to progress, confidence, verification, and heuristic baselines.

## Introduction Argument

- VLAs are evaluated by closed-loop success.
- Most action-learning objectives optimize action likelihood, regression, or flow loss.
- Recent fixes target verification, confidence, progress, correction, and robustness.
- These help, but they do not directly answer which action chunk caused the physical predicate transition needed for the current phase.
- ECHO-VLA makes action-effect credit explicit and testable.

## Method Section Outline

1. Problem setup: VLA action chunks and task predicate effects.
2. Phase-conditioned predicate effect representation.
3. Counterfactual effect labels and matched contexts.
4. Effect model and training objective.
5. Inference-time effect-guided action selection.
6. Non-privileged deployment via visual predicate distillation.

## Experiment Section Outline

1. First decisive prototype.
2. Full two-backbone validation.
3. Standard LIBERO and controlled perturbation condition.
4. Baselines and ablations.
5. Metrics: success, effect F1, calibration, latency, VRAM, forward passes.
6. Negative cases and failure analysis.

## Main Table Design

Rows: frozen backbone, standard adaptation, heuristic, progress/value, validity/advantage, verifier rerank, ECHO ablations, ECHO full.

Columns: standard success, perturbation success, effect F1, calibration ECE, latency, VRAM, forward passes.

Separate tables for SmolVLA and OpenVLA-OFT INT4, plus combined task-balanced average.

## Main Figure Design

Figure 1: method diagram showing observation/instruction/phase, VLA chunk proposal, counterfactual effect model, phase-required effect score, and non-privileged deployment.

Figure 2: effect-credit examples showing two action chunks with similar L1/likelihood but different predicate effects.

Figure 3: perturbation degradation curve comparing ECHO to strongest baseline.

## Ablation Table Design

Rows:

- ECHO full,
- no counterfactual ranking,
- no phase conditioning,
- no predicate distillation,
- scalar progress target instead of effect vector,
- single candidate only,
- oracle predicate diagnostic only.

Columns:

- success,
- effect F1,
- calibration,
- intervention count,
- latency.
