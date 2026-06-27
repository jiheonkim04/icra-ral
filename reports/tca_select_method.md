# TCA-Map + TCA-Select Method

## Why TCA-Select Is Needed

TCA-Map alone can look like ActionMap plus a target head. That is a real Reviewer #2 risk. A target heatmap and target-conditioned action heatmap are useful, but the method needs a stronger inference-time mechanism that uses the geometry of those heatmaps.

TCA-Select is that mechanism. It is an inference-time selection component for target-conditioned action decoding.

## Core Idea

TCA-Select samples `K` candidate actions from the target-conditioned action heatmap and selects the candidate using internal target/action consistency signals.

Default settings:

- `K=4`.
- `temperature=0.5`.
- No external verifier.
- No privileged simulator state at inference.

The selector uses only information already produced by the policy head:

- target heatmap confidence,
- action heatmap confidence,
- target-conditioned action geometry,
- condition sensitivity between full-instruction and masked/ablated heatmaps when available.

## What TCA-Select Must Not Use

TCA-Select must not use:

- external verifiers,
- language-model judges,
- simulator object state,
- oracle target labels,
- privileged ground-truth poses,
- post-hoc rollout success information.

Privileged simulator labels remain allowed only for training supervision, evaluation metrics, and oracle ablations, never default inference.

## Difference From MG-Select

TCA-Select is distinct from MG-Select.

MG-Select uses autoregressive action-token condition-masking KL. It is framed around tokenized action generation and condition masking over the autoregressive distribution.

TCA-Select uses continuous/voxel action heatmaps and target-conditioned action geometry. It samples candidates from a target-conditioned action heatmap and scores them using internal heatmap consistency and target/action alignment signals.

The distinction matters because TCA-Select is tied to the ActionMap/TCA-Map representation rather than a generic language/action-token scoring trick.

## Inference Flow

1. Predict a target heatmap from observation and instruction.
2. Predict a target-conditioned action heatmap.
3. Sample top-`K` candidate actions from the action heatmap.
4. Score each candidate with target/action consistency.
5. Optionally score condition sensitivity using a masked-instruction heatmap.
6. Select the highest-scoring candidate.
7. Return the selected action and diagnostics.

## Publishable Claim Role

The publishable low-compute claim should not be that LoRA or frozen smoke is novel. The claim should be:

> TCA-Map improves target-conditioned action decoding under strict compute constraints, preserving standard performance while improving counterfactual target grounding over ActionMap and native heads.

TCA-Select strengthens that claim by making inference explicitly choose actions that are both likely under the action heatmap and internally consistent with the predicted target grounding.
