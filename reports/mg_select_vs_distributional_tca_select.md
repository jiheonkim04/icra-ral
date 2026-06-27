# MG-Select vs Distributional TCA-Select

## Core Difference

MG-Select uses autoregressive action-token distributions and condition-masking KL. It asks whether an action-token policy changes its token distribution when relevant conditioning is masked.

Distributional TCA-Select uses continuous or voxel action heatmap distributions and target heatmap distributions. It asks whether a target-conditioned heatmap decoder produces action geometry that is internally consistent with the intended target and separated from counterfactual targets.

## What We Can Claim

Distributional TCA-Select extends verifier-free distributional test-time selection to target-conditioned continuous action heatmap decoders.

The claim is not that Distributional TCA-Select dominates MG-Select on autoregressive VLAs. MG-Select is naturally matched to autoregressive action-token policies. Distributional TCA-Select is naturally matched to heatmap or continuous-action decoders that expose action distributions over spatial/action candidates.

## Signal Mapping

| MG-Select-style signal | Distributional TCA-Select analogue |
| --- | --- |
| Autoregressive action-token distribution | Low-resolution action heatmap distribution |
| Condition-masked token KL | Condition-masked action heatmap KL |
| Token sequence likelihood | Candidate log probability under action heatmap |
| Negative prompt token distribution | Counterfactual-instruction action heatmap distribution |
| Verifier-free selection | Verifier-free heatmap candidate selection |

## Required Boundaries

Distributional TCA-Select must not use:

- an external verifier,
- privileged simulator state,
- oracle object pose,
- rollout success during inference,
- hidden labels at inference.

It may use distributions produced by the model heads under full, masked, and counterfactual instructions.

## Required Evaluation Set

A fair low-compute evaluation should compare:

1. native head,
2. ActionMap,
3. ActionMap + counterfactual augmentation,
4. TCA-Map,
5. TCA-Map + heuristic TCA-Select,
6. TCA-Map + Distributional TCA-Select.

If LoRA/QLoRA is added, report it as a separate support-tool ablation rather than merging it into the main novelty claim.

## Reporting Rule

Any SOTA-style claim must be restricted to low-compute target-conditioned action decoding or counterfactual robustness unless full standard baselines are directly reproduced with simulator rollouts. Offline proxy scores are not standard success.
