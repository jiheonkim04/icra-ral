# Reviewer #2 Risk: TCA-Select + LoRA Low-Compute Method

## Is This Just ActionMap Plus A Target Head?

It will look that way if TCA-Select is absent or weak. TCA-Select must show that target-conditioned heatmaps are not merely an auxiliary target classifier, but a geometry-aware action candidate selection mechanism.

## Does LoRA Make The Paper Less Novel?

LoRA helps efficiency but is not novel by itself. If LoRA is the only reason performance improves, the method claim weakens. LoRA/QLoRA must be reported as optional support, with head-only and selection-only ablations separated.

## What Baseline Kills The Paper?

ActionMap + counterfactual augmentation under the same frozen SmolVLA/cached-feature budget is the killer baseline. If it matches TCA-Map + TCA-Select on wrong-target rate and counterfactual success, the claimed method value is thin.

## What Must The Ablation Prove?

The ablation must show incremental value from:

1. target-conditioned heatmap decoding,
2. TCA-Select inference-time selection,
3. LoRA/QLoRA only if used.

TCA-Select should improve counterfactual selection or wrong-target rate without harming standard performance beyond 1-2 percentage points.

## What Is Invalid?

Invalid claims:

- OpenVLA-OFT SOTA without direct reproduction.
- Frozen OpenVLA-OFT smoke as a result.
- LoRA as the main novelty.
- Offline proxy as standard manipulation success.
- Any inference result using privileged simulator state.

## What Would Make This Publishable?

A credible package would show:

- SmolVLA-native vs ActionMap vs ActionMap+augmentation vs TCA-Map vs TCA-Map+TCA-Select.
- At least one small simulator rollout benchmark.
- +10 percentage point counterfactual improvement.
- 20 percent relative wrong-target reduction.
- <=1-2 percentage point standard-performance degradation.
- compute table and trainable parameter counts.
- clear no-privileged-inference audit.
