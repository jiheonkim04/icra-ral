# Final Method Spec: Distributional TCA-Map

## Definition

**Distributional TCA-Map = TCA-Map + Distributional TCA-Select + required LoRA/QLoRA experimental tracks.**

The method is a low-compute target-conditioned action decoder for counterfactual robustness. It keeps the VLA backbone frozen by default, trains a small target-conditioned action heatmap head, and uses a verifier-free distributional inference-time selector.

LoRA/QLoRA are required experimental tracks after head-only validation. They are not the core novelty and must be reported separately from TCA-Map and Distributional TCA-Select gains.

## Decoder

The default decoder predicts a low-resolution target-conditioned action heatmap. The heatmap may be voxelized or otherwise discretized over an action candidate space. Full-resolution voxel heatmaps are out of scope for the local RTX 5080 16GB pilot.

Optional refinement is allowed only as a local coarse-to-fine step:

1. predict a coarse target-conditioned action heatmap,
2. sample or choose a small candidate set around high-probability cells,
3. refine only local neighborhoods,
4. keep memory bounded and report latency/VRAM.

## Distributional TCA-Select

At inference time, Distributional TCA-Select samples or selects the best `K` candidate actions from the target-conditioned action heatmap. The default is:

- `K=4`,
- `temperature=0.5`,
- no external verifier,
- no privileged simulator state,
- no oracle object pose,
- no rollout-time success signal.

The selector scores candidates using only internal model distributions:

- log probability under the full target-conditioned action heatmap,
- KL divergence between full action heatmap and condition-masked action heatmap,
- KL or JS divergence between full action heatmap and counterfactual-instruction negative heatmaps,
- target/action consistency from the target heatmap,
- target heatmap counterfactual margin or target heatmap divergence,
- entropy penalty on uncertain action distributions.

## Required Distributions

### Full action heatmap distribution

The model predicts `p(a | image, instruction, target)` over low-resolution action bins or candidates. This is the base distribution used for candidate sampling and log probability.

### Condition-masked heatmap distribution

The same model interface should support a masked or neutralized instruction/target condition, producing `p(a | image, masked_condition)`. The selector rewards candidates whose action distribution meaningfully changes under target masking.

This is the heatmap analogue of condition-masking distributional selection, but it is not an autoregressive token method.

### Counterfactual-instruction heatmap distribution

For hard negatives, the model predicts action heatmaps under target-swapped or counterfactual instructions. The selector should prefer candidates that remain strong under the intended instruction and become less favored under negative instructions.

### Target heatmap distribution

The target head predicts `p(t | image, instruction)`. Candidate scores may use:

- target heatmap KL/JS against counterfactual target distributions, or
- a target margin between intended target score and hardest negative target score.

The local scaffold implements the margin path first because it is easy to test without heavy dependencies.

## Candidate Score

A default score can be written as:

```text
score(a_k) =
  w_logp * log p_full(a_k)
  + w_mask * D_KL(p_full(a) || p_masked(a))
  + w_neg * mean_j D_JS(p_full(a), p_negative_j(a))
  + w_tgt * target_consistency(a_k)
  + w_margin * target_counterfactual_margin(a_k)
  - w_entropy * H(p_full(a))
```

Implementation details may use KL for masked distributions and JS for hard negatives because JS is symmetric and finite. Any change to this scoring rule must be reported as an ablation.

## No External Verifier

Distributional TCA-Select must not call:

- a separate success classifier,
- a simulator oracle,
- privileged object pose,
- rollout success during inference,
- human labels at inference,
- any off-policy evaluator not produced by the policy/head itself.

Simulator state can be used for training labels, evaluation metrics, and oracle ablations only. Those must be named as such.

## Evaluation Ladder

Minimum comparisons:

1. native head,
2. ActionMap,
3. ActionMap + counterfactual augmentation,
4. TCA-Map without selection,
5. TCA-Map + heuristic TCA-Select,
6. TCA-Map + Distributional TCA-Select,
7. TCA-Map + Distributional TCA-Select + LoRA.
8. TCA-Map + Distributional TCA-Select + QLoRA if memory/tooling allows.

Required metrics:

- offline action L1/MSE,
- action voxel hit rate or distance-to-expert voxel,
- target heatmap top-1/top-k accuracy,
- wrong-target proxy rate,
- counterfactual target/action separation margin,
- nuisance stability score,
- latency,
- max GPU memory,
- trainable parameter count.

Paper-grade standard success requires simulator rollout success and should not be inferred from offline proxies.
