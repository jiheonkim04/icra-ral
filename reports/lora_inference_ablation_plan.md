# LoRA and Inference Ablation Plan

## Strategy

LoRA and QLoRA are required experimental tracks after head-only validation, not the core novelty. Their role is to test compute-efficient adaptation arms for TCA-Map. The main method contribution is Distributional TCA-Select: verifier-free test-time selection using target-conditioned action heatmap distributions.

Default training remains:

- SmolVLA-first,
- frozen backbone,
- cached features,
- head-only TCA-Map,
- low-resolution or coarse-to-fine heatmaps,
- no OpenVLA-OFT large local experiments.

## Separation of Effects

Experiments must separate three gains:

1. **Head-only gain:** native head vs ActionMap/TCA-Map heads with frozen backbone.
2. **LoRA gain:** head-only TCA-Map vs TCA-Map with small LoRA/QLoRA adapters.
3. **Inference-selection gain:** TCA-Map without selection vs heuristic TCA-Select vs Distributional TCA-Select.

Do not report a single combined number as if it explains all three effects.

## Table Template

| Variant | Backbone | Trainable modules | Selector | Offline proxy | Counterfactual proxy | Wrong-target proxy | Latency | Max VRAM | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | Frozen | Native head only or frozen head | None | TBD | TBD | TBD | TBD | TBD | Baseline only |
| ActionMap | Frozen | Action heatmap head | Heatmap argmax/top-k | TBD | TBD | TBD | TBD | TBD | Required baseline |
| TCA-Map | Frozen | Target + action heatmap heads | None or argmax | TBD | TBD | TBD | TBD | TBD | Tests target conditioning |
| TCA-Map + heuristic TCA-Select | Frozen | Target + action heatmap heads | Geometry/consistency heuristic | TBD | TBD | TBD | TBD | TBD | Ablation against distributional selector |
| TCA-Map + Distributional TCA-Select | Frozen | Target + action heatmap heads | KL/JS/margin selector | TBD | TBD | TBD | TBD | TBD | Main method |
| ActionMap + LoRA | Frozen backbone plus small adapters | Action projection and allowed adapters | None | TBD | TBD | TBD | TBD | TBD | Required PEFT baseline |
| TCA-Map + LoRA | Frozen backbone plus small adapters | Target fusion, action projection, allowed adapters | None | TBD | TBD | TBD | TBD | TBD | Required PEFT target-conditioned arm |
| TCA-Map + Distributional TCA-Select + LoRA | Frozen backbone plus small adapters | Target fusion, action projection, allowed adapters | KL/JS/margin selector | TBD | TBD | TBD | TBD | TBD | Required PEFT + selector arm |

## Go / No-Go for LoRA

Run LoRA as a required track only if the bounded policy is satisfied:

- local memory checks pass,
- trainable parameter count remains small,
- adapters are limited to target fusion layers, action head projection, or explicitly configured small adapter layers.

Do not enable full backbone fine-tuning. QLoRA requires an explicit config and is treated as a memory-saving engineering option, not the novelty.
