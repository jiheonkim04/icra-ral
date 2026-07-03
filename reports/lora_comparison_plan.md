# LoRA Comparison Plan

## Purpose

This planning-only report defines the required LoRA/QLoRA comparison matrix. It does not train, construct adapters, import heavy VLA models, load models, run inference, download assets, use GPU, rollout, execute simulators, execute OpenVLA-OFT, or make paper claims.

LoRA/QLoRA are required experimental tracks, but they are not the main novelty. The main novelty remains target-conditioned action heatmaps, counterfactual target/action consistency, and Distributional TCA-Select.

## Required LoRA Comparisons

The required LoRA-side comparisons are:

- ActionMap + LoRA vs TCA-Map + LoRA,
- TCA-Map + LoRA vs TCA-Map + LoRA + Distributional TCA-Select,
- TCA-Map head-only vs TCA-Map + LoRA,
- ActionMap head-only vs ActionMap + LoRA,
- QLoRA variant if memory/tooling allows.

The comparison must separate:

- head architecture gain,
- LoRA adaptation gain,
- inference-time Distributional TCA-Select gain.

## Fairness Requirements

All LoRA comparison arms must use the same:

- SmolVLA checkpoint family,
- cached feature contract or synthetic interface contract,
- frozen backbone policy,
- max 100 tiny-smoke steps for local smoke,
- max 15 minutes for local smoke,
- max 14GB VRAM target,
- batch size 1,
- offline proxy metric schema.

No comparison may use privileged simulator state at inference.

## Metrics

Minimum local/offline metrics:

- action L1 / MSE,
- action voxel hit rate or distance-to-expert voxel,
- target heatmap top-1 / top-k accuracy,
- wrong-target proxy rate,
- counterfactual target/action separation margin,
- nuisance stability score,
- latency,
- max GPU memory,
- trainable parameter estimate.

Paper-grade rollout metrics remain blocked until simulator paths exist and rollout risk assessment passes.

## Command

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\34_plan_lora_comparison.ps1
```

It writes an ignored runtime report:

```text
reports\lora_comparison_plan_report.json
```
