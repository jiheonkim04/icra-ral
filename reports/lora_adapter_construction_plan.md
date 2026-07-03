# LoRA Adapter Construction Plan

## Purpose

This is a planning-only step for the required LoRA/QLoRA experiment tracks. It does not construct adapters, import SmolVLA, load models, train, rollout, download assets, execute OpenVLA-OFT, or make paper claims.

The goal is to define the safe construction boundary before any tiny LoRA smoke exists.

## Command

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\32_plan_lora_adapter_construction.ps1
```

It writes an ignored runtime report:

```text
reports\lora_adapter_construction_plan_report.json
```

## Required Tracks

The plan covers:

- ActionMap + LoRA,
- TCA-Map + LoRA,
- TCA-Map + LoRA + Distributional TCA-Select,
- TCA-Map + QLoRA + Distributional TCA-Select if memory/tooling allows.

LoRA/QLoRA remain supporting adaptation arms. The novelty remains TCA-Map, counterfactual target/action consistency, and Distributional TCA-Select.

## Construction Boundary

Allowed later:

- adapter construction from explicit config,
- target fusion layers,
- action head projection,
- small adapter layers,
- frozen backbone except adapter weights,
- batch size 1,
- max 100 tiny-smoke steps,
- max 15 minutes,
- max 14GB VRAM.

Forbidden:

- full backbone fine-tuning,
- OpenVLA-OFT execution,
- simulator or rollout,
- dataset download,
- CUDA/PyTorch major changes,
- training beyond tiny smoke.

## Current Status

This plan is safe to run now because it is check-only. Actual LoRA adapter construction or tiny LoRA smoke requires a later bounded scaffold and must still respect all hard-stop gates.
