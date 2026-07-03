# LoRA Tiny Smoke Scaffold

## Purpose

This is a dry-run scaffold for the required LoRA/QLoRA experiment tracks. It does not construct adapters, train, import SmolVLA, load models, run inference, download assets, use GPU, rollout, execute simulators, execute OpenVLA-OFT, or make paper claims.

The goal is to define the bounded tiny-smoke envelope before a future runner is allowed to perform any adapter update.

## Command

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\33_plan_lora_tiny_smoke.ps1
```

It writes an ignored runtime report:

```text
reports\lora_tiny_smoke_scaffold_report.json
```

## Future Tiny-Smoke Envelope

A future LoRA tiny smoke may be considered only if it stays inside all of these bounds:

- frozen backbone,
- train LoRA adapter weights only,
- no full fine-tuning,
- cached/dummy features or synthetic interface data only,
- batch size 1,
- max 100 tiny-smoke steps,
- max 15 minutes,
- max 14GB VRAM target,
- no rollout,
- no simulator,
- no OpenVLA-OFT,
- no paper-grade claim.

The future execution runner must require `ALLOW_TINY_TRAINING=1` only for that bounded task and must refuse download, heavy-import, GPU-training, rollout, simulator, runtime-install, and OpenVLA-OFT gates.

## Current Status

This scaffold is safe to run now because it is planning-only. It keeps `safe_to_execute_lora_tiny_smoke_now=false` until a separate bounded runner exists and passes the repository self-check gates.
