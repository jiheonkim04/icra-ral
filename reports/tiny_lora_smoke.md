# Tiny LoRA Smoke

## Purpose

This bounded local pilot runner exercises the required LoRA experiment track on cached/dummy features only.

It is an offline proxy diagnostic. It is not standard success, not rollout success, not a paper-grade result, and not a SOTA claim.

## Command

Run:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\37_tiny_lora_smoke.ps1 -PrepareDummyCache
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

It writes an ignored runtime report:

```text
reports\tiny_lora_smoke_report.json
```

## Bounds

The runner is standing-approved only inside these limits:

- cached/dummy features only,
- frozen backbone,
- train LoRA adapter weights only,
- no full fine-tuning,
- no SmolVLA or OpenVLA model load,
- no model inference,
- no dataset download,
- no GPU job,
- no rollout,
- no simulator,
- no OpenVLA-OFT,
- no paper-grade claim,
- max 100 steps,
- max 200 samples,
- max 15 minutes,
- rank at most 16.

The runner requires `ALLOW_TINY_TRAINING=1` for the bounded adapter update and refuses download, heavy-import, GPU-training, rollout, runtime-install, single-sample-inference, and cloud-handoff gates.

## Arms

The smoke reports these required local LoRA arms:

- `actionmap_lora`,
- `tca_map_lora`,
- `tca_map_lora_distributional_select`.

All metrics are offline proxy metrics and must stay separate from future simulator rollout metrics.
