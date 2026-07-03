# Tiny LoRA Comparison

## Purpose

This bounded local pilot report compares the existing tiny LoRA smoke outputs:

- `actionmap_lora`,
- `tca_map_lora`,
- `tca_map_lora_distributional_select`.

It is an offline proxy diagnostic only. It is not standard success, not rollout success, not paper-grade evidence, and not a SOTA claim.

## Command

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\38_compare_tiny_lora_pilot.ps1
```

It writes ignored runtime reports:

```text
reports\tiny_lora_comparison_report.json
reports\tiny_lora_comparison_report.md
```

## Bounds

The comparison is inside the risk-assessed pilot envelope because it:

- reads an existing bounded tiny LoRA smoke report,
- uses offline proxy metrics only,
- does not download assets,
- does not run GPU jobs,
- does not train,
- does not import heavy VLA models,
- does not load models,
- does not run model inference,
- does not rollout or execute simulators,
- does not execute OpenVLA-OFT,
- does not make paper claims.

The comparison refuses execution gates, including `ALLOW_TINY_TRAINING`, because it is analysis-only.
