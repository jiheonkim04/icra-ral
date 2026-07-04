# Bounded LoRA / Offline Proxy Scale-Up Plan

This planning-only gate follows the offline evidence gap report.

Future runner budget:

- CPU-only by default,
- at most 16 counterfactual pairs,
- at most 64 samples,
- at most 64 optimization steps,
- LoRA rank 4,
- frozen base weights,
- no full fine-tuning,
- no simulator rollout,
- no model loading or heavy VLA import,
- no GPU job,
- no OpenVLA-OFT,
- no paper claim.

The future runner must require `ALLOW_TINY_TRAINING=1` and must continue labeling outputs as offline proxy diagnostics only.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\125_plan_bounded_lora_offline_scaleup.ps1
```
