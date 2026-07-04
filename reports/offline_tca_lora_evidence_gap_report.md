# Offline TCA-Map / LoRA Evidence Gap Report

This report consolidates the real-LIBERO offline proxy evidence ladder and the remaining blockers after checkpoint/task provenance resolution.

It is intentionally not a paper-grade result. It is not standard success, not benchmark rollout success, and not SOTA evidence.

Included arms:

- ActionMap head-only,
- TCA-Map head-only,
- TCA-Map + Distributional TCA-Select,
- ActionMap + LoRA,
- TCA-Map + LoRA,
- TCA-Map + LoRA + Distributional TCA-Select.

Required interpretation:

- offline proxy improvements are useful for method debugging,
- learned-policy rollout with the current base checkpoint remains blocked by checkpoint/action provenance,
- paper claims require valid simulator rollout evidence, comparable baselines, compute reporting, and no privileged inference.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\124_generate_offline_evidence_gap_report.ps1
```
