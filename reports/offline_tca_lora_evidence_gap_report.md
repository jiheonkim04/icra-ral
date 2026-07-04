# Offline TCA-Map / LoRA Evidence Gap Report

This report consolidates the real-LIBERO offline proxy evidence ladder and the remaining blockers after checkpoint/task provenance resolution.

It is intentionally not a paper-grade result. It is not standard success, not benchmark rollout success, and not SOTA evidence.

Included arms:

- ActionMap head-only,
- TCA-Map head-only,
- TCA-Map + Distributional TCA-Select,
- ActionMap + LoRA,
- TCA-Map + LoRA,
- TCA-Map + LoRA + Distributional TCA-Select,
- bounded ActionMap + LoRA scale-up, when `reports\bounded_lora_offline_scaleup_report.json` exists,
- bounded TCA-Map + LoRA scale-up, when that report exists,
- bounded TCA-Map + LoRA + Distributional TCA-Select scale-up, when that report exists,
- Distributional TCA-Select ambiguity stress, when `reports\tca_select_ambiguity_stress_report.json` exists.

Required interpretation:

- offline proxy improvements are useful for method debugging,
- bounded LoRA scale-up evidence remains offline proxy evidence only,
- TCA-Select ambiguity-stress evidence is selection-specific offline proxy evidence only,
- learned-policy rollout with the current base checkpoint remains blocked by checkpoint/action provenance,
- paper claims require valid simulator rollout evidence, comparable baselines, compute reporting, and no privileged inference.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\124_generate_offline_evidence_gap_report.ps1
```

Current local result:

- evidence row count: `10`
- `tca_select_ambiguity_stress_included=true`
- TCA-Select ambiguity stress selected wrong-target proxy rate: `0.0`
- TCA-Select ambiguity stress selected action L1: `0.0`
- TCA-Select ambiguity stress wrong-target delta vs top heatmap: `-1.0`
- TCA-Select ambiguity stress action L1 delta vs top heatmap: `-0.164299`
- learned-policy rollout scaling: `false`
- paper claim readiness: `false`
