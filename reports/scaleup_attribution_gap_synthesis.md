# Scale-Up Attribution Gap Synthesis

This report-only step synthesizes the scale-up-aware offline evidence table after bounded LIBERO offline LoRA scale-up.

It is not standard success, not rollout success, not SOTA evidence, and not paper-grade evidence.

The synthesis is implemented by:

- `scripts\127_synthesize_scaleup_attribution_gaps.ps1`
- `tca_map.smolvla.scaleup_attribution_gap_synthesis`

Purpose:

- separate TCA-Map target-conditioning gains from LoRA adaptation gains,
- record that Distributional TCA-Select currently adds no extra LoRA proxy gain in this bounded runner,
- include the offline TCA-Select ambiguity stress-test result when `reports\tca_select_ambiguity_stress_report.json` is present,
- identify what must be improved before a publishable attribution claim,
- keep current-checkpoint learned-policy rollout scaling and paper claims blocked.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\127_synthesize_scaleup_attribution_gaps.ps1
```

Current interpretation:

- the bounded LoRA runner still shows zero extra Distributional TCA-Select delta in that specific proxy,
- the ambiguity stress test separately shows selection-specific proxy gain against a top-heatmap baseline,
- both results remain offline proxy evidence only,
- no standard success, rollout success, benchmark claim, or paper-grade claim is unlocked.
