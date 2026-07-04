# Scale-Up Attribution Gap Synthesis

This report-only step synthesizes the scale-up-aware offline evidence table after bounded LIBERO offline LoRA scale-up.

It is not standard success, not rollout success, not SOTA evidence, and not paper-grade evidence.

The synthesis is implemented by:

- `scripts\127_synthesize_scaleup_attribution_gaps.ps1`
- `tca_map.smolvla.scaleup_attribution_gap_synthesis`

Purpose:

- separate TCA-Map target-conditioning gains from LoRA adaptation gains,
- record that Distributional TCA-Select currently adds no extra LoRA proxy gain in this bounded runner,
- identify what must be improved before a publishable attribution claim,
- keep current-checkpoint learned-policy rollout scaling and paper claims blocked.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\127_synthesize_scaleup_attribution_gaps.ps1
```
