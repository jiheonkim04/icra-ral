# LIBERO Offline LoRA Comparison Plan

This plan adds the required tiny real/offline LoRA comparison after the LIBERO HDF5 ActionMap vs TCA-Map proxy gate.

Scope:
- Read the ignored counterfactual split report from `scripts\51_build_libero_offline_counterfactual_split.ps1`.
- Read a bounded number of local LIBERO HDF5 action snippets.
- Train only tiny NumPy LoRA adapter matrices for `actionmap_lora`, `tca_map_lora`, and `tca_map_lora_distributional_select`.
- Use a low-dimensional action-prefix proxy, matching the current low-resolution ActionMap/TCA-Select smoke helpers.
- Require `ALLOW_TINY_TRAINING=1` for execution.

This is not:
- standard success,
- rollout success,
- a full SmolVLA adapter result,
- paper-grade evidence,
- simulator execution,
- OpenVLA-OFT execution.

Command:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\53_compare_libero_offline_lora.ps1
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

The output reports are runtime artifacts:
- `reports\libero_offline_lora_comparison_report.json`
- `reports\libero_offline_lora_comparison_report.md`

## Current Local Attribution Result

Command run:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\53_compare_libero_offline_lora.ps1 -MaxPairs 4 -MaxActionSteps 16 -MaxSteps 64 -MaxSamples 8 -Rank 4
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

Result summary:

- training happened: true,
- LoRA training happened: true,
- rollout happened: false,
- data source: local LIBERO HDF5 snippets from `reports\libero_offline_counterfactual_split_report.json`,
- same split as head-only: true,
- records: 8 total, 6 train / 2 eval,
- steps: 64,
- batch size: 1,
- device: CPU NumPy,
- result label: exploratory offline proxy, not paper-grade.

Sanity checks:

- target labels aligned: true,
- wrong-target proxy not inverted: true,
- target-conditioning input non-constant: true,
- TCA-Select candidate scores checked: true,
- TCA-Select candidate scores degenerate: false,
- external verifier used: false,
- privileged inference used: false.

Arm results:

- ActionMap + LoRA: initial loss `0.034182`, final loss `0.034104`, loss decreased true, LoRA params `84`, eval standard proxy `0.454351`, eval action L1 `0.091298`, target top1 `0.5`, wrong-target proxy `0.5`, counterfactual margin `0.012553`.
- TCA-Map + LoRA: initial loss `0.716626`, final loss `0.656217`, loss decreased true, LoRA params `168`, eval standard proxy `0.0`, eval action L1 `0.082307`, target top1 `0.0`, wrong-target proxy `1.0`, counterfactual margin `0.019284`.
- TCA-Map + LoRA + Distributional TCA-Select: matched TCA-Map + LoRA in this run; no measurable TCA-Select gain.

Conclusion: `lora_weakens_tca_map`. LoRA did not rescue the weak head-only TCA-Map result on this split. TCA-Map + LoRA improved action L1 and counterfactual margin versus ActionMap + LoRA, but target accuracy, wrong-target proxy, action-target consistency, and standard proxy remained worse. This is weak novelty evidence for the current TCA-Map formulation under this diagnostic.
