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
