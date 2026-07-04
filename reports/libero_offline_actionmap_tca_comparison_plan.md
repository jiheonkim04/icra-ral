# LIBERO Offline ActionMap vs TCA-Map Training/Eval

This executable milestone runs a tiny exploratory offline proxy comparison over local LIBERO HDF5 action snippets.

Command:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\52_compare_libero_offline_actionmap_tca.ps1
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

Scope:

- reads `reports\libero_offline_counterfactual_split_report.json`,
- reads only a tiny deterministic subset of action rows from local HDF5 demos,
- trains tiny CPU NumPy head-only models with batch size 1,
- reports loss curves, initial/final loss, trainable parameter counts, and offline proxy metrics,
- compares ActionMap head-only, TCA-Map head-only, and TCA-Map + Distributional TCA-Select.

This is exploratory offline proxy evidence. It is not standard success, not rollout success, and not paper-grade evidence.

Forbidden behavior:

- no downloads,
- no GPU jobs,
- no simulator execution,
- no rollouts,
- no heavy VLA imports,
- no OpenVLA-OFT execution,
- no token access,
- no LoRA training in this milestone,
- no paper-grade claims.

## Current Local Training/Eval Result

Command run:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\52_compare_libero_offline_actionmap_tca.ps1 -MaxPairs 4 -MaxActionSteps 16 -MaxSteps 64 -GridSize 8
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

Result summary:

- training happened: true,
- LoRA training happened: false,
- rollout happened: false,
- loss was computed: true,
- data source: local LIBERO HDF5 action snippets from `reports\libero_offline_counterfactual_split_report.json`,
- record count: 8,
- train/eval records: 6 / 2,
- batch size: 1,
- steps: 64,
- result label: exploratory offline proxy, not paper-grade.

Arm results:

- ActionMap head-only: initial loss `0.162408`, final loss `0.010239`, loss decreased true, trainable parameters `119`, eval standard proxy `0.434797`, eval action L1 `0.130406`, target top1 `0.5`, wrong-target proxy `0.5`, counterfactual margin `0.0`.
- TCA-Map head-only: initial loss `0.855555`, final loss `0.126224`, loss decreased true, trainable parameters `167`, eval standard proxy `0.0`, eval action L1 `0.111259`, target top1 `0.0`, wrong-target proxy `1.0`, counterfactual margin `0.02313`.
- TCA-Map + Distributional TCA-Select: reused the trained TCA-Map head, added 0 trainable parameters, eval metrics matched TCA-Map in this tiny proxy; no TCA-Select gain was measured.

Conclusion: `weakens_tca_map` for this tiny exploratory split. TCA-Map improved action L1 and counterfactual margin, but worsened target accuracy, wrong-target proxy rate, and standard proxy score. Distributional TCA-Select added no measurable gain in this run.
