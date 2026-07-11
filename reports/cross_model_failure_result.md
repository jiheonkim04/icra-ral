# Cross-Model Failure Result

Date: 2026-07-11 KST

Final decision: `SECOND_BACKBONE_OR_BENCHMARK_BLOCKED`

## Execution Summary

- selected second backbone: `OpenVLA-OFT`
- selected checkpoint: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`
- selected second benchmark: `LIBERO-PRO`
- downloads happened: `false`
- training happened: `false`
- rollout happened: `false`
- method implemented: `false`
- episodes completed: `0`
- videos recorded: `0`

## Why No Cross-Backbone Rollout Ran

The selected OpenVLA-OFT checkpoint is public, MIT, and non-gated, but the checkpoint is `14.845` GiB. The user explicitly forbade large asset downloads without reporting size and approval. The local RTX 5080 16GB path is also not proven safe for official full-precision inference because the checkpoint size leaves little VRAM margin.

Therefore State 4 cannot run in this pass. The correct State 1 decision is `SECOND_BACKBONE_DOWNLOAD_APPROVAL_REQUIRED`.

## Results

No cross-backbone or cross-benchmark task-success result exists yet.

Mechanism generality:

- `stable_grasp`: unproven; second backbone not run.
- `long_horizon_compounding`: unproven; second backbone not run.

The current evidence remains SmolVLA-only and does not support an RA-L method.

## Next Gate

The next valid action is an explicit approval decision for the OpenVLA-OFT checkpoint download and hardware path. After that, run only the predeclared bounded protocol in `reports/cross_model_failure_manifest.json`.
