# Official SmolVLA Skipped Stage Audit

Date: 2026-07-10 KST

Audit scope: official SmolVLA-LIBERO history from `72ed23e` through `5d48b1e`.

## Stage Checklist

| # | stage | classification | evidence / note |
| ---: | --- | --- | --- |
| 1 | Official SmolVLA model load | `COMPLETE` | `reports/official_smolvla_libero_mini_repro_result.md` records model loaded on `cuda:0`; later JSON reports use `C:\assets\checkpoints\smolvla_libero`. |
| 2 | Official processor/preprocessor load | `COMPLETE` | Mini repro records `make_pre_post_processors`, processor/preprocessor loaded, official postprocessed 7D actions. |
| 3 | Official LeRobot LIBERO dataset load | `COMPLETE` | Mini repro and asset verification record `LeRobotDataset` / `LeRobotDatasetMetadata` passing on `C:\assets\datasets\lerobot_libero`. |
| 4 | Official rank-4 LoRA CUDA training | `COMPLETE` | Baseline scaleup, stable artifact, and seed repro all record rank-4 LoRA CUDA training with model and inputs on `cuda:0`. |
| 5 | Stable episode-disjoint manifest | `COMPLETE` | `reports/official_smolvla_split_manifest.json`: train/val/test episode intersections are all `0`. |
| 6 | Larger stable prediction artifact | `COMPLETE` | `reports/official_smolvla_stable_prediction_artifact.json`, `2800` records, SHA256 recorded in ledger JSON. |
| 7 | Multiple independent LoRA seeds | `COMPLETE` | Seeds `11`, `22`, `33`; each has its own prediction artifact and seed metadata. |
| 8 | Validation-only static-alpha selection | `COMPLETE` | Metric protocol says validation-only; stable evaluator records `selection_split: val`, `test_tuning_allowed: False`. |
| 9 | Official closed-loop LIBERO evaluation | `NOT_DONE_REQUIRED_NEXT` | Reports repeatedly state no simulator rollout. This is the main missing milestone after protocol fixes. |
| 10 | Official task success-rate evaluation | `NOT_DONE_REQUIRED_NEXT` | No official success-rate rollout exists. Current metrics are offline action/regression metrics only. |
| 11 | Full benchmark | `INTENTIONALLY_DEFERRED` | Explicitly not run; should follow a successful official rollout readiness gate. |
| 12 | Official MoIRA reproduction | `INTENTIONALLY_DEFERRED` | Current router is a local task/instruction proxy, not official MoIRA. Required only if future claims compare to MoIRA. |
| 13 | True adapter-weight soup/merge | `MISNAMED_OR_MISREPRESENTED` | Current static mixture is action-space interpolation of base and LoRA predictions, not adapter-weight merging. |
| 14 | Task-balanced metric reporting | `COMPLETE` | Metric protocol requires task-balanced means; stable and seed reports include task-balanced fields and task win counts. |
| 15 | Prediction/checkpoint artifact persistence | `PARTIAL` | Prediction artifacts are persisted; seed-specific LoRA adapter checkpoints are not persisted. |
| 16 | Hugging Face model/dataset revision pinning | `NOT_DONE_REQUIRED_NEXT` | Local paths and file hashes are recorded, but HF source revisions/snapshot IDs are not pinned in final reports. |
| 17 | Reproducible commands for every final baseline | `PARTIAL` | Final runners `247`, `248`, `249` are preserved and hashed; exact asset revisions and environment lock are incomplete. |

## Skipped Or Deferred Stages

Required next before paper-grade evidence:

- official closed-loop LIBERO evaluation
- official task success-rate evaluation
- Hugging Face model/dataset revision pinning
- naming correction for proxy baselines
- persistence policy for seed-specific LoRA adapter checkpoints

Intentionally deferred:

- full benchmark
- official MoIRA reproduction
- true adapter-weight soup/merge, unless future method comparisons require it

## Result Validity

The skipped stages do not invalidate the current offline action-L2 result. They limit the result to offline evidence and prevent RA-L-grade claims until official closed-loop evaluation and revision pinning are added.
