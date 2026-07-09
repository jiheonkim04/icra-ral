# Official SmolVLA Artifact Integrity Audit

Date: 2026-07-10 KST

Audit-only boundary: no model load, no inference, no training, no GPU.

## Fixed Manifest Integrity

Manifest: `reports/official_smolvla_split_manifest.json`

- SHA256: `1279F939648CF13E2F599084E42631681E1DFA5606B5D9B0851FFEB32710934B`
- manifest version: `1`
- dataset root: `C:\assets\datasets\lerobot_libero`
- train frames: `1200`
- validation frames: `400`
- test frames: `1200`
- total frames: `2800`
- train episodes: `80`
- validation episodes: `40`
- test episodes: `80`
- train/validation episode intersection: `0`
- train/test episode intersection: `0`
- validation/test episode intersection: `0`
- task count per split: `40`

Verdict: manifest integrity is `PASS`.

## Prediction Artifact Integrity

| artifact | bytes | SHA256 | artifact version | dataset record count | actual records | seed metadata |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `reports/official_smolvla_stable_prediction_artifact.json` | `7219361` | `88DCA06AA05D69E8BC4FB3F1C5A7C7D22B1DC4438C65103EFD2389F24D35D59C` | `2` | `2800` | `2800` | n/a |
| `reports/official_smolvla_lora_seed_11_prediction_artifact.json` | `6707622` | `F40298ACB449FFCBB8FBDFA341B65FDB6120259F7986559E644BB7771CD5A331` | `3` | `2800` | `2800` | `11` |
| `reports/official_smolvla_lora_seed_22_prediction_artifact.json` | `6707605` | `913CB7A3D228002BB73D059D23F5112AC537B5A560CEED19DFB8A2C976A5EF86` | `3` | `2800` | `2800` | `22` |
| `reports/official_smolvla_lora_seed_33_prediction_artifact.json` | `6707807` | `14568E506D0D5FCC9FABA8EDF7C5D3CDE628F9321AE3CC071CBC4537F41D9363` | `3` | `2800` | `2800` | `33` |

Verdict: seed artifacts exist, have distinct hashes, preserve seed metadata, and were not overwritten by each other.

## Validation-Only Static Alpha

Evidence:

- `reports/official_smolvla_metric_protocol.md` requires static alpha selection on validation only.
- `tca_map/smolvla/official_libero_stable_artifact_eval.py` records `selection_split` and `test_tuning_allowed: False`.
- `reports/official_smolvla_lora_seed_repro_result.json` records the stable artifact static selection being reproduced rather than retuned on test.

Verdict: no test leakage found for static alpha selection.

## Old Custom Route

Evidence:

- Final reports record `old_custom_route_used: false` or `custom_libero_7d_route_used: false`.
- Official runners use `tca_map.smolvla.official_libero_*` modules and the official LeRobot dataset/checkpoint paths.
- The archived `tca_map/smolvla_lora_baseline/*` route remains present for history but is not used by final official runners.

Verdict: no accidental use of the old custom `LIBERO_7D` route found in final official protocol runs.

## Report/JSON Consistency

Spot-checked report values agree with JSON evidence:

- final stable artifact decision: `NEEDS_LONGER_LORA_BASELINE_REPRO`
- final seed reproduction decision: `STATIC_MERGE_ROBUST_BASELINE_READY`
- seed list: `11`, `22`, `33`
- static seed win count: `3` / `3`
- rank-4 LoRA action L2 mean/std: `0.088239344` / `0.002908670`
- static mix action L2 mean/std: `0.080616431` / `0.002595356`
- frame oracle headroom after static mean: `0.011499227`

Verdict: no artifact/report inconsistency found.

## Integrity Gaps

- Hugging Face model/dataset source revisions are not pinned.
- Seed-specific LoRA adapter checkpoints are not persisted; prediction artifacts are persisted.
- Some early manual/mini-repro commands are documented by reports but not preserved as final runner scripts in the audited commit range.

These are protocol gaps, not evidence that the current offline metrics are invalid.
