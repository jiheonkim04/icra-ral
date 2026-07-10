# Project State

Date: 2026-07-10 KST

Target branch: `main`

Implementation branch: `codex/regenerate-official-smolvla-lora-checkpoints`

Current decision: `LORA_REGEN_METRIC_DRIFT_BLOCKS_ROLLOUT`

## Current Route

The archived custom SmolVLA 7D adapter route remains stopped. The valid route is official SmolVLA/LeRobot reproduction first, using official preprocessing, normalization, action conventions, dataset format, and evaluation stack.

This update regenerated the required standard rank-4 LoRA adapter checkpoints for seeds `11`, `22`, and `33`. It did not install `libero` or `robosuite`, initialize a simulator, run closed-loop rollout, download assets, run OpenVLA-OFT, revive FCAR, design a method, change the model/dataset revisions, change the split manifest, change the metric protocol, tune static alpha on test, or rewrite historical metrics.

## Locked Inputs

- model: `lerobot/smolvla_libero`
  - revision: `31d453f7edd78c839a8bbc39744a292686daf0de`
  - local path: `C:\assets\checkpoints\smolvla_libero`
- dataset: `lerobot/libero`
  - revision: `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
  - local path: `C:\assets\datasets\lerobot_libero`
- split manifest SHA256: `1279F939648CF13E2F599084E42631681E1DFA5606B5D9B0851FFEB32710934B`
- metric protocol SHA256: `64430225940C5168B3734BB40F9F48AD02877E0BA04DC804367AFBB214AE486E`
- prior seed result SHA256: `BAA9BD61DA4631F8CF7020198147A52F66435DBFCDDF02717BE2188CC8E79505`

## Checkpoint Regeneration Status

All three required seed adapter bundles now exist and were disk-reload verified:

| Seed | Checkpoint path | Status |
| ---: | --- | --- |
| `11` | `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_11` | `CHECKPOINT_COMPLETE_VERIFIED` |
| `22` | `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_22` | `CHECKPOINT_COMPLETE_VERIFIED` |
| `33` | `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_33` | `CHECKPOINT_COMPLETE_VERIFIED` |

Central manifest:

- `reports/official_smolvla_lora_checkpoint_manifest.json`

Seed-specific prediction artifacts generated from disk-reloaded adapters:

- `reports/official_smolvla_seed_11_prediction_artifact.json`
- `reports/official_smolvla_seed_22_prediction_artifact.json`
- `reports/official_smolvla_seed_33_prediction_artifact.json`

## Reproduction Result

Frozen tolerance:

- per-seed rank-4 LoRA action L2 diff must be `<= 0.002`
- per-seed static-mix action L2 diff must be `<= 0.002`
- aggregate mean diffs must be `<= 0.002`
- static mix must remain stronger than standalone LoRA

Outcome:

- checkpoint bundle completeness: passed
- checksum recording: passed
- disk reload verification: passed
- CUDA device placement: passed
- static-mix qualitative conclusion preserved: yes
- frozen metric tolerance: failed

Per-seed tolerance failures:

- seed `11` rank-4 LoRA action L2 diff: `0.003085157`
- seed `33` rank-4 LoRA action L2 diff: `0.004492506`
- seed `33` static-mix action L2 diff: `0.004988174`

Aggregate regenerated metrics:

- rank-4 LoRA mean/std: `0.087287222` / `0.001135689`
- validation-selected action-space static mix mean/std: `0.079536743` / `0.001025200`
- frame oracle upper-bound mean/std: `0.069590253` / `0.001577148`
- task oracle upper-bound mean/std: `0.080959383` / `0.001502216`

## Runtime And Device

- CUDA available: yes
- GPU: `NVIDIA GeForce RTX 5080`
- CPU fallback: no
- model/input devices recorded as `cuda:0`
- torch CUDA peak memory recorded in artifacts: about `1105 MB`
- seed elapsed times: seed `11` `1119.422s`, seed `22` `1114.5s`, seed `33` `1121.218s`

## Conclusion

`LORA_REGEN_METRIC_DRIFT_BLOCKS_ROLLOUT`

The checkpoint persistence gap is fixed, but rollout remains blocked because regenerated metrics exceeded the predeclared tolerance. Do not proceed to official rollout until the configuration drift is diagnosed under a new explicit objective.
