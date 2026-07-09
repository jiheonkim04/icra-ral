# Official SmolVLA Revision Lock

Date: 2026-07-10 KST

Decision: `LORA_CHECKPOINTS_MISSING_REGENERATION_REQUIRED`

This is a no-experiment protocol-fix record. It did not run training, model inference, GPU code, downloads, simulator rollout, or metric regeneration.

## Lock File

Machine-readable source of truth:

- `configs/official_smolvla_repro_lock.yaml`

Validation helper:

- `scripts/check_official_smolvla_repro_lock.py`

## Model Revision

Status: `REVISION_LOCKED`

- repo: `lerobot/smolvla_libero`
- local path: `C:\assets\checkpoints\smolvla_libero`
- locked Hugging Face revision: `31d453f7edd78c839a8bbc39744a292686daf0de`
- proof: `9` local Hugging Face download metadata files all point to the same revision
- local HF `main` ref: `31d453f7edd78c839a8bbc39744a292686daf0de`
- visible files excluding `.cache`: `9`
- visible bytes excluding `.cache`: `906741829`
- license in README metadata: `apache-2.0`

The model config contains internal `repo_id: pepijn223/smolvla_libero`, but the local cache path and download metadata lock the artifact used here to `lerobot/smolvla_libero` at the revision above.

Key model config fields:

- policy type: `smolvla`
- action dim: `7`
- chunk size: `50`
- action steps: `50`
- config device: `cuda`
- `use_amp`: `false`
- `use_peft`: `false`
- VLM: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`

## Dataset Revision

Status: `REVISION_LOCKED`

- repo: `lerobot/libero`
- local path: `C:\assets\datasets\lerobot_libero`
- locked Hugging Face revision: `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
- proof: `457` local Hugging Face download metadata files all point to the same revision
- local HF `main` ref: `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
- local HF `v3.0` ref: `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
- visible files excluding `.cache`: `457`
- visible bytes excluding `.cache`: `1935512060`
- license in README metadata: `apache-2.0`

Dataset summary from local `meta/info.json`:

- codebase version: `v3.0`
- robot type: `panda`
- total episodes: `1693`
- total frames: `273465`
- total tasks: `40`
- fps: `10.0`
- action shape: `[7]`
- observation state shape: `[8]`

The checkpoint `train_config.json` has dataset revision `null`; this lock therefore uses the local Hugging Face metadata and refs as the reproduction source of truth.

## Package And Environment Lock

Status: `VERSION_LOCKED_SOURCE_COMMIT_UNAVAILABLE`

- OS: `Windows-10-10.0.26200-SP0`
- Python executable: `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe`
- Python: `3.10.20`
- PyTorch: `2.10.0+cu128`
- PyTorch compiled CUDA: `12.8`
- LeRobot: `0.4.4`
- transformers: `4.57.6`
- peft: `0.19.1`
- bitsandbytes: `0.49.2`
- accelerate: `1.14.0`
- huggingface_hub: `0.35.3`
- mujoco: `2.3.7`
- robosuite: `NOT_INSTALLED`
- libero: `NOT_INSTALLED`

Local package commits are not recoverable from installed wheel metadata, so package versions are pinned while package source commits remain unknown.

## Locked Project Artifacts

- split manifest: `reports/official_smolvla_split_manifest.json`
  - SHA256: `1279F939648CF13E2F599084E42631681E1DFA5606B5D9B0851FFEB32710934B`
- metric protocol: `reports/official_smolvla_metric_protocol.md`
  - SHA256: `64430225940C5168B3734BB40F9F48AD02877E0BA04DC804367AFBB214AE486E`
- stable prediction artifact: `reports/official_smolvla_stable_prediction_artifact.json`
  - SHA256: `88DCA06AA05D69E8BC4FB3F1C5A7C7D22B1DC4438C65103EFD2389F24D35D59C`
- LoRA seed repro result: `reports/official_smolvla_lora_seed_repro_result.json`
  - SHA256: `BAA9BD61DA4631F8CF7020198147A52F66435DBFCDDF02717BE2188CC8E79505`

## Revision-Lock Conclusion

Model and dataset revisions are locked from local metadata. The remaining rollout blocker is not revision ambiguity; it is missing persisted official LoRA adapter checkpoint bundles and missing local official LIBERO eval dependencies.
