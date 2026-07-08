# TG-VLA Source Feasibility

Date: 2026-07-09 KST

No downloads, installs, training, rollouts, GPU jobs, or OpenVLA-OFT execution were performed.

## Source Matrix

| Source | Local status | Official/source status | Size / access | Use in TG-VLA | Risk |
| --- | --- | --- | --- | --- | --- |
| SmolVLA checkpoint | Present at configured local path; checker reports config/weights and external tokenizer dependency present. | `lerobot/smolvla_base` on Hugging Face; model card documents inference and training-step APIs. | Local checkpoint files sum to 906,732,304 bytes. HF model card lists 0.5B params. | First real VLA backbone candidate. | Feasible but GPU memory is tight. |
| SmolVLA tokenizer/processor dependency | Present under `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`. | Referenced by local `policy_preprocessor.json`. | Local tokenizer/config files only. | Required for local text preprocessing. | Full dependency weights should not be downloaded during this run. |
| Local LIBERO HDF5 demos | Present; asset checker sampled local files and counterfactual split builder matched local HDF5 demos. | Standard LIBERO benchmark data path in local assets. | 266 HDF5 files, 100,442,962,652 bytes. | Offline action labels, clean/counterfactual action chunks, possible first-frame observations. | Offline only; not rollout success. |
| LIBERO-Para metadata | Present at `C:\assets\data\libero_para\libero_para_metadata.csv`. | Hugging Face `HAI-Lab/LIBERO-Para`, MIT license, official paper/code. | Local CSV has 4,092 rows; HF page lists total file size 12.1 MB. | Held-out paraphrase and object lexical variation groups. | Evaluation-only paraphrases must not be used as target labels at inference. |
| LIBERO-Para original instruction matches | Local check found all 10 original LIBERO-Para instructions have matching local LIBERO-Goal HDF5 demos. | LIBERO-Para is built on LIBERO-Goal. | No download needed. | Enables small official/standard paraphrase/action-chunk split. | Need deterministic group split and leakage audit. |
| OpenVLA-OFT | No local checkpoint; intentionally blocked. | Official project reports strong LIBERO and ALOHA performance but heavy compute. | Project FAQ lists 25.6GB minimum GPU memory for LIBERO training config and 1-2 days on 8 A100/H100 80GB GPUs. | Baseline/reference only. | Too heavy locally; do not run. |

## Local Checks Performed

- `scripts\11_check_real_assets.ps1`
- `scripts\13_check_smolvla_adapter_smoke.ps1`
- `scripts\17_check_smolvla_runtime_deps.ps1`
- `scripts\47_build_libero_metadata_subset.ps1`
- `scripts\51_build_libero_offline_counterfactual_split.ps1`
- local LIBERO-Para CSV schema/count inspection
- local disk, GPU, and package availability checks

## Source Decision

Source status alone is not blocked.

The local machine has enough official/standard assets for a small SmolVLA plus LIBERO/LIBERO-Para adapter smoke without new downloads. The blocking issue is not source availability; it is the novelty/baseline gate and absence of a repo-integrated real TG-VLA adapter runner.
