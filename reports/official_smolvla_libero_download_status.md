# Official SmolVLA-LIBERO Download Status

Date: 2026-07-09 KST

Branch: `codex/official-smolvla-libero-mini-repro`

## Scope

The user approved downloading exactly:

1. `lerobot/smolvla_libero`
2. `lerobot/libero`

No other model or dataset asset was approved for download.

## Risk Assessment

- Source: official Hugging Face Hub repos.
- Expected persistent size: about `2.647 GiB` plus cache/filesystem overhead.
- Target roots:
  - `C:\assets\checkpoints\smolvla_libero`
  - `C:\assets\datasets\lerobot_libero`
- Disk free before download: `416,420,818,944` bytes.
- Token/login/license click-through: none observed.
- GPU/training/rollout/OpenVLA-OFT: none during download.
- Decision: proceed, because the user explicitly approved these two assets and disk budget was safe.

## Download Attempts

Attempt 1:

```powershell
huggingface-cli download ...
```

Result: failed before downloading because `huggingface-cli` was not on PowerShell `PATH`.

Attempt 2:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\Scripts\huggingface-cli.exe download ...
```

Result: failed before downloading because the CLI emitted a Unicode deprecation warning that could not be encoded by the CP949 console.

Attempt 3:

```powershell
$env:HF_HOME='C:\assets\hf_home'
$env:PYTHONIOENCODING='utf-8'
C:\Users\jiheo\miniconda3\envs\tca_map\Scripts\huggingface-cli.exe download lerobot/smolvla_libero --local-dir C:\assets\checkpoints\smolvla_libero
C:\Users\jiheo\miniconda3\envs\tca_map\Scripts\huggingface-cli.exe download lerobot/libero --repo-type dataset --local-dir C:\assets\datasets\lerobot_libero
```

Result: success.

The CLI reported that Xet storage was enabled but `hf_xet` was not installed, so it fell back to regular HTTP. No additional package was installed.

## Downloaded Sizes

| asset | visible files | visible bytes | all files incl. `.cache` | all bytes incl. `.cache` |
| --- | ---: | ---: | ---: | ---: |
| `C:\assets\checkpoints\smolvla_libero` | 9 | 906,741,829 | 19 | 906,742,834 |
| `C:\assets\datasets\lerobot_libero` | 457 | 1,935,512,060 | 915 | 1,935,570,335 |

Visible total:

```text
2,842,253,889 bytes = about 2.647 GiB
```

## Download Decision

Download status: success.

Final download set remained exactly the two user-approved official assets.
