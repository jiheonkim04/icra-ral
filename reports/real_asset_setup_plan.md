# Real Asset Setup Plan

This is the next setup step after the scaffold, preflight, dummy smoke, and pytest pass. It prepares local asset readiness checks only. It does not authorize GPU training, downloads, real rollouts, or VLA model execution.

## Recommendation

Use **SmolVLA first** for the first real-adapter smoke on the local RTX 5080 16GB machine.

Keep **OpenVLA-OFT** as the primary paper-grade baseline target, but do not attempt full OpenVLA-OFT fine-tuning locally. Treat OpenVLA-OFT as a later load/import and frozen-backbone smoke target after the local asset checker, memory check, and environment check pass.

Rationale:

- RTX 5080 16GB is strong, but OpenVLA-OFT can be memory-sensitive depending on checkpoint format, precision, tokenizer/cache state, and CUDA/PyTorch support for the GPU generation.
- Windows is acceptable for path checks, dummy smoke, and offline proxy preparation, but robotics simulator rollout work is much more likely to be stable under WSL2/Linux.
- SmolVLA-first reduces the chance that the first real-adapter smoke becomes a CUDA/VRAM/debugging exercise instead of a clean interface validation.

## Exact Asset Checklist

Configure local paths through `configs/paths.local.yaml` or environment variables. No script in this stage downloads anything.

| Asset | Environment variable | Needed for | Required before |
| --- | --- | --- | --- |
| OpenVLA-OFT checkpoint or local model directory | `OPENVLA_OFT_CKPT` | Paper-grade baseline target | OpenVLA-OFT adapter smoke |
| SmolVLA checkpoint or local model directory | `SMOLVLA_CKPT` | First recommended real-adapter smoke | SmolVLA adapter smoke |
| LIBERO source checkout | `LIBERO_ROOT` | Dataset loading and rollouts | LIBERO pilot and rollout |
| LIBERO data/demos root | `LIBERO_DATA_ROOT` | Offline LIBERO-style subset and rollout tasks | LIBERO pilot and rollout |
| RoboSuite checkout/install root | `ROBOSUITE_ROOT` | Simulator execution dependency | Any rollout |
| General local data root | `DATA_ROOT` | Counterfactual JSONL, tiny subsets, generated metadata | Tiny real pilot |
| Local checkpoint root | `CHECKPOINT_ROOT` | Local model cache/checkpoint organization | Real adapter smoke |
| Hugging Face cache root | `HF_HOME` | Offline tokenizer/model cache lookup | Real adapter smoke if model uses HF cache |

## Expected Disk Usage

These are planning reserves, not download instructions. Verify actual model and dataset sizes from the source you choose before copying assets.

- Scaffold and dummy reports: less than 1 GB.
- SmolVLA-first smoke assets: reserve 20-50 GB for checkpoint files, tokenizer files, and local cache overhead.
- OpenVLA-OFT smoke assets: reserve 80-200 GB for checkpoints, base model/cache files, and variants.
- LIBERO/RoboSuite setup plus demonstration data: reserve 50-150 GB depending on task suites and generated counterfactual subsets.
- Practical local free-space target before any real pilot: at least 150 GB free; 300 GB free is safer if OpenVLA-OFT and LIBERO data are both present.

## Expected VRAM Usage

- `scripts/11_check_real_assets.ps1`: 0 GB VRAM; it only checks paths.
- Dummy smoke: CPU-only by design.
- SmolVLA real-adapter smoke later: target 8-12 GB VRAM with batch size 1, frozen backbone, fp16/bf16 if safe, and no training.
- OpenVLA-OFT real-adapter smoke later: high OOM risk on 16 GB unless strictly load-only/frozen/head-only, low-resolution inputs, and mixed precision are used. Target under 14 GB before considering any tiny pilot.
- Any adapter/head training later: batch size 1 first, gradient accumulation only after load/inference smoke passes.

## Windows Native vs WSL2/Linux

Windows native is acceptable for:

- tree check,
- preflight,
- dummy train/eval smoke,
- `scripts/11_check_real_assets.ps1`,
- offline proxy dataset planning,
- possibly lightweight model import checks if the Python/CUDA stack supports the RTX 5080 cleanly.

WSL2/Linux is strongly recommended, and effectively required before paper-grade rollout work, for:

- LIBERO simulator rollouts,
- RoboSuite/MuJoCo rendering stability,
- CUDA/PyTorch wheels that match new GPU support,
- reproducible training runs,
- later cluster/SLURM parity.

## Environment Variables

PowerShell examples:

```powershell
$env:OPENVLA_OFT_CKPT="C:\assets\checkpoints\openvla-oft"
$env:SMOLVLA_CKPT="C:\assets\checkpoints\smolvla"
$env:LIBERO_ROOT="C:\assets\repos\LIBERO"
$env:LIBERO_DATA_ROOT="C:\assets\data\libero"
$env:ROBOSUITE_ROOT="C:\assets\repos\robosuite"
$env:DATA_ROOT="C:\assets\data"
$env:CHECKPOINT_ROOT="C:\assets\checkpoints"
$env:HF_HOME="C:\assets\hf_home"
```

Command Prompt examples:

```bat
set OPENVLA_OFT_CKPT=C:\assets\checkpoints\openvla-oft
set SMOLVLA_CKPT=C:\assets\checkpoints\smolvla
set LIBERO_ROOT=C:\assets\repos\LIBERO
set LIBERO_DATA_ROOT=C:\assets\data\libero
set ROBOSUITE_ROOT=C:\assets\repos\robosuite
set DATA_ROOT=C:\assets\data
set CHECKPOINT_ROOT=C:\assets\checkpoints
set HF_HOME=C:\assets\hf_home
```

## `configs/paths.local.yaml` Example

Copy `configs/paths.local.yaml.example` to `configs/paths.local.yaml`; the local file is ignored by git.

```yaml
assets:
  openvla_oft_ckpt: "C:/assets/checkpoints/openvla-oft"
  smolvla_ckpt: "C:/assets/checkpoints/smolvla"
  libero_root: "C:/assets/repos/LIBERO"
  libero_data_root: "C:/assets/data/libero"
  robosuite_root: "C:/assets/repos/robosuite"
  data_root: "C:/assets/data"
  checkpoint_root: "C:/assets/checkpoints"
  hf_home: "C:/assets/hf_home"
  wandb_api_key: null
```

## Commands After Assets Are Installed

Run these from the local clone on Windows. They still do not run VLA models, GPU training, downloads, or rollouts.

```powershell
cd C:\Users\jiheo\tca_map
git fetch origin
git switch codex/real-asset-setup-plan
conda activate tca_map
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
powershell -ExecutionPolicy Bypass -File scripts\99_tree_check.ps1
powershell -ExecutionPolicy Bypass -File scripts\00_preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts\04_train_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\05_eval_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
```

Linux/WSL path-check equivalent:

```bash
git fetch origin
git switch codex/real-asset-setup-plan
bash scripts/11_check_real_assets.sh
```

## Go / No-Go Criteria For Real Adapter Smoke

Go for **SmolVLA adapter smoke** only if:

- `ready_for_smolvla_smoke: true`,
- preflight and dummy smoke still pass,
- `safe_to_run_pilot_gpu` remains false until a separate explicit real-smoke task is approved,
- no downloads are needed,
- checkpoint/tokenizer files are local,
- the next task is load/inference smoke only, not training.

Go for **OpenVLA-OFT adapter smoke** only if:

- `ready_for_openvla_oft_smoke: true`,
- SmolVLA-first path has already validated adapter interfaces or is intentionally skipped with a written reason,
- local CUDA/PyTorch stack is known to support RTX 5080,
- smoke is load-only/frozen-backbone/head-only, batch size 1, low-resolution inputs, and fp16/bf16 only if safe,
- no full fine-tuning is attempted locally.

Go for **LIBERO rollout** only if:

- `ready_for_libero_rollout: true`,
- simulator import/render checks pass in WSL2/Linux or a known-good Linux environment,
- a separate rollout task is approved.

No-go if:

- any required local asset path is missing,
- the model would need a download to run,
- CUDA or PyTorch does not recognize the GPU,
- estimated memory leaves less than about 2 GB VRAM headroom,
- Windows-native simulator checks fail.

## Likely RTX 5080 16GB Failure Modes

- CUDA/PyTorch build does not yet support the GPU architecture cleanly.
- `CUDA out of memory` during model load, even before training.
- Model cache is incomplete because downloads are disabled.
- Tokenizer/config files are missing from local checkpoint directories.
- fp16/bf16 mismatch or unsupported kernels on the installed stack.
- Windows path quoting issues, especially with spaces in asset paths.
- RoboSuite/MuJoCo rendering failures on Windows.
- 24 GB system RAM causing paging during large checkpoint load.
- OpenVLA-OFT requiring base model files not present under `HF_HOME`.

## Fallback Path If OpenVLA-OFT Is Too Heavy

1. Keep OpenVLA-OFT as the paper-grade baseline target in reports.
2. Run SmolVLA-first adapter smoke to validate the real adapter path.
3. Use frozen-backbone or head-only training for any local tiny pilot.
4. Keep heatmaps low-resolution or coarse-to-fine.
5. Move OpenVLA-OFT load/inference smoke to WSL2/Linux or a larger-GPU machine if 16 GB VRAM is insufficient.
6. Do not claim SOTA or paper-grade OpenVLA-OFT comparison until the real baseline runs successfully.
