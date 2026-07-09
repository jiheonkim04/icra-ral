# Official SmolVLA / LeRobot Environment Status

Date: 2026-07-09 KST

## Required Initial Repo Checks

- Initial branch: `main`
- Initial commit: `72ed23e Archive custom SmolVLA adapter route`
- Working branch created: `codex/official-smolvla-lerobot-baseline`

## Python

- Plain `python`: resolves to Windows Store alias, not usable.
- Required interpreter: `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe`
- Python version: `3.10.20`

## CUDA

Command requested by user:

```powershell
nvidia-smi
```

Result:

- GPU: NVIDIA GeForce RTX 5080
- Driver: 596.21
- CUDA shown by `nvidia-smi`: 13.2
- Memory: 16,303 MiB total
- Memory in use at check time: 3,843 MiB, mostly display/desktop processes

Command requested by user, run with the repo interpreter:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Result:

```text
True
NVIDIA GeForce RTX 5080
```

## Package Status

| package | status |
| --- | --- |
| torch | `2.10.0+cu128`, import ok |
| CUDA in torch | available, device `NVIDIA GeForce RTX 5080` |
| peft | `0.19.1`, import ok |
| bitsandbytes | `0.49.2`, import ok |
| transformers | `4.57.6`, import ok |
| accelerate | `1.14.0`, import ok |
| lerobot | `0.4.4`, import ok |
| safetensors | `0.8.0`, import ok |
| datasets | `4.8.5`, import ok |

## bitsandbytes CUDA Smoke

Executed a tiny Adam8bit optimizer step on a CUDA tensor.

- Training: no
- Downloads: no
- CUDA available: yes
- Parameter device: `cuda:0`
- Loss computed: `4.0`
- CUDA max allocated: `0.002 MB`
- Result: pass

## CPU-Only Diagnostic Clarification

The official SmolVLA mini-repro in this pass was intentionally CPU-only. It was not LoRA training. Therefore model parameters and tensors being on CPU in that smoke are expected and are not `CPU_FALLBACK_BUG`.

Future LoRA training must log:

- model parameter device;
- input tensor devices;
- `torch.cuda.memory_allocated()`;
- `torch.cuda.max_memory_allocated()`;
- autocast/fp16/bf16 state.

