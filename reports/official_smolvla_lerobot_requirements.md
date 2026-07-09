# Official SmolVLA / LeRobot Requirements

Date: 2026-07-09 KST

## Hardware

- GPU: NVIDIA GeForce RTX 5080
- VRAM: 16,303 MiB
- Driver: 596.21
- CUDA visible to PyTorch: yes
- CPU-only diagnostics are allowed when explicitly labeled.
- Long CPU training is forbidden.

## Python Environment

Use the explicit interpreter:

```text
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe
```

Plain `python` resolves to the Windows Store alias in this shell and must not be used for training or validation.

Installed package versions:

| package | version |
| --- | --- |
| Python | 3.10.20 |
| torch | 2.10.0+cu128 |
| CUDA runtime reported by torch | 12.8 |
| peft | 0.19.1 |
| bitsandbytes | 0.49.2 |
| transformers | 4.57.6 |
| accelerate | 1.14.0 |
| lerobot | 0.4.4 |
| huggingface_hub | 0.35.3 |
| safetensors | 0.8.0 |
| datasets | 4.8.5 |

## Required Local Assets

Current local assets:

- `C:\assets\checkpoints\smolvla`: exists
- `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`: exists
- local LIBERO HDF5 data under `C:\assets\data\libero`: exists
- local LeRobot official `lerobot/libero` dataset cache: not found
- local `lerobot/smolvla_libero` checkpoint: not found

## Required Official-Recipe Conditions

Proceed beyond mini-repro only when all are true:

- SmolVLA loads with LeRobot's official `SmolVLAPolicy.from_pretrained`.
- Pre/postprocessors are made by LeRobot's official processor factory.
- Tokenizer, processor, and normalizer files resolve locally or through an approved official source.
- Action/state convention is explicit.
- Any LIBERO run uses LeRobot's official LIBERO processor and 8D state / 7D action convention, not the archived custom 7D adapter route.
- GPU training logs model parameter device, input tensor devices, allocated/max CUDA memory, and autocast/fp16/bf16 state.

## Stop Conditions

Stop if:

- the only runnable path is the archived custom 7D adapter route;
- action normalizer provenance is unclear;
- local checkpoint schema is silently coerced to LIBERO 7D;
- full training, full benchmark, simulator rollout, OpenVLA-OFT, or large downloads are required;
- CUDA is available but a supposed GPU training run keeps model or inputs on CPU.

