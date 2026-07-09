# LoRA Environment Status

Date: 2026-07-09 KST

## Status

Local SmolVLA LoRA infrastructure is available and reusable.

This status is independent of the PatchGuard-VLA method kill. PatchGuard failed the baseline gate, but the LoRA environment worked.

## Installed Packages

- `peft`: `0.19.1`, import succeeded.
- `bitsandbytes`: `0.49.2`, import succeeded.
- `accelerate`: `1.14.0`, already available.
- `transformers`: `4.57.6`, already available.
- `torch`: `2.10.0+cu128`.

No repo requirements file was modified in the STATE 1B run.

## CUDA/GPU

- CUDA available: yes.
- Torch CUDA runtime: `12.8`.
- GPU: NVIDIA GeForce RTX 5080.
- Total VRAM reported by torch: `16302.562` MB.

## Dependency Smokes

- Tiny CUDA tensor smoke: passed.
- bitsandbytes 8-bit `Linear8bitLt` CUDA smoke: passed.
- bitsandbytes 4-bit `Linear4bit` CUDA smoke: passed.
- PEFT dummy LoRA smoke: passed.

## Local SmolVLA LoRA Path

- Model path tested: `C:\assets\checkpoints\smolvla`.
- External tokenizer dependency found locally under `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`.
- LoRA target modules used: `state_proj`, `action_in_proj`, `action_out_proj`.
- LoRA rank: `4`.
- Total params: `450056160`.
- Trainable params: `9984`.
- PEFT-wrapped SmolVLA retained usable policy behavior in the tiny smoke.

## Resource Envelope

Observed STATE 1B tiny smoke:

- batch size: `1`,
- rank: `4`,
- max steps: `10` per variant,
- VRAM peak: `2224.845` MB,
- runtime: `57.438` sec,
- loss computed: yes.

This is a small feasibility envelope, not a capacity claim for full training or benchmark-scale runs.

## Reuse Guidance

Use this environment for baseline-only SmolVLA LoRA work. Do not use it to launch a new custom method yet.

The first standard LoRA baseline reproduction ran after PatchGuard archive. It showed:

- loss decreased meaningfully,
- standard LoRA beat frozen/base SmolVLA,
- standard LoRA did not beat mean-action on the held-out split.

Before any new method is considered, the project still needs to understand:

- clean retention under standard LoRA,
- patched/perturbed behavior under standard LoRA,
- whether generic adversarial augmentation already explains gains,
- memory and runtime scaling beyond the one-sample smoke,
- why a trivial mean-action baseline dominates this local standard LoRA eval.

## Forbidden Interpretation

- LoRA working locally is not a paper claim.
- LoRA working locally does not rescue PatchGuard.
- bitsandbytes passing CUDA smokes does not authorize full QLoRA training.
- The next step is baseline/interface diagnosis, not method invention.
