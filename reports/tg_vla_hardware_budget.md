# TG-VLA Hardware Budget

Date: 2026-07-09 KST

## Current Machine

- GPU: NVIDIA GeForce RTX 5080
- GPU memory total: 16,303 MiB
- GPU memory free at check: 12,603 MiB
- Driver: 596.21
- CPU: Ryzen 7 7800X3D per user context
- RAM: 24GB per user context
- OS: Windows 11
- Disk free on C: 420,019,408,896 bytes

## Local Asset Footprint

- SmolVLA checkpoint directory: 906,732,304 bytes across 10 files
- LIBERO HDF5 files: 100,442,962,652 bytes across 266 HDF5 files
- LIBERO-Para metadata CSV: 708,053 bytes

## SmolVLA Budget

Repo checker estimate:

- expected SmolVLA load: 12,000 MB
- required headroom: 2,048 MB
- fits RTX 5080 16GB budget: true

Practical constraint:

- this is tight but plausible for frozen/backbone or tiny adapter batch-1 smoke,
- first real smoke should prefer CPU or carefully bounded CUDA load, then stop on OOM,
- no full fine-tuning,
- no rollout,
- no multi-seed,
- no full benchmark.

## OpenVLA-OFT Budget

OpenVLA-OFT is blocked locally.

The official project FAQ lists minimum training memory for LIBERO at 25.6GB for batch size 1, and recommended training configurations at 44.1GB or higher. It also reports training jobs using 8 A100/H100 80GB GPUs for 1-2 days. This exceeds the RTX 5080 16GB local budget.

## Runtime Estimates

For a future first TG-VLA smoke:

- model load/interface: previously under 1 minute on CPU for synthetic single-sample smoke,
- tiny frozen SmolVLA inference over a few LIBERO/Para samples: likely minutes, not hours,
- tiny adapter optimization: seconds to minutes if only the adapter is trained,
- VRAM target: <= 14GB,
- hard stop: OOM, CUDA failure, or runtime above predeclared cap.

## Download Budget

No download is required for STATE 0-1 or the smallest local SmolVLA/LIBERO-Para smoke. Any future package install or model/dataset download needs a separate risk assessment.
