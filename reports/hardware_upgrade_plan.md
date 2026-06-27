# Hardware Upgrade Plan

## Current Machine Summary

- GPU: NVIDIA RTX 5080, about 16 GB VRAM.
- CPU: AMD Ryzen 7800X3D.
- System RAM: 24 GB.
- OS: Windows currently.
- Conda environment: `tca_map`, Python 3.10.

This machine is suitable for scaffold development, dummy smoke, real asset readiness checks, SmolVLA-first adapter smoke after assets are local, small offline proxy pilots, and possibly small WSL2/Linux LIBERO rollouts.

It is not a comfortable target for full OpenVLA-OFT fine-tuning, multi-seed sweeps, or large paper-grade baseline matrices.

## RAM Recommendation

Minimum practical RAM upgrade: **64 GB**.

Ideal RAM upgrade: **128 GB**.

Why RAM matters:

- Large checkpoint loads often stage tensors, tokenizer files, configs, and framework caches in system memory before or during GPU placement.
- Simulator rollouts can keep environment state, rendering buffers, observations, and replay/log buffers alive at the same time as model inference.
- LIBERO/RoboSuite debugging under WSL2/Linux benefits from extra memory because WSL, Python, simulator, and host tools can all consume memory concurrently.
- Counterfactual dataset generation and visual diagnostics are much smoother when cached images, JSONL metadata, and heatmap reports do not trigger paging.
- 24 GB RAM can work for scaffold and tiny checks, but it is easy to hit slow paging with a simulator plus a VLA stack.

## Disk Layout Recommendation

Use a dedicated asset layout and keep it outside the git repository:

```text
C:/assets/checkpoints
C:/assets/data
C:/assets/repos
C:/assets/hf_home
```

Recommended mapping:

- `CHECKPOINT_ROOT=C:/assets/checkpoints`
- `DATA_ROOT=C:/assets/data`
- `HF_HOME=C:/assets/hf_home`
- `LIBERO_ROOT=C:/assets/repos/LIBERO`
- `ROBOSUITE_ROOT=C:/assets/repos/robosuite`
- `LIBERO_DATA_ROOT=C:/assets/data/libero`
- `SMOLVLA_CKPT=C:/assets/checkpoints/smolvla`
- `OPENVLA_OFT_CKPT=C:/assets/checkpoints/openvla-oft`

Disk target:

- 150 GB free minimum before serious local asset setup.
- 300 GB free recommended if OpenVLA-OFT and LIBERO are both local.
- 500 GB or more if keeping multiple checkpoints, generated rollouts, heatmaps, and failure visualizations.

## What A RAM Upgrade Enables

With 64 GB RAM:

- More reliable SmolVLA local smoke and tiny pilots.
- Less paging during checkpoint load and offline dataset preprocessing.
- More comfortable WSL2/Linux simulator setup.
- Small LIBERO rollouts become more realistic to debug locally.

With 128 GB RAM:

- Better headroom for simulator plus model plus logging.
- More comfortable OpenVLA-OFT frozen/load smoke attempts.
- Larger offline proxy subsets and diagnostic visualization batches.
- Less risk that system RAM, rather than VRAM, becomes the limiting factor.

## What Remains Impossible Or Unwise Locally

Even after a RAM upgrade, the RTX 5080 16 GB VRAM remains the main limiter for large model work.

Still not recommended locally:

- Full OpenVLA-OFT fine-tuning.
- Full multi-seed LIBERO/RoboCasa sweeps.
- Large ablation matrix across all baselines.
- Full-resolution voxel heatmaps with large backbones.
- Claims of paper-grade standard success without simulator rollouts.

Likely cloud/remote needs remain:

- 24 GB GPU minimum for OpenVLA-OFT frozen/head-only experiments.
- 48 GB GPU recommended for larger baseline comparisons.
- 80 GB GPU recommended for multi-seed full baselines and broad ablations.
