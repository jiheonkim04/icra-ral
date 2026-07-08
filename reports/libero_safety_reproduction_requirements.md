# LIBERO-Safety Reproduction Requirements

Date: 2026-07-09

Decision: `TOO_HEAVY_LOCAL`

This file lists what an official LIBERO-Safety reproduction would require. It is
not an execution plan for this run.

## Full Official Reproduction

Required inputs:

- Official code: https://github.com/LIBERO-SAFETY/LIBERO-Safety
- Official assets archive: https://huggingface.co/datasets/LIBERO-Safety/libero_safety_assets/tree/main
- Official training dataset if training or data-scaling is reproduced: https://huggingface.co/datasets/LIBERO-Safety/libero_safety
- Official pi0.5 checkpoint if reproducing the released pi0.5 path: https://huggingface.co/LIBERO-Safety/pi05_libero_safety/tree/main
- Additional model repositories/checkpoints for the paper's other VLA and embodied-model baselines.

Required setup:

- Clone/install the LIBERO-Safety fork instead of standard LIBERO.
- Configure `~/.libero/config.yaml` or `LIBERO_CONFIG_PATH` so assets,
  BDDL files, benchmark roots, datasets, and init states point to the
  LIBERO-Safety tree.
- Install package requirements, extra requirements, and
  `third_party/robosuite-1.4`.
- Install rendering/system dependencies for robosuite/MuJoCo as needed.
- Download and unzip `assets.zip` into `LIBERO-Safety/libero/libero/`.
- Run official rollout/evaluation scripts for the chosen benchmark suite and
  model path.

Expected resource floor:

- Assets: 10,670,353,443 bytes.
- pi0.5 model metadata total: 12,440,507,736 bytes.
- Training dataset page reports about 19.1 GB; metadata reports 19,664 episodes
  and 3,443,735 frames.
- Lower-bound disk footprint before code, environments, caches, and outputs:
  about 42 GB.
- Runtime is not specified by official docs. Full model reproduction likely
  requires GPU. The paper reports large training runs for several baselines,
  including tens of thousands of optimization steps and an 8-GPU GR00T setting.

## Small Official Subset

A valid mini reproduction would still need:

- Official LIBERO-Safety code and assets.
- A declared official task/suite subset.
- Official simulator reset and rollout path.
- One official policy/checkpoint or a clearly sanctioned official baseline.
- Official metrics, not a local proxy metric.

This is not feasible in the current run because large downloads and rollouts are
forbidden.

## Metric-Only Reproduction

Metric-only reproduction would require released rollout logs, per-episode
records, or metric artifact files. This scout did not find such artifacts on the
official project, GitHub, or Hugging Face pages.

Therefore metric-only reproduction is not available.

## Current Run Boundary

Forbidden in this run:

- Large downloads.
- Simulator rollouts.
- Training.
- GPU use.
- OpenVLA-OFT.
- Local proxy benchmarks.
- New VLA method work.

Under these constraints, only source-level feasibility is possible, and that is
not enough to open a benchmark reproduction gate.
