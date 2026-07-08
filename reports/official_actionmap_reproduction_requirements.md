# Official ActionMap Reproduction Requirements

Date: 2026-07-09

## Current Status

`SOURCE_BLOCKED`

The official release is not yet sufficient for reproduction. The GitHub repository exposes a core action-head preview, but not the official training/evaluation stack.

## Required For Exact Official Reproduction

- Official tagged code release or pinned commit.
- Complete install instructions, including Python/JAX/PyTorch/CUDA versions.
- Official training and evaluation scripts.
- Official configuration files for OpenVLA-OFT and pi0.5 backbones.
- Official LIBERO dataset preparation instructions.
- Official checkpoint acquisition instructions for the baseline backbones.
- Official ActionMap checkpoint or training recipe.
- Official metric computation and expected outputs.
- Official simulator setup for LIBERO/RoboSuite.
- Hardware requirements and expected runtime.

## Required For Small Official Subset Reproduction

A small subset would still need official support:

- one named LIBERO suite/task subset;
- one named backbone;
- one released or officially prepared checkpoint;
- one official command for either evaluation-only or bounded finetuning;
- expected metric and tolerance;
- disk and VRAM estimate.

No such subset is currently published.

## Required For Metric-Only Reproduction

Metric-only reproduction would require released logs, checkpoints, or evaluation artifacts:

- per-suite LIBERO success logs;
- baseline and ActionMap run identifiers;
- evaluation seeds;
- raw success/failure counts;
- script or notebook that recomputes the paper table.

No such logs or artifacts are currently linked from the paper, project page, or repository.

## Hardware Requirements From Paper

The paper reports:

- OpenVLA-OFT LIBERO finetuning with effective batch size 64 on 2 H200 GPUs.
- pi0.5/JAX LIBERO runs with global batch size 256 on 8 H200 GPUs.
- Real-world Franka pi0.5/JAX runs with global batch size 64 on 4 H200 GPUs.

These are outside the current local constraints. CPU-only reproduction is not described by the official paper or project page.

## Local Constraint Conflicts

- No GPU use is allowed in this scout.
- No training is allowed in this scout.
- No large downloads are allowed in this scout.
- OpenVLA-OFT checkpoint path is missing locally.
- Full official ActionMap code/assets are not yet released.

## Minimum Future Command Shape

There is no valid command to run now.

The first future command, if the official release appears, should be a read-only source check such as:

```powershell
git ls-remote https://github.com/showlab/ActionMap.git
```

Only after a full official release with explicit instructions should any install, download, training, or evaluation command be considered.
