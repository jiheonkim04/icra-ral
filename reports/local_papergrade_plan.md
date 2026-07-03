# Local Paper-Grade Experiment Plan

This plan defines how to move from scaffold validation toward paper-grade rollout and baseline evidence without leaving the home workflow. It does not authorize downloads, GPU jobs, training, heavy model imports, or real rollouts. Those actions remain behind explicit environment gates and separate tasks.

## Safety Gates

Heavy actions are disabled by default. Scripts and operators must require explicit gates before doing any of the following:

- `ALLOW_DOWNLOADS=1` before downloading checkpoints, datasets, or model cache files.
- `ALLOW_HEAVY_IMPORT=1` before importing OpenVLA-OFT, SmolVLA, or other heavy VLA stacks.
- `ALLOW_GPU_TRAINING=1` before any training or fine-tuning.
- `ALLOW_ROLLOUTS=1` before simulator rollouts.
- `ALLOW_CLOUD_HANDOFF=1` before packaging or launching remote/cloud work.

Default policy: plan, check, and write manifests only.

## Tier 1: Local-Now Feasible

Purpose: use the current Windows RTX 5080 machine for safe development and small offline evidence once assets are available locally.

Candidate work:

- SmolVLA asset readiness.
- SmolVLA adapter smoke, load-only and inference-only in a later task.
- TCA-Map head-only tiny pilot.
- ActionMap vs TCA-Map offline proxy.
- Counterfactual target-swap tiny dataset.

Required assets:

- `SMOLVLA_CKPT` local checkpoint/model directory.
- `HF_HOME` or `CHECKPOINT_ROOT` for local cache lookup.
- `DATA_ROOT` for tiny offline subset outputs.
- Optional `LIBERO_DATA_ROOT` if using a small local LIBERO-style offline subset.

Disk estimate:

- 20-50 GB for SmolVLA checkpoint/cache reserve.
- 5-20 GB for tiny offline subsets, generated counterfactual JSONL, and reports.
- Keep at least 50 GB free before attempting a real-adapter smoke.

VRAM estimate:

- Asset readiness checks: 0 GB.
- Dummy smoke/offline script checks: CPU-only.
- SmolVLA load-only smoke later: target 8-12 GB, leave at least 2 GB headroom.
- Head-only tiny pilot later: target under 12 GB with batch size 1 and low-resolution heatmaps.

RAM estimate:

- Current 24 GB is acceptable for scaffold and small offline work.
- 32 GB is marginal for model loading plus data transforms.
- 64 GB is the practical minimum upgrade target for comfortable local pilots.

Expected runtime:

- Asset readiness: seconds.
- Dummy smoke: seconds.
- Tiny offline proxy after implementation: minutes to under 1 hour.
- SmolVLA load-only smoke later: minutes if dependencies and local cache are correct.

Go/no-go criteria:

- Tree check, preflight, dummy train/eval smoke, and pytest pass.
- `ready_for_smolvla_smoke=true` from the asset checker.
- No required model or data download at runtime.
- VRAM headroom at least 2 GB for any later model load.
- No privileged simulator state in default inference.

Failure modes:

- Missing local checkpoint/tokenizer/config files.
- Python/CUDA/Torch stack not ready for RTX 5080.
- Windows path quoting issues.
- 24 GB RAM paging during model load.
- Offline proxy accidentally described as standard success; reports must avoid this.

## Tier 2: Local-After-Setup Feasible

Purpose: add WSL2/Linux simulator capability and small rollout evidence while keeping workloads conservative.

Candidate work:

- WSL2/Linux LIBERO install.
- Small LIBERO rollout.
- SmolVLA small rollout.
- OpenVLA-OFT frozen/load smoke.
- Low-resolution ActionMap/TCA-Map comparison.

Required assets:

- Tier 1 assets.
- `LIBERO_ROOT`, `LIBERO_DATA_ROOT`, and `ROBOSUITE_ROOT`.
- Working WSL2 Ubuntu or Linux environment.
- CUDA/PyTorch stack that sees RTX 5080 from WSL2/Linux.
- Local OpenVLA-OFT checkpoint/cache for frozen/load smoke only.

Disk estimate:

- 50-150 GB for LIBERO/RoboSuite data and simulator assets.
- 80-200 GB reserve for OpenVLA-OFT checkpoint/cache files.
- 150 GB free minimum; 300 GB free is safer.

VRAM estimate:

- Small SmolVLA rollout: target 10-14 GB, batch size 1.
- OpenVLA-OFT frozen/load smoke: high OOM risk on 16 GB; target under 14 GB before continuing.
- Low-resolution ActionMap/TCA-Map comparison: keep heatmaps coarse and batch size 1.

RAM estimate:

- 24 GB may work for tiny simulator checks but is tight.
- 64 GB recommended before local rollouts.
- 128 GB ideal for simulator plus large model/cache operations.

Expected runtime:

- WSL2/Linux setup checks: minutes.
- Simulator install/debug: hours, depending on dependencies.
- Tiny rollout: tens of minutes to a few hours.
- Frozen OpenVLA-OFT smoke: minutes if load succeeds, but debugging can take longer.

Go/no-go criteria:

- WSL2/Linux reports working Ubuntu and GPU visibility.
- LIBERO/RoboSuite imports and minimal simulator checks pass in a separate approved task.
- `ready_for_libero_rollout=true` from asset checker.
- Standard metric degradation is no more than 1-2 percentage points versus the strongest implemented local baseline.
- Robust/counterfactual gain is at least +10 percentage points.
- Wrong-target proxy or rollout rate reduces by at least 20 percent relative.
- No privileged inference.

Failure modes:

- WSL2 GPU passthrough or driver mismatch.
- MuJoCo/RoboSuite render failures.
- Simulator install drift across Windows and WSL paths.
- OpenVLA-OFT memory pressure on 16 GB VRAM.
- Low system RAM causing slow checkpoint load and simulator paging.

## Tier 3: Cloud/Remote Required

Purpose: produce large-baseline, multi-seed, and paper-grade evidence when local hardware is insufficient.

Candidate work:

- OpenVLA-OFT full baseline.
- Multi-seed LIBERO/RoboCasa sweep.
- Full ActionMap vs TCA-Map vs OpenVLA-OFT benchmark.
- Large ablation matrix.

Required assets:

- All Tier 2 assets.
- Remote Linux GPU environment with reproducible Python/CUDA stack.
- Explicit approval for model/data download or upload.
- Cloud handoff manifest with repo commit, configs, asset checklist, and commands.

Disk estimate:

- 300 GB minimum for focused OpenVLA-OFT and LIBERO baseline work.
- 500 GB to 1 TB recommended for multi-seed sweeps, checkpoints, logs, and RoboCasa/LIBERO variants.

VRAM estimate:

- 24 GB minimum for OpenVLA-OFT frozen/head-only smoke.
- 48 GB recommended for larger baseline comparisons.
- 80 GB recommended for multi-seed full baseline and larger ablations.

RAM estimate:

- 64 GB minimum.
- 128 GB recommended.
- 256 GB useful for large dataset staging and parallel evaluation workers.

Expected runtime:

- Frozen/load smoke: minutes to under 1 hour after environment setup.
- Single-seed baseline/pilot: hours to a day.
- Multi-seed full benchmark: multiple days depending on tasks, seeds, and rollouts.

Go/no-go criteria:

- Remote preflight, dummy smoke, real asset check, and simulator checks pass.
- No hidden downloads or provider-specific secrets in tracked configs.
- All run logs include git commit, config, random seed, dataset version, checkpoint path, GPU type, walltime, max memory, stdout/stderr, and metrics JSON.
- Full TCA-Map beats ActionMap + augmentation and target-head-only ablations before paper claims.

Failure modes:

- Asset transfer or cache mismatch between local and remote.
- Remote checkpoint download requires a green remote/cloud risk assessment and valid credentials that are not committed.
- Non-reproducible CUDA or simulator versions.
- Baseline implementation gaps invalidate SOTA-facing claims.
- Cost/time grows before a clear pilot signal appears.
