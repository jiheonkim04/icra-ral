# LIBERO-Safety Go/No-Go

Date: 2026-07-09

Branch: `codex/libero-safety-official-feasibility`

Final decision: `TOO_HEAVY_LOCAL`

## Decision Options

- `GO_LIBERO_SAFETY_MINI_REPRO`: not selected.
- `SOURCE_BLOCKED`: not selected.
- `TOO_HEAVY_LOCAL`: selected.
- `NO_CLEAR_METHOD_GAP`: not selected.
- `STOP_VLA_METHOD_SEARCH_UNDER_CURRENT_CONSTRAINTS`: not selected.

## Why Not GO

The smallest credible official mini reproduction still requires official
LIBERO-Safety assets, simulator setup, rollout/evaluation, and likely an official
checkpoint or official baseline path. The assets archive alone is 10.67 GB, and
the released pi0.5 model metadata totals 12.44 GB. These requirements violate
the current no-large-download and no-rollout constraints.

## Why Not Source Blocked

Official sources are available:

- Paper: https://arxiv.org/abs/2606.23686
- Project page: https://libero-safety.github.io/
- Code: https://github.com/LIBERO-SAFETY/LIBERO-Safety
- Dataset: https://huggingface.co/datasets/LIBERO-Safety/libero_safety
- Assets: https://huggingface.co/datasets/LIBERO-Safety/libero_safety_assets/tree/main
- Model: https://huggingface.co/LIBERO-Safety/pi05_libero_safety/tree/main

The data generation pipeline is marked as coming soon, and no release tag or
metric-only logs were found, but source access itself is not blocked.

## Why Not No Clear Method Gap

The benchmark exposes more than collision-free imitation: physical safety,
semantic safety, success rate, collision rate, refusal rate, balanced semantic
failure modes, robustness, and collision-free incompletion are all visible in
official materials. That is enough to preserve a possible method gap after
official reproduction is feasible.

No method is authorized here. Future method work would still need to beat simple
safety-only, stop-on-risk, clipping/no-op, generic SFT/DPO, and adapter/LoRA
baselines.

## Future GO Requirements

A future `GO_LIBERO_SAFETY_MINI_REPRO` would require explicit approval to:

- Download the official assets and any required official checkpoint.
- Install the official LIBERO-Safety fork and simulator stack.
- Run a bounded official rollout/evaluation subset.
- Record official metrics without introducing local proxy metrics.

Under the current constraints, there is no valid execution next step.
