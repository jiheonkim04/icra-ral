# Low-Compute Experiment Plan

This plan avoids local OpenVLA-OFT large runs. It is designed for SmolVLA-first development, cached features, frozen backbones, and head-only ActionMap/TCA-Map training.

## Stage A: SmolVLA Real Adapter Smoke

Goal: validate real adapter plumbing with the smallest practical public backbone path.

Rules:

- Load/check only.
- No training.
- No rollout.
- No downloads unless a separate task explicitly sets `ALLOW_DOWNLOADS=1`.
- No OpenVLA-OFT.

Success criteria:

- `SMOLVLA_CKPT` resolves locally.
- Adapter returns hidden tokens and action-shaped outputs on a tiny sample.
- VRAM headroom is at least 2 GB.
- Batch size is 1.

## Stage B: Feature Cache Pilot

Goal: remove repeated backbone cost from local head experiments.

Protocol:

- Run frozen SmolVLA encoder on a tiny dataset only after an explicit heavy-import/model-smoke task is approved.
- Cache hidden tokens and required metadata.
- Train only native, ActionMap, and TCA-Map heads on cached features.
- Do not use OpenVLA-OFT.

Constraints:

- Frozen backbone.
- Batch size 1 during feature extraction.
- Cached feature files are versioned by dataset version, checkpoint path, seed, and config hash.
- No privileged simulator state at default inference.

## Stage C: Tiny Offline Counterfactual Pilot

Goal: test whether TCA-Map improves target-conditioned action decoding under counterfactual language/target swaps before simulator rollouts.

Compare:

- SmolVLA native head.
- ActionMap.
- ActionMap + counterfactual augmentation.
- TCA-Map.

Metrics:

- `offline_standard_proxy`.
- Target top-1/top-k accuracy.
- Wrong-target proxy rate.
- Counterfactual separation margin.
- Nuisance stability.
- Latency.
- VRAM.
- Trainable parameters.

Rules:

- Do not call offline proxy standard success.
- Keep heatmap grid initial size at or below 8.
- Keep trainable parameters at or below 50M initially.
- Keep local pilot steps at or below 1000 initially.

## Stage D: Small Rollout After WSL2/LIBERO Passes

Goal: produce the first paper-relevant simulator evidence.

Compare:

- SmolVLA native.
- ActionMap.
- TCA-Map.

Requirements:

- WSL2/Linux simulator checks pass.
- `LIBERO_ROOT`, `LIBERO_DATA_ROOT`, and `ROBOSUITE_ROOT` resolve.
- `ALLOW_ROLLOUTS=1` is explicitly set in a separate rollout task.
- Rollouts are tiny and single-seed first.

Metrics:

- LIBERO rollout success rate.
- Task success.
- Counterfactual rollout success where measurable.
- Wrong-target rollout rate where measurable.
- Latency and max VRAM.

## Stage E: Optional OpenVLA-OFT Frozen Smoke

Goal: determine whether OpenVLA-OFT can be loaded locally for interface and memory feasibility only.

Rules:

- Load/interface only.
- No training.
- No rollout.
- No performance claim.
- No full fine-tuning.
- No multi-seed sweep.
- No large local baseline.

Outcome:

- If it loads safely, record it as feasibility context.
- If it fails, move OpenVLA-OFT baseline work to WSL2/Linux or cloud/remote GPU.
- Keep OpenVLA-OFT as a paper-grade reference target, not as a local-large experiment path.
