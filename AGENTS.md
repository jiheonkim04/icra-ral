# AGENTS.md

This repository is for the TCA-Map robot-learning research pilot.

Future Codex sessions should read `reports/codex_delegation_manual.md` first. The repository files, reports, configs, scripts, tests, and git history are the source of truth; do not rely on old ChatGPT conversation context.

Self-check routine state before asking the user. Branch, commit, git status, pytest, safe runner, SmolVLA path readiness, checkpoint file completeness, and checker policy fields should be inspected directly with existing scripts. The bounded SmolVLA autonomous pilot path has standing approval; ask only at true hard-stop gates such as OpenVLA-OFT execution, dataset/simulator downloads, rollouts, long or large training, more than 14GB VRAM, major CUDA/PyTorch changes, unplanned large package installs, secrets, or paper-level empirical claims.

## Non-negotiable rules

1. Do not fabricate results.
2. Do not hide failed runs.
3. Do not overwrite previous experiment outputs.
4. Do not use privileged simulator state at default inference time.
5. Simulator labels may be used only for training supervision, evaluation metrics, or oracle ablations.
6. Do not launch GPU jobs until preflight and dummy smoke tests pass.
7. Do not download OpenVLA-OFT, SmolVLA, LIBERO, RoboCasa, checkpoints, or datasets automatically.
8. If local assets are missing, keep dummy smoke and interface validation running and document setup in `reports/missing_assets.md`.
9. Do not call offline proxy metrics standard success. Use names such as `offline_standard_proxy` or `standard_proxy_score`.
10. Final paper-grade standard success requires simulator rollouts.
11. OpenVLA-OFT large experiments are forbidden on local hardware. OpenVLA-OFT may only be used for frozen/load smoke unless a separate explicit approval branch changes this policy.
12. TCA-Select inference trick is required for the publishable low-compute method.
13. TCA-Select must be distributional for the final method, not only heuristic geometry. Heuristic target/action consistency is an ablation.
14. LoRA/QLoRA are required experimental tracks after the head-only path is validated, but they are supporting adaptation arms, not the main novelty.
15. Any SOTA claim must be restricted to low-compute target-conditioned action decoding or counterfactual robustness unless full standard baselines are directly reproduced.
16. Heavy actions require explicit environment gates such as `ALLOW_DOWNLOADS=1`, `ALLOW_HEAVY_IMPORT=1`, `ALLOW_TINY_TRAINING=1`, `ALLOW_GPU_TRAINING=1`, `ALLOW_ROLLOUTS=1`, or `ALLOW_CLOUD_HANDOFF=1`; bounded SmolVLA load-only smoke may set `ALLOW_HEAVY_IMPORT=1`, and bounded tiny head-only smoke may set `ALLOW_TINY_TRAINING=1`, only inside the standing-approved autonomous pilot budget.
17. Run compute-budget enforcement before any new local config or pilot command.

## Low-compute protocol

The local publishable path is SmolVLA-first: frozen backbone, head-only ActionMap/TCA-Map training, cached hidden features, low-resolution or coarse-to-fine heatmaps, Distributional TCA-Select inference-time candidate selection, and required LoRA/QLoRA experimental tracks after head-only validation.

Do not plan local OpenVLA-OFT full fine-tuning, full rollout, multi-seed sweep, or large ActionMap/TCA-Map training.

## SmolVLA autonomous pilot standing approval

Codex may autonomously continue through the expected low-compute SmolVLA pilot steps without asking the user to approve each one:

- load-only SmolVLA heavy import/model construction smoke from local files,
- load-only debugging for missing dependencies, import paths, API mismatch, local file layout, Windows path issues, and minor compatibility fixes,
- one synthetic or dummy single-sample interface smoke,
- tiny feature-cache/interface validation,
- tiny head-only training smoke with frozen backbone, dummy or tiny non-paper data, max 100 steps, max 15 minutes, max 14GB VRAM, and no rollout/OpenVLA-OFT/paper claim,
- LoRA/QLoRA planning, config validation, and adapter-construction scaffolds that do not train, download, import heavy VLA models, rollout, or execute OpenVLA-OFT.

Stop before OpenVLA-OFT download/import/load/execution, LIBERO/RoboSuite/RoboCasa/dataset download, simulator execution, rollout, real benchmark evaluation, multi-seed experiments, training beyond the tiny-smoke budget, jobs expected over 30 minutes, more than 14GB VRAM, major CUDA/PyTorch changes, unplanned large package installs, token/secret access, external submission/upload/publishing, or paper-level empirical claims.

## Local paper-grade runner protocol

Preserve the Windows PowerShell scripts and Linux/WSL shell scripts that perform readiness checks, asset directory planning, local experiment matrix planning, cloud handoff manifest generation, WSL2 setup checks, and compute-budget enforcement. These scripts are planning/readiness tools unless a later task explicitly authorizes a heavy action gate.

## Required first milestone

1. Scaffold repository structure.
2. Run repository-local preflight.
3. Run dummy smoke train/eval only if preflight passes.
4. Skip real OpenVLA-OFT, SmolVLA, LIBERO, RoboCasa, simulator, and rollout work unless local paths and safety checks pass in a later step.

## Local path policy

Read optional local paths from `configs/paths.local.yaml` or environment variables. Never commit `configs/paths.local.yaml` if it contains machine paths or tokens.
