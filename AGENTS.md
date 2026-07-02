# AGENTS.md

This repository is for the TCA-Map robot-learning research pilot.

Future Codex sessions should read `reports/codex_delegation_manual.md` first. The repository files, reports, configs, scripts, tests, and git history are the source of truth; do not rely on old ChatGPT conversation context.

Self-check routine state before asking the user. Branch, commit, git status, pytest, safe runner, SmolVLA path readiness, checkpoint file completeness, and checker policy fields should be inspected directly with existing scripts. Ask only at dangerous gates such as downloads, heavy imports, GPU inference/training, rollouts, simulator execution, OpenVLA-OFT execution, secrets, or paper-level empirical claims.

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
14. LoRA/QLoRA are supporting tools, not the main novelty.
15. Any SOTA claim must be restricted to low-compute target-conditioned action decoding or counterfactual robustness unless full standard baselines are directly reproduced.
16. Heavy actions require explicit environment gates such as `ALLOW_DOWNLOADS=1`, `ALLOW_HEAVY_IMPORT=1`, `ALLOW_GPU_TRAINING=1`, `ALLOW_ROLLOUTS=1`, or `ALLOW_CLOUD_HANDOFF=1`.
17. Run compute-budget enforcement before any new local config or pilot command.

## Low-compute protocol

The local publishable path is SmolVLA-first: frozen backbone, head-only ActionMap/TCA-Map training, cached hidden features, low-resolution or coarse-to-fine heatmaps, Distributional TCA-Select inference-time candidate selection, and optional LoRA/QLoRA only for small adapters if needed.

Do not plan local OpenVLA-OFT full fine-tuning, full rollout, multi-seed sweep, or large ActionMap/TCA-Map training.

## Local paper-grade runner protocol

Preserve the Windows PowerShell scripts and Linux/WSL shell scripts that perform readiness checks, asset directory planning, local experiment matrix planning, cloud handoff manifest generation, WSL2 setup checks, and compute-budget enforcement. These scripts are planning/readiness tools unless a later task explicitly authorizes a heavy action gate.

## Required first milestone

1. Scaffold repository structure.
2. Run repository-local preflight.
3. Run dummy smoke train/eval only if preflight passes.
4. Skip real OpenVLA-OFT, SmolVLA, LIBERO, RoboCasa, simulator, and rollout work unless local paths and safety checks pass in a later step.

## Local path policy

Read optional local paths from `configs/paths.local.yaml` or environment variables. Never commit `configs/paths.local.yaml` if it contains machine paths or tokens.
