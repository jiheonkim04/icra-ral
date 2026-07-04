# AGENTS.md

This repository is for the TCA-Map robot-learning research pilot.

Future Codex sessions should read `reports/codex_delegation_manual.md` first. The repository files, reports, configs, scripts, tests, and git history are the source of truth; do not rely on old ChatGPT conversation context.

Self-check routine state before asking the user. Branch, commit, git status, pytest, safe runner, SmolVLA path readiness, checkpoint file completeness, and checker policy fields should be inspected directly with existing scripts. Do not ask for routine approval merely because a task involves downloads, GPU, training, datasets, simulator readiness, learned-policy rollout, or benchmark rollout. Run the repository risk assessment first; proceed autonomously when source, size, disk, RAM/VRAM, runtime, dependency, license/token, and repo-policy checks are inside budget. Stop only when risk cannot be evaluated, exceeds budget, requires external irreversible action, requires OpenVLA-OFT execution, or would make an unsupported empirical claim.

## Non-negotiable rules

1. Do not fabricate results.
2. Do not hide failed runs.
3. Do not overwrite previous experiment outputs.
4. Do not use privileged simulator state at default inference time.
5. Simulator labels may be used only for training supervision, evaluation metrics, or oracle ablations.
6. Do not launch GPU jobs until preflight and dummy smoke tests pass.
7. Do not download OpenVLA-OFT automatically. Other downloads may proceed only after risk assessment confirms an official/documented unambiguous source, no token/login/payment/license click-through, budgeted size, approved target root, and enough disk margin.
8. If local assets are missing, keep dummy smoke and interface validation running and document setup in `reports/missing_assets.md`.
9. Do not call offline proxy metrics standard success. Use names such as `offline_standard_proxy` or `standard_proxy_score`.
10. Final paper-grade standard success requires simulator rollouts.
11. OpenVLA-OFT large experiments are forbidden on local hardware. OpenVLA-OFT download/import/load/execution remains blocked unless a separate OpenVLA risk budget is added later.
12. TCA-Select inference trick is required for the publishable low-compute method.
13. TCA-Select must be distributional for the final method, not only heuristic geometry. Heuristic target/action consistency is an ablation.
14. LoRA/QLoRA are required experimental tracks after the head-only path is validated, but they are supporting adaptation arms, not the main novelty.
15. Any SOTA claim must be restricted to low-compute target-conditioned action decoding or counterfactual robustness unless full standard baselines are directly reproduced.
16. Heavy actions must have a short automatic risk assessment before launch. Environment gates such as `ALLOW_DOWNLOADS=1`, `ALLOW_HEAVY_IMPORT=1`, `ALLOW_TINY_TRAINING=1`, `ALLOW_GPU_TRAINING=1`, `ALLOW_ROLLOUTS=1`, or `ALLOW_CLOUD_HANDOFF=1` may be set task-locally only when the risk assessment says proceed.
17. Run compute-budget enforcement before any new local config or pilot command.

## Low-compute protocol

The local publishable path is SmolVLA-first: frozen backbone, head-only ActionMap/TCA-Map training, cached hidden features, low-resolution or coarse-to-fine heatmaps, Distributional TCA-Select inference-time candidate selection, and required LoRA/QLoRA experimental tracks after head-only validation.

Do not plan local OpenVLA-OFT full fine-tuning, full rollout, multi-seed sweep, or large ActionMap/TCA-Map training.

## Risk-assessed autonomous execution policy

Codex must not ask the user for routine approval when risk can be checked automatically. Inspect source, disk, RAM, VRAM, runtime, dependency, license/token requirements, and repo safety policy. If all checks pass within budget, proceed autonomously. If any check is ambiguous or outside budget, stop and report the blocker.

Default local risk budgets:

- downloads: source official/documented/unambiguous, no token/login/payment/license click-through, single task soft limit 80GB, keep at least 100GB free disk, write only under approved roots such as `C:\assets`, never commit checkpoint/cache/data files. Official LIBERO data is the only current exception: `yifengzhu-hf/LIBERO-datasets` may use a 180GB task budget only if at least 250GB free disk remains after acquisition and no token/login/payment/license click-through is required,
- GPU: SmolVLA/local-pilot related, no OpenVLA-OFT, expected VRAM <=14GB, runtime <=30 minutes, batch size 1 or equivalent, timeout/stop condition, memory/runtime logged when measurable,
- training: SmolVLA-only, frozen backbone or LoRA/QLoRA adapter only, no full fine-tuning, no rollout, max 300 local pilot steps after smaller smoke is stable, runtime <=30 minutes, VRAM <=14GB, batch size 1, proxy/local-pilot labels only,
- real datasets: official/documented/unambiguous source, no token/login/payment/license click-through, inside download/disk budget, no simulator rollout triggered automatically, prefer metadata-only or tiny subset first,
- simulator readiness: prefer WSL2/Linux, no token/manual license, runtime <=10 minutes for import/render smoke, no policy rollout, no paper claim. Minimal WSL Python packaging setup is standing-approved after risk assessment; credentialed/system-driver/license-gated changes remain hard-stop,
- bounded rollout: only after simulator import/render/reset-step smoke passes, task count <=5 for the first local benchmark rung, runtime <=30 minutes, no OpenVLA-OFT, no external service/token, no unbounded render loop, and no unsupported claim. This covers toy MuJoCo diagnostics, LIBERO/RoboSuite zero-action diagnostics, and later tiny learned-policy or benchmark rollouts when their own risk assessment is green.

Always stop before token/secret/API key access, paid services, license click-through, external upload/submission/publishing, deleting user files outside approved cache/repo cleanup, system-wide CUDA/PyTorch/driver changes, credentialed/system-driver/license-gated system setup, OpenVLA-OFT execution, or unsupported paper-level empirical claims. Paper-grade candidate reports are allowed only from verified experiment outputs with honest evidence labels. Minimal WSL Python packaging setup may proceed autonomously only after a green WSL simulator dependency-ladder risk assessment.

Before any bounded download/GPU/training/dataset/simulator step, write or print a short risk assessment with task, source, expected size, target path, disk free before/after estimate, expected runtime, expected RAM/VRAM, allowed budget, official/documented source status, token/license/payment status, decision, and reason.

## Local paper-grade runner protocol

Preserve the Windows PowerShell scripts and Linux/WSL shell scripts that perform readiness checks, asset directory planning, local experiment matrix planning, cloud handoff manifest generation, WSL2 setup checks, compute-budget enforcement, and risk assessment. These scripts are planning/readiness tools unless a risk assessment says a bounded heavy action is inside budget.

## Required first milestone

1. Scaffold repository structure.
2. Run repository-local preflight.
3. Run dummy smoke train/eval only if preflight passes.
4. Skip real OpenVLA-OFT, SmolVLA, LIBERO, RoboCasa, simulator, and rollout work unless local paths and safety checks pass in a later step.

## Local path policy

Read optional local paths from `configs/paths.local.yaml` or environment variables. Never commit `configs/paths.local.yaml` if it contains machine paths or tokens.
