# Risk-Gate Status

## Purpose

This report explains the current bounded-autopilot stop condition in one place.

The project has completed the safe scaffolds, planners, readiness checks, dummy cache path, cached-feature eval-only smoke, tiny head-only pilot planner, bounded tiny head-only smoke runner, go/no-go status summary scaffold, the bounded SmolVLA runtime package install, and the SmolVLA autonomous pilot risk policy. Larger experimental stages require a green risk assessment and must stop at external irreversible, OpenVLA-OFT, token/license/payment, system-level, or paper-claim gates.

## Summary Command

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\27_summarize_hard_stop_status.ps1
```

It writes an ignored runtime report:

```text
reports\hard_stop_status_report.json
```

The script is summary-only. It does not install packages, download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, access tokens, execute simulators, execute OpenVLA-OFT, or make paper-grade claims.

## SmolVLA Autonomous Pilot Risk Envelope

The runtime install gate has been used and is currently clear: `torch`, `transformers`, `lerobot`, `safetensors`, and `num2words` are installed in the `tca_map` environment.

The bounded SmolVLA load-only smoke has passed on CPU. It loaded local policy weights with `load_vlm_weights=false` and did not run model inference, training, rollouts, OpenVLA-OFT, downloads, or token access.

The bounded single-sample interface smoke has also passed on CPU. It used one synthetic observation and one `select_action` call, produced a finite `[1, 6]` action tensor, and did not train, rollout, evaluate datasets, execute OpenVLA-OFT, download assets, access tokens, or make paper claims.

The dummy feature-cache/interface validation has passed. It wrote and validated a dummy cache and ran eval-only cached-feature metrics without heavy model import, model inference, training, rollout, simulator execution, OpenVLA-OFT, downloads, or token access.

The bounded tiny head-only smoke has passed on cached/dummy features. It trained tiny CPU NumPy heads for 16 steps and did not use GPU, import or load SmolVLA/OpenVLA, run VLA inference, rollout, execute simulators, download assets, or make paper claims.

Codex may continue autonomously through a green risk assessment for:

- SmolVLA load-only heavy import/model construction smoke with `ALLOW_HEAVY_IMPORT=1` set only inside that task,
- load-only debugging for dependency/import/API/layout/Windows/minor compatibility issues,
- one synthetic or dummy single-sample interface smoke with `ALLOW_SINGLE_SAMPLE_INFERENCE=1` set only inside that task,
- tiny feature-cache/interface validation,
- tiny head-only training smoke with frozen backbone, max 300 steps after stable smaller smoke, max 30 minutes, max 14GB VRAM, and no paper claim.

Do not combine this risk envelope with true external stop gates.

## Current Known Blockers

- Runtime dependencies are present in the current `tca_map` environment.
- The bounded tiny head-only smoke has passed within its smoke caps, frozen backbone/cached-feature, no GPU job, no rollout, no simulator, no OpenVLA-OFT, and no paper claim.
- LIBERO/RoboSuite/RoboCasa/simulator assets remain missing for rollout work.
- OpenVLA-OFT local large execution remains forbidden.

## Remaining True Hard-Stops

Codex must still stop before:

- OpenVLA-OFT download/import/load/execution,
- LIBERO/RoboSuite/RoboCasa/dataset download without passing source/size/license/token/disk risk assessment,
- simulator execution without passing readiness risk assessment,
- rollout without passing bounded rollout risk assessment,
- real benchmark evaluation that could be mistaken for paper-grade evidence,
- training longer than 30 minutes or more than 300 steps after stable smaller smoke,
- any job expected to exceed 30 minutes,
- using more than 14GB VRAM,
- changing CUDA/PyTorch major versions,
- installing large unplanned packages,
- token/secret/API key/login requirement,
- multi-seed experiment,
- paper-level empirical claim,
- external submission/upload/publishing,
- destructive file deletion outside repository or approved cache cleanup.

## Safe Commands Still Allowed

These remain safe routine checks:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\40_cursor_safe_local_check.ps1
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\13_check_smolvla_adapter_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\17_check_smolvla_runtime_deps.ps1
powershell -ExecutionPolicy Bypass -File scripts\18_plan_smolvla_runtime_install.ps1
powershell -ExecutionPolicy Bypass -File scripts\26_plan_tiny_head_only_pilot.ps1
powershell -ExecutionPolicy Bypass -File scripts\29_tiny_head_only_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\31_generate_go_no_go_report.ps1
```

Set `ALLOW_TINY_TRAINING=1` only for the bounded tiny head-only smoke task, then remove it after the command finishes.
