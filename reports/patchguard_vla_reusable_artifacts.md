# PatchGuard-VLA Reusable Artifacts

Date: 2026-07-09 KST

## Reuse Boundary

These artifacts are reusable as diagnostics and local LoRA infrastructure. They are not evidence that PatchGuard-VLA is RA-L-stable, and they do not authorize PatchGuard STATE 2.

## Reusable Code

- `tca_map/patchguard_vla/diagnostic.py`
  - local SmolVLA patch-sensitivity diagnostic,
  - clean/random/fixed patch/cutout/generic visual augmentation variants,
  - action-divergence metrics,
  - non-leaking kinematic/proprioceptive signal checks,
  - gated report writer.
- `scripts/230_patchguard_vla_state1_diagnostic.ps1`
  - bounded STATE 1 runner,
  - heavy-import and PatchGuard gates,
  - no training, rollout, download, OpenVLA-OFT, or paper claim.
- `tca_map/patchguard_vla/state1b.py`
  - dependency status checks,
  - PEFT dummy LoRA smoke,
  - bitsandbytes 4-bit and 8-bit CUDA smokes,
  - local SmolVLA LoRA injection path,
  - tiny rank-4 batch-size-1 training smoke,
  - standard LoRA, generic adversarial LoRA, PatchGuard LoRA, and cutout/random-erasing comparisons.
- `scripts/231_patchguard_vla_state1b_probe.ps1`
  - gated STATE 1B environment and tiny adapter runner.
- `tests/test_patchguard_vla_diagnostic.py`
  - focused tests for patch variant leakage behavior, STATE 1 decision priority, STATE 1B gate refusal, and exact STATE 1B decision labels.

## Reusable Reports

- `reports/patchguard_vla_state1_result.md`
- `reports/patchguard_vla_state1_result.json`
- `reports/patchguard_vla_state1b_result.md`
- `reports/patchguard_vla_state1b_result.json`
- `reports/patchguard_vla_task_definition.md`
- `reports/patchguard_vla_experiment_plan.md`
- `reports/patchguard_vla_kill_criteria.md`
- `reports/patchguard_vla_autopilot_state.md`
- `reports/patchguard_vla_risk_register.md`

## Reusable Evidence

- Patch effect measurable:
  - max attacked policy-action L1 vs clean `0.181765`,
  - max attacked translation-action L2 vs clean `0.213965`.
- Kinematic/proprioceptive signal available.
- PEFT/bitsandbytes/CUDA path works:
  - PEFT `0.19.1`,
  - bitsandbytes `0.49.2`,
  - bitsandbytes 4-bit and 8-bit CUDA smokes passed,
  - PyTorch `2.10.0+cu128`,
  - CUDA runtime `12.8`,
  - NVIDIA GeForce RTX 5080.
- SmolVLA LoRA injection works:
  - model path `C:\assets\checkpoints\smolvla`,
  - target modules `state_proj`, `action_in_proj`, `action_out_proj`,
  - trainable params `9984`.
- Tiny training resource data:
  - VRAM peak `2224.845` MB,
  - runtime `57.438` sec,
  - batch size 1,
  - rank 4,
  - 10 steps per variant.

## What Must Not Be Reused As A Claim

- Do not claim PatchGuard is a successful defense.
- Do not claim RA-L-ready robustness from these offline proxy metrics.
- Do not report the tiny LoRA smoke as a benchmark.
- Do not treat LoRA working locally as method novelty.
- Do not start a renamed PatchGuard-style method before a standard LoRA baseline is reproduced and understood.
