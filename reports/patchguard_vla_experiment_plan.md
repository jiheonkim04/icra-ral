# PatchGuard-VLA Experiment Plan

Date: 2026-07-09 KST

## STATE 0: Literature and Method Alignment

Deliverables:

- `reports/patchguard_vla_task_definition.md`
- `reports/patchguard_vla_related_work_matrix.md`
- `reports/patchguard_vla_experiment_plan.md`
- `reports/patchguard_vla_kill_criteria.md`
- `reports/patchguard_vla_autopilot_state.md`
- `reports/patchguard_vla_risk_register.md`

STATE 0 acceptance:

- novelty is framed around visual-proprioceptive and kinematic consistency, not LoRA,
- related work includes VLA-Hijack, partial-observation patch attacks, generic multimodal robustness, RobustVLA, OpenVLA/OpenVLA-OFT/SmolVLA, adversarial training, random erasing, cutout, and generic augmentation,
- no paper claims are made.

## STATE 1: Attack and Defense Feasibility Gate

Runner:

```powershell
$env:ALLOW_HEAVY_IMPORT="1"
$env:ALLOW_PATCHGUARD_VLA_STATE1="1"
powershell -ExecutionPolicy Bypass -File scripts\230_patchguard_vla_state1_diagnostic.ps1
Remove-Item Env:\ALLOW_HEAVY_IMPORT -ErrorAction SilentlyContinue
Remove-Item Env:\ALLOW_PATCHGUARD_VLA_STATE1 -ErrorAction SilentlyContinue
```

The runner writes:

- `reports/patchguard_vla_state1_result.json`
- `reports/patchguard_vla_state1_result.md`

Allowed in STATE 1:

- local SmolVLA CPU load and inference,
- local LIBERO HDF5 observation reads,
- clean, random patch, fixed visible patch, cutout defense proxy, and cheap visual augmentation proxy,
- action divergence metrics,
- EEF/proprio signal inventory.

Forbidden in STATE 1:

- training,
- full benchmark,
- simulator rollout,
- large downloads,
- OpenVLA-OFT,
- full adversarial patch optimization,
- GPU jobs,
- paper-grade claims.

## Required Baselines

| Baseline | STATE 1 implementation |
| --- | --- |
| clean observation | unchanged local LIBERO HDF5 sample |
| random patch | random visible square in agentview image |
| fixed visible patch | high-contrast checkerboard in agentview image |
| random erasing / cutout defense | mean-color cutout over the fixed patch region |
| generic visual augmentation proxy | cheap blocky smoothing and brightness proxy after fixed patch |
| PatchGuard signal availability | unchanged EEF/proprio state with image-only patch divergence |

## Required Metrics

- clean action,
- patched action divergence,
- EEF/action trajectory deviation proxy,
- target/action consistency proxy via expert-action alignment delta,
- arm/proprioception consistency proxy via unchanged EEF state under visual perturbation,
- whether patch effect is nontrivial,
- whether PatchGuard can compute a non-leaking kinematic signal.

## Exact STATE 1 Decision Set

The decision must be exactly one of:

- `READY_FOR_PATCHGUARD_LORA_SMOKE`
- `KILL_ATTACK_NOT_REPRODUCIBLE`
- `KILL_NO_KINEMATIC_SIGNAL`
- `KILL_BASELINE_DOMINATED`
- `TOO_HEAVY_LOCAL`
- `SOURCE_BLOCKED`

## Continue Gate

Continue only if:

- patch causes measurable action or trajectory degradation,
- kinematic/proprioceptive signal is available,
- random erasing/cutout does not trivially solve it,
- real LoRA/adapter training path is feasible.

## STATE 1B: Environment Unblock and Tiny Adapter Feasibility Gate

STATE 1B reclassifies the prior STATE 1 `TOO_HEAVY_LOCAL` as an installable environment blocker, not a method kill. It may install only missing or incompatible adapter/runtime packages needed for the gate: `peft`, `accelerate`, `bitsandbytes`, and `transformers`.

Runner:

```powershell
$env:ALLOW_HEAVY_IMPORT="1"
$env:ALLOW_PATCHGUARD_VLA_STATE1B="1"
$env:ALLOW_PATCHGUARD_TINY_LORA_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\231_patchguard_vla_state1b_probe.ps1 -MaxSteps 10 -DependencyInstallHappened
Remove-Item Env:\ALLOW_HEAVY_IMPORT -ErrorAction SilentlyContinue
Remove-Item Env:\ALLOW_PATCHGUARD_VLA_STATE1B -ErrorAction SilentlyContinue
Remove-Item Env:\ALLOW_PATCHGUARD_TINY_LORA_TRAINING -ErrorAction SilentlyContinue
```

Allowed in STATE 1B:

- Python, PyTorch, CUDA, GPU, `transformers`, `accelerate`, `peft`, and `bitsandbytes` checks,
- bitsandbytes CUDA 4-bit and 8-bit kernel smokes,
- PEFT dummy LoRA smoke,
- local SmolVLA LoRA injection dry run,
- one local LIBERO HDF5 sample,
- batch size 1, rank 4, at most 10-30 tiny optimization steps,
- clean, patched, cutout/random-erasing, generic adversarial LoRA, and PatchGuard kinematic-consistency LoRA metrics.

Forbidden in STATE 1B:

- full research training,
- full benchmark or simulator rollout,
- large model or dataset downloads,
- OpenVLA-OFT,
- paper-grade claims.

The STATE 1B decision must be exactly one of:

- `READY_FOR_PATCHGUARD_LORA_STATE2`
- `QLORA_BLOCKED_BUT_LORA_POSSIBLE`
- `ENV_BLOCKED_INSTALL_FAILED`
- `KILL_NO_ADAPTER_PATH`
- `KILL_BASELINE_DOMINATED`
- `TOO_HEAVY_LOCAL`

Current STATE 1B result: `KILL_BASELINE_DOMINATED`. PEFT, bitsandbytes, CUDA, local SmolVLA LoRA injection, and tiny training all worked, but PatchGuard did not beat both generic adversarial augmentation and cutout/random-erasing in the tiny smoke.

