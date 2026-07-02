# Project State

## Repository State

Canonical repository root:

```text
C:\Users\jiheo\tca_map
```

GitHub repository:

```text
jiheonkim04/icra-ral
```

Canonical branch:

```text
main
```

Current main commit at this update:

```text
b09b684dcb0bf96910f2a2e3a9d604da97174e1b
```

Use explicit Python for validation:

```text
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe
```

## Completed Major Features

- scaffold and dummy smoke,
- preflight and smoke reports,
- compute budget guard,
- no-large local OpenVLA-OFT policy,
- Distributional TCA-Select scaffold,
- LoRA/QLoRA config guards,
- local paper-grade runner and planning scripts,
- Cursor safe local runner,
- SmolVLA asset prep,
- SmolVLA readiness semantics split,
- SmolVLA download plan guard,
- Windows Bash shim handling for Bash-specific tests,
- manual SmolVLA acquisition checklist,
- Codex delegation manual and project state files.

## Current Asset Status

Known current state:

```text
SMOLVLA_CKPT=C:\assets\checkpoints\smolvla
CHECKPOINT_ROOT=C:\assets\checkpoints
HF_HOME=C:\assets\hf_home
```

The SmolVLA checkpoint directory exists, but required checkpoint files are missing.

Current checker output:

```text
ready_for_smolvla_path_check=true
smolvla_checkpoint_files_present=false
ready_for_smolvla_adapter_smoke=false
ready_for_openvla_oft_smoke=false
ready_for_libero_rollout=false
```

The current gate is Case B from the self-check gate policy: path exists, but config/tokenizer/weights are missing.

Other missing assets currently expected:

- OpenVLA-OFT checkpoint,
- LIBERO source checkout,
- LIBERO data root,
- RoboSuite root.

## Current Blocker

The next real-adapter step is blocked by missing SmolVLA checkpoint files under:

```text
C:\assets\checkpoints\smolvla
```

Required groups:

- `config.json`,
- tokenizer file,
- weights file.

Codex should not ask whether these files were placed. It should run the readiness checkers, report exact missing file classes, and stop at the checkpoint-file gate until files are present or a dangerous gate is explicitly approved.

## Validation Commands

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\40_cursor_safe_local_check.ps1
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\13_check_smolvla_adapter_smoke.ps1
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest -q
```

Relevant dry-run planner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\14_plan_smolvla_download.ps1
```

Codex should run these commands itself when routine state is needed. The user should only be asked at dangerous gates.

## Safety Gates

Forbidden until explicitly approved:

- automatic downloads,
- setting `ALLOW_DOWNLOADS=1`,
- setting `ALLOW_HEAVY_IMPORT=1`,
- GPU inference,
- GPU training,
- rollouts,
- simulator execution,
- heavy VLA imports,
- OpenVLA-OFT execution,
- token or secret handling,
- paper-level empirical claims.

## Research Direction Summary

TCA-Map / Distributional TCA-Map / Distributional TCA-Select is a low-compute VLA action-decoding and counterfactual grounding project.

Main hypothesis:

A VLA should first ground an instruction to a target distribution, then decode a target-conditioned action heatmap. Counterfactual target changes should shift target/action distributions consistently. Nuisance or paraphrase changes that preserve the target should keep distributions stable.

Core method:

- target heatmap / target distribution,
- target-conditioned ActionMap head,
- Distributional TCA-Select,
- counterfactual target/action consistency,
- nuisance invariance,
- optional LoRA/QLoRA only as compute-saving support.

## Immediate Next Step

Codex should self-check current state. If checkpoint files remain missing, it should report the missing file classes and stop. If checkpoint files are present, it should verify readiness and prepare a load-only adapter smoke plan, but must not cross heavy import, GPU, download, rollout, simulator, token, or OpenVLA-OFT gates without explicit approval.
