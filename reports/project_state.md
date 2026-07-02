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
bd069a82c1f1b268dbfd9c21a9c18dd1b2ccc448
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

The SmolVLA checkpoint directory exists. The approved SmolVLA source has been acquired from `lerobot/smolvla_base`.

Approved SmolVLA checkpoint source for the acquisition task:

```text
lerobot/smolvla_base
```

Acquisition target:

```text
C:\assets\checkpoints\smolvla
```

Cache target:

```text
C:\assets\hf_home
```

The approved tokenizer/processor dependency source has also been acquired:

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

Dependency target:

```text
C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct
```

Only tokenizer/processor/config files are retained for this dependency. Full SmolVLM2 model weights were avoided.

The approvals were limited to SmolVLA checkpoint acquisition from `lerobot/smolvla_base` and tokenizer/processor/config dependency acquisition from `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. They did not approve GPU jobs, model inference, training, rollouts, heavy VLA imports, `ALLOW_HEAVY_IMPORT=1`, OpenVLA-OFT execution/download, dataset downloads, token/secret access, or committing checkpoint/cache files.

Current checker output after acquiring `lerobot/smolvla_base` and its tokenizer/processor dependency:

```text
ready_for_smolvla_path_check=true
smolvla_checkpoint_files_present=true
ready_for_smolvla_adapter_smoke=true
ready_for_openvla_oft_smoke=false
ready_for_libero_rollout=false
```

Detected checkpoint files include `config.json`, `model.safetensors`, `policy_preprocessor.json`, `policy_postprocessor.json`, and processor safetensors.

`policy_preprocessor.json` references tokenizer/model source:

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

The external tokenizer/processor dependency is now detected under `C:\assets\hf_home`. The current gate is Case C from the self-check gate policy: readiness is true, but the next load-only adapter smoke still requires separate explicit approval before any heavy import or model load.

Other missing assets currently expected:

- OpenVLA-OFT checkpoint,
- LIBERO source checkout,
- LIBERO data root,
- RoboSuite root.

## Current Gate

The next real-adapter step is a separately approved SmolVLA load-only adapter smoke. It remains blocked by the heavy import/model-load gate, not by file readiness.

```text
C:\assets\checkpoints\smolvla
```

Ready file groups:

- `config.json`,
- external tokenizer/processor/config dependency files,
- weights file.

Codex should not ask whether these files were placed. It should run the readiness checkers. Because readiness is true, Codex may prepare a load-only adapter smoke plan, but must stop before setting `ALLOW_HEAVY_IMPORT=1`, importing SmolVLA/SmolVLM2, running inference, using GPU execution, training, rollouts, simulator execution, token access, or OpenVLA-OFT.

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

Codex should self-check current state. Since config/tokenizer dependency/weights are present and adapter-smoke readiness is true, prepare the next safe load-only adapter smoke plan, but do not cross heavy import, GPU, download, rollout, simulator, token, or OpenVLA-OFT gates without explicit approval.
