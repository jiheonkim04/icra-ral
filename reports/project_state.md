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

The approval was limited to SmolVLA checkpoint acquisition from `lerobot/smolvla_base`. It did not approve GPU jobs, model inference, training, rollouts, heavy VLA imports, `ALLOW_HEAVY_IMPORT=1`, OpenVLA-OFT execution/download, dataset downloads, token/secret access, or committing checkpoint/cache files.

Current checker output after acquiring `lerobot/smolvla_base`:

```text
ready_for_smolvla_path_check=true
smolvla_checkpoint_files_present=false
ready_for_smolvla_adapter_smoke=false
ready_for_openvla_oft_smoke=false
ready_for_libero_rollout=false
```

Detected local files include `config.json`, `model.safetensors`, `policy_preprocessor.json`, `policy_postprocessor.json`, and processor safetensors. The readiness checker still reports `smolvla_checkpoint_files_present=false` because no repo-local tokenizer file is present.

`policy_preprocessor.json` references tokenizer/model source:

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

That external tokenizer/model source was not downloaded because the approved acquisition scope was only `lerobot/smolvla_base`. The current gate remains Case B from the self-check gate policy: path exists and config/weights exist, but the tokenizer file group is missing under the local readiness semantics.

Other missing assets currently expected:

- OpenVLA-OFT checkpoint,
- LIBERO source checkout,
- LIBERO data root,
- RoboSuite root.

## Current Blocker

The next real-adapter step is blocked by missing repo-local tokenizer files under:

```text
C:\assets\checkpoints\smolvla
```

Required groups:

- `config.json`,
- tokenizer file,
- weights file.

Codex should not ask whether these files were placed. It should run the readiness checkers, report exact missing file classes, and stop at the checkpoint-file gate until files are present or a dangerous gate is explicitly approved. Any future acquisition outside `lerobot/smolvla_base`, including the referenced `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` tokenizer/model, requires a separate explicit approval.

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

Codex should self-check current state. If tokenizer files remain missing, it should report the missing tokenizer file group and stop. If config/tokenizer/weights are present, it should verify readiness and prepare a load-only adapter smoke plan, but must not cross heavy import, GPU, download, rollout, simulator, token, or OpenVLA-OFT gates without explicit approval.
