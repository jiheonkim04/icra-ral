# SmolVLA Load-Only Adapter Smoke Plan

## Purpose

This plan prepares the next SmolVLA step after file readiness became true. It is a planning and guard artifact only. It is not a model load, not inference, not training, not a rollout, and not paper-grade evidence.

## Current State

Readiness checks now report:

```text
ready_for_smolvla_path_check=true
smolvla_checkpoint_files_present=true
ready_for_smolvla_adapter_smoke=true
```

Local assets:

```text
SMOLVLA_CKPT=C:\assets\checkpoints\smolvla
HF_HOME=C:\assets\hf_home
CHECKPOINT_ROOT=C:\assets\checkpoints
```

The SmolVLA checkpoint source is `lerobot/smolvla_base`. The tokenizer/processor dependency source is `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, retained as tokenizer/processor/config files only.

## Planning Command

Run the planning-only script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\15_plan_smolvla_load_only_smoke.ps1
```

It writes an ignored runtime report:

```text
reports\smolvla_load_only_smoke_plan_report.json
```

## Next Gate

Actual load-only smoke remains behind a dangerous gate:

```text
ALLOW_HEAVY_IMPORT=1
```

Do not set this gate during planning. A later task must explicitly approve heavy import/model load and define the exact execution script.

## Future Load-Only Scope

When separately approved, the load-only smoke should:

- import only the minimal SmolVLA/LeRobot stack required to instantiate or load the policy,
- load local files only,
- record wall time and max GPU memory,
- exit without inference,
- exit without training,
- exit without optimizer creation,
- exit without rollouts or simulator imports,
- exit without OpenVLA-OFT execution,
- report any missing package, CUDA, memory, or file-layout issue exactly.

## No-Go Conditions

Stop if:

- a runtime download is attempted,
- token or secret access is requested,
- full SmolVLM2 model weights are requested unexpectedly,
- OpenVLA-OFT is imported or executed,
- LIBERO/RoboSuite/RoboCasa/dataset code is pulled in,
- the task attempts inference, training, rollout, or simulator execution,
- max memory cannot be measured or projected safely.
