# Project State

## Repository State

Canonical repository root:

```text
C:\Users\jiheo\tca_map
```

Canonical branch:

```text
main
```

Latest known validated main commit before this checklist branch:

```text
982160c89e62ec3f871ee39fe46c277de0949d5c
```

## Completed

- scaffold and dummy smoke,
- compute budget guard,
- no-large OpenVLA-OFT local policy,
- Distributional TCA-Select scaffold,
- LoRA/QLoRA config guards,
- Cursor safe local runner,
- SmolVLA asset prep,
- SmolVLA readiness semantics split,
- SmolVLA download plan guard,
- Windows Bash shim handling for Bash-specific tests.

## Current Blocker

The local SmolVLA checkpoint directory exists, but required checkpoint files are not present.

Current readiness expectation:

```text
ready_for_smolvla_path_check=true
smolvla_checkpoint_files_present=false
ready_for_smolvla_adapter_smoke=false
```

## Active Safety Policy

The project remains in local, low-compute, planning/readiness mode.

Forbidden until explicitly approved:

- automatic downloads,
- GPU jobs,
- training,
- rollouts,
- heavy VLA imports,
- OpenVLA-OFT execution,
- token or secret handling in committed files.

## Immediate Next Step

Follow `reports/smolvla_manual_acquisition_checklist.md`, manually place valid SmolVLA checkpoint files, then rerun the readiness checks.
