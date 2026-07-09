# Official SmolVLA Protocol Fix Decision

Date: 2026-07-10 KST

Final decision: `LORA_CHECKPOINTS_MISSING_REGENERATION_REQUIRED`

## Boundary

This was a no-experiment protocol-fix pass.

- experiments: no
- training: no
- GPU: no
- downloads: no
- model inference: no
- simulator rollout: no
- OpenVLA-OFT: no
- FCAR: no
- LoRA seed regeneration: no
- historical metric rewrite: no

## What Is Fixed

- model revision is locked from local Hugging Face metadata;
- dataset revision is locked from local Hugging Face metadata;
- package versions are recorded;
- package source-commit gap is explicitly recorded;
- baseline naming policy is frozen;
- LoRA checkpoint persistence policy is frozen;
- official rollout action semantics are frozen;
- static-mix compute accounting requirement is frozen;
- two-stage official rollout protocol is frozen;
- official eval env readiness is classified without running rollout.

## What Blocks Rollout

Primary blocker:

- official seed-specific LoRA adapter checkpoint bundles are missing for seeds `11`, `22`, and `33`.

Additional execution blocker:

- the local official LIBERO eval stack is missing `libero` and `robosuite`; official native Windows rollout remains unproven and should move to WSL/Linux or a fixed dependency environment.

## Why This Decision Is Not `REVISION_LOCK_INCOMPLETE`

Both required source artifacts have exact local Hugging Face revisions:

- model: `31d453f7edd78c839a8bbc39744a292686daf0de`
- dataset: `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`

The package/source environment is only version-locked, not source-commit locked, but that is recorded as an environment limitation rather than a model/dataset revision-lock failure.

## Why This Decision Is Not `ROLLOUT_PROTOCOL_READY`

The protocol is defined, but rollout is not ready because the required LoRA adapter checkpoint bundles do not exist. The official eval environment is also not locally executable for LIBERO until missing dependencies are resolved.

## Exact Next Command

No training or regeneration command is safe to run under this no-experiment protocol-fix boundary.

The future required regeneration command is documented in `reports/official_smolvla_lora_checkpoint_policy.md`, but it must not be run without explicit approval for a training/regeneration pass.
