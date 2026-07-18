# SGL-XVLA Stage 0 Gate Freeze

Decision: `SGL_XVLA_STAGE0_GATE_FROZEN_TESTED_NO_TRAINING_NO_OURS`

I froze a no-training Stage 0 gate for the primary task5 candidate, `SGL-XVLA` / Support-Gated Lift for X-VLA.

What is frozen:

- Development residual identities: `20260727`, `20260730`, `20260733`
- Clean-retention identities: `20260731`, `20260732`
- Held-out identity pool: `20260734`, `20260735`, `20260736`, `20260737`
- Forbidden inference inputs: simulator state/contact, reward, success flag, HDF5 identity, reset identity, phase label, and task-success oracle
- Required Stage 0 checks: support observability, action-bias bounds, fixed-lift simple control, and held-out identity manifest

Validation:

- `py_compile` passed for `tca_map/xvla_spatial_task5/sgl_stage0_gate.py`
- Focused pytest passed: `5 passed`
- The frozen JSON artifact was emitted to `runs/xvla_prior/epoch5_sgl_xvla_task5_stage0_gate_v1.json`

No training, optimizer step, checkpoint write, model load, simulator episode, closed-loop Ours evaluation, or LoRA/QLoRA update happened.

Next action: create a Stage 0 observability/report-only check for SGL support-condition observability using existing artifacts only.
