# RA-L Strategy Reset

Date: 2026-07-09 KST

## Current Strategic Decision

`OFFICIAL_VLA_RECIPE_REPRODUCTION_REQUIRED`

## Reset Statement

The custom SmolVLA 7D adapter route is stopped as a main RA-L path. The project should not start a new VLA method from this local adapter stack.

## Routes Killed Or Stopped

- PatchGuard-VLA: killed by baseline dominance.
- Target-Grounded ActionMap / TG-7D: do not revive under this objective.
- SafeLoRA / PRISM: do not revive under this objective.
- Custom SmolVLA 7D adapter route: stopped after feature, replay, and range repairs failed to produce control transfer.

## What Remains Valid

- Use existing reports as negative evidence and infrastructure documentation.
- Reuse PEFT/bitsandbytes/CUDA/SmolVLA smoke evidence.
- Reuse LIBERO 7D interface and feature-schema diagnostics for official baseline checks.
- Reuse exact-init expert replay only as a local diagnostic, not as paper evidence.

## Next Strategy

Reproduce an official VLA baseline recipe first. If that is not feasible under local constraints, stop VLA method search under the current setup rather than inventing another local proxy method.
