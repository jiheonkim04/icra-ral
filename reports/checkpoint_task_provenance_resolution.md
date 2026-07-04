# Checkpoint / Task Provenance Resolution

This is a report-only audit after the normalized action-space probe plan.

Purpose:

- decide whether the current local `lerobot/smolvla_base` checkpoint should continue to be used for learned-policy LIBERO rollout evidence,
- separate checkpoint/task provenance problems from TCA-Map method evidence,
- prevent ad hoc action postprocessor changes from hiding a checkpoint/action convention mismatch.

Current evidence to inspect:

- checkpoint config action/state/image shapes,
- checkpoint policy preprocessor and postprocessor metadata,
- local checkpoint model card README,
- LIBERO action-stat subset audit,
- normalized action-space probe plan output.

Expected interpretation:

- if checkpoint metadata remains 6D/SO100-like while local LIBERO demonstrations remain 7D/unit-scale, learned-policy LIBERO rollout scaling with this checkpoint stays no-go,
- the next safe direction is either an offline/head TCA-Map and required LoRA evidence path, or a separate source-resolution plan for a LIBERO-action-aligned SmolVLA checkpoint,
- no normalized-action runner, postprocessor bypass, learned-policy rollout, benchmark claim, or paper claim is authorized by this audit.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\122_resolve_checkpoint_task_provenance.ps1
```
