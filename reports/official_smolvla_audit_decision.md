# Official SmolVLA Audit Decision

Date: 2026-07-10 KST

Final audit decision:

`AUDIT_FOUND_PROTOCOL_GAPS_FIX_BEFORE_ROLLOUT`

## Reason

The audit found no exact duplicate run, no test leakage, no seed artifact overwrite, no report/JSON mismatch that invalidates the final offline result, and no accidental use of the old custom `LIBERO_7D` route in the final official protocol runs.

However, protocol gaps remain before official rollout or paper-facing claims:

- Hugging Face model/dataset revisions are not pinned.
- Future reports must rename the local MoIRA-style proxy as `task_or_instruction_router_proxy`.
- Future reports must rename static action interpolation as `validation_selected_action_space_static_mix`.
- Seed-specific LoRA adapter checkpoints are not persisted; prediction artifacts are persisted.
- Some avoidable regeneration occurred because early per-frame prediction artifacts were not reusable.

## Current Offline Result Validity

Current offline results remain valid within their stated scope.

The strongest realistic offline baseline is:

`validation_selected_action_space_static_mix`

Evidence:

- seeds: `11`, `22`, `33`
- seed win count: `3` / `3`
- static mix action L2 mean/std: `0.080616431` / `0.002595356`
- frozen/base action L2 mean/std: `0.085558433` / `0.000000000`
- rank-4 LoRA action L2 mean/std: `0.088239344` / `0.002908670`
- frame oracle upper-bound action L2 mean/std: `0.069117204` / `0.002049401`

## Exact Next Step

Before official rollout, create a no-experiment protocol-fix branch that records Hugging Face model/dataset revision pins, enforces the future baseline naming glossary, and decides whether LoRA adapter checkpoints must be persisted alongside prediction artifacts.
