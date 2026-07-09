# Next Actions

Date: 2026-07-09 KST

Current decision:

`READY_FOR_RA_L_METHOD_ON_SMOLVLA_7D`

## Immediate Next Action

Preserve the fixed-interface baseline table, then plan any future RA-L method only with these baselines predeclared.

## Why

The standard fixed-interface 7D baseline reproduction passed the required action metric gate:

- fixed `LIBERO_7D` labels were used,
- train-split-only 7D normalization was used,
- gripper output was learned,
- no old 6D/SO100 action label path was used,
- no hard-coded gripper fill was used,
- rank-8 `state_proj` LoRA + 7D adapter beat mean-action,
- rank-8 `state_proj` LoRA + 7D adapter beat the best ridge/MLP baseline on held-out action L2.

## Current Metrics

- primary split: `same_task_demo_holdout`
- train/eval records: `300 / 100`
- mean-action action L2: `1.082453`
- ridge action L2: `0.890603`
- small MLP action L2: `0.518738`
- frozen/base SmolVLA 7D adapter action L2: `0.890604`
- no-LoRA SmolVLA 7D adapter action L2: `0.561651`
- rank-4 LoRA + 7D adapter action L2: `0.504675`
- rank-8 LoRA + 7D adapter action L2: `0.494959`
- rank-8 LoRA gripper accuracy: `0.88`
- rank-8 LoRA train/eval action-L2 gap: about `0.210549`

## Important Caveat

The previous-action persistence diagnostic reached action L2 `0.181765`, but it uses the previous expert action from the held-out HDF5 sequence. Treat it as a diagnostic persistence oracle, not as a closed-loop learned-action baseline unless an executable persistence policy is constructed.

Optional replay/progress was not run because this runner does not include a bounded executable LIBERO bridge for the learned 7D adapter.

## Allowed Next Work

- Plan a future method only after freezing this baseline table.
- Keep the rank-8 fixed-interface SmolVLA 7D LoRA/adapter as the standard learned baseline.
- Include mean-action, ridge/MLP, frozen/base 7D adapter, no-LoRA 7D adapter, and persistence diagnostics in future comparisons.
- If a replay/progress step is needed, first implement a bounded executable 7D adapter bridge and compare against expert replay and simple executable baselines.

## Disallowed Next Work

Do not:

- invent a method that lacks this baseline table,
- continue PatchGuard,
- start Target-Grounded ActionMap, SafeLoRA, PRISM, ActionMap, or another route without a fresh method-specific baseline plan,
- run OpenVLA-OFT,
- run a full benchmark,
- download large assets,
- make paper claims from this local action metric gate,
- use the old broken 6D/SO100 action path,
- use hard-coded gripper fill.
