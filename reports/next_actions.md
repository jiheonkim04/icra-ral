# Next Actions

Date: 2026-07-10 KST

Current decision: `STATIC_MERGE_ROBUST_BASELINE_READY`

## Immediate Next Action

Treat validation-selected static merge as the main realistic baseline for any later planning gate. Do not design or implement a new method in the current state.

## Required Boundary For Any Later Planning Gate

- Use official assets only.
- Use the fixed manifest: `reports/official_smolvla_split_manifest.json`.
- Use the fixed metric protocol: `reports/official_smolvla_metric_protocol.md`.
- Keep static alpha selection validation-only; never tune alpha on test.
- Compare against frozen/base, rank-4 LoRA seeds, mean-action prior, MoIRA-style task router, task oracle, frame oracle, and validation-selected static mix.
- Treat frame oracle and task oracle as upper bounds only.
- Do not revive FCAR or FCAR v2.
- Do not train any routing method unless a later explicit method-design objective allows it.
- Do not use the archived custom `LIBERO_7D` adapter route.
- Do not run OpenVLA-OFT.
- Do not run simulator rollout or full benchmark as a substitute for the fixed offline protocol.
- Do not download additional assets unless explicitly approved.

## Evidence To Preserve

- Stable artifact: `reports/official_smolvla_stable_prediction_artifact.json`.
- Seed artifacts:
  - `reports/official_smolvla_lora_seed_11_prediction_artifact.json`
  - `reports/official_smolvla_lora_seed_22_prediction_artifact.json`
  - `reports/official_smolvla_lora_seed_33_prediction_artifact.json`
- Static mix wins all reproduced seeds.
- Static mix action L2 mean/std across seeds: `0.080616431` / `0.002595356`.
- Frozen/base action L2 mean/std: `0.085558433` / `0.0`.
- Rank-4 LoRA action L2 mean/std: `0.088239344` / `0.002908670`.
- Frame oracle action L2 mean/std: `0.069117204` / `0.002049401`.
- Task oracle action L2 mean/std: `0.081138309` / `0.001955707`.
- MoIRA-style task router action L2 mean/std: `0.088145305` / `0.001719703`.
- Mean-action prior action L2: `1.197255124`.
- Realistic task win counts summed over seeds: static mix `93`, frozen/base `20`, rank-4 LoRA `7`.
- Frame oracle headroom after static remains: mean `0.011499227`.
- Task oracle is not consistently meaningful across seeds.
- MoIRA-style task router remains weak.

## Exact Next Step

If the next objective is planning, create a planning-only gate whose first hard requirement is beating validation-selected static merge under the fixed manifest. No method implementation should start until that gate is explicitly requested.
