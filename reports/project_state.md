# Project State

Date: 2026-07-10 KST

Target branch: `main`

Implementation branch: `codex/audit-official-smolvla-execution-ledger`

Current decision: `AUDIT_FOUND_PROTOCOL_GAPS_FIX_BEFORE_ROLLOUT`

## Current Route

The archived custom SmolVLA 7D adapter route remains stopped. The valid route is official SmolVLA/LeRobot reproduction first, using official preprocessing, normalization, action conventions, dataset format, and evaluation stack.

This state update is a repository audit only. It did not run experiments, train, use GPU, download assets, run simulator rollout, run a full benchmark, run OpenVLA-OFT, design a new method, revive FCAR, rerun LoRA seeds, modify past results, delete artifacts, or overwrite artifacts.

## Audit Scope

- audited commit range: `72ed23e` through `5d48b1e`
- audited ledger entries: `13`
- runner-backed historical executions in the audited range: `8`
- current final offline result audited: `STATIC_MERGE_ROBUST_BASELINE_READY`
- audit decision: `AUDIT_FOUND_PROTOCOL_GAPS_FIX_BEFORE_ROLLOUT`

Audit reports:

- `reports/official_smolvla_execution_ledger.md`
- `reports/official_smolvla_execution_ledger.json`
- `reports/official_smolvla_duplicate_run_audit.md`
- `reports/official_smolvla_skipped_stage_audit.md`
- `reports/official_smolvla_artifact_integrity_audit.md`
- `reports/official_smolvla_baseline_naming_audit.md`
- `reports/official_smolvla_protocol_compliance_audit.md`
- `reports/official_smolvla_audit_decision.md`

## Audit Findings

- exact duplicate runs found: `0`
- possible exact duplicate runs found: `0`
- avoidable regenerations found: `2`
- artifact inconsistencies found: `0`
- test leakage found: `0`
- old custom `LIBERO_7D` route in final official runs: `False`
- current offline results remain valid: `True`

Avoidable regenerations:

1. Routing design gate regenerated rank-4 LoRA predictions for oracle analysis because earlier per-frame artifacts were not reusable.
2. FCAR tiny gate regenerated a fixed rank-4 LoRA prediction artifact for the same reason.

These do not invalidate the final stable offline result.

## Fixed Protocol Evidence

- split manifest: `reports/official_smolvla_split_manifest.json`
- metric protocol: `reports/official_smolvla_metric_protocol.md`
- stable base artifact: `reports/official_smolvla_stable_prediction_artifact.json`
- train: `80` episodes / `1200` frames
- validation: `40` episodes / `400` frames
- test: `80` episodes / `1200` frames
- tasks: `40`
- episode intersections: train/validation `0`, train/test `0`, validation/test `0`
- manifest SHA256: `1279F939648CF13E2F599084E42631681E1DFA5606B5D9B0851FFEB32710934B`

## Current Offline Baseline Status

Technically correct future name:

`validation_selected_action_space_static_mix`

Current offline evidence:

- seeds audited: `11`, `22`, `33`
- seed win count: `3` / `3`
- action L2 mean/std:
  - frozen/base: `0.085558433` / `0.000000000`
  - rank-4 LoRA: `0.088239344` / `0.002908670`
  - validation-selected action-space static mix: `0.080616431` / `0.002595356`
  - frame oracle upper bound: `0.069117204` / `0.002049401`
- realistic task win counts summed over seeds: static mix `93`, frozen/base `20`, rank-4 LoRA `7`
- frame oracle headroom after static mix: mean `0.011499227`

This remains offline evidence only. It is not official closed-loop LIBERO success-rate evidence.

## Naming Status

Future reports must use:

- `task_or_instruction_router_proxy` for the current MoIRA-style local proxy
- `validation_selected_action_space_static_mix` for current static base/LoRA action interpolation
- `frame_oracle_upper_bound` and `task_oracle_upper_bound` for oracle results

Future reports must not call the proxy an official MoIRA reproduction. Future reports must not call action-space interpolation adapter soup or adapter-weight merge.

## Protocol Gaps Before Rollout

- Hugging Face model/dataset revisions are not pinned.
- Seed-specific LoRA adapter checkpoint persistence policy is not settled.
- Future baseline naming must be corrected before paper-facing summaries.
- Official closed-loop LIBERO rollout is not done.
- Official task success-rate evaluation is not done.
- Full benchmark is not done.
- Official MoIRA reproduction is not done.
- True adapter-weight soup/merge is not done.

## Conclusion

`AUDIT_FOUND_PROTOCOL_GAPS_FIX_BEFORE_ROLLOUT`

The current offline static-mix result remains valid, but the next step is a no-experiment protocol-fix branch before official rollout.
