# Autonomous Compact Handoff

Updated: 2026-07-18 KST

## Current State

- Branch: `codex/epoch5-official-prior-first`
- HEAD before latest candidate-generation commit: `7528ce99fb1a3f483c0974cf852d92b1f6a8666a`
- Epoch/cycle: `5 / 0`
- Paper status: no `PROTOTYPE_GO`, no `PAPER_READY`, no `READY_TO_DRAFT_RAL_PAPER_PACKAGE`.
- Active Ours training/worker: none.
- Preserve ignored `rollouts/` and `runs/` artifacts.

## Standing Rules

- Official-prior-first remains active.
- Do not rescue or retune MCI/CSPR/R2R/CR-LightVLA/ATCD/BR-XVLA/MPR-XVLA/PRC-XVLA.
- LoRA/QLoRA is implementation infrastructure only, never the contribution.
- Do not treat prior failure, prior success, headroom, candidate generation, or uniform-ablation success as Ours.
- Candidate generation does not authorize training, optimizer steps, checkpoint writes, or closed-loop Ours rollout.
- Keep this file under 250 lines.

## Comparator Calibration

Durable addendum: `reports/comparator_role_calibration.md`

- Future unfrozen results use role-specific Base/Prior/Ablation/Control interpretation.
- Frozen historical non-GO decisions are not rewritten post hoc.
- If a future protocol explicitly freezes a universal beat-all scalar rule, report both `FROZEN_PROTOCOL_DECISION` and `CALIBRATED_SCIENTIFIC_INTERPRETATION`.

## Audit Baseline

Full audit: `reports/autonomous_research_full_history_audit.md`

- Ledger routes: 95; selected formal Ours methods: 50.
- No paper-ready method.
- Strongest historical Ours remains CAVM 24/58, but no third expansion is allowed.
- BR-XVLA and MPR-XVLA are no-pass and must not be reopened.

## Closed / Non-Target Evidence

- Task75 (`libero_90/task_75`, identity `20260725`) has preserved evidence in `reports/task75_local_evidence_manifest.json`; X-VLA and SmolVLA Base failed, task-level headroom was positive, but valid second prior was infrastructure-blocked because local OpenVLA-OFT/LightVLA stats lacked `libero_90`.
- Task75 second-prior decision: `TASK75_SECOND_PRIOR_INFRASTRUCTURE_BLOCKED`; report `reports/task75_second_prior_result.json`.
- X-VLA scans saturated `libero_spatial`, `libero_goal`, and `libero_object` identity `20260725`.
- X-VLA scans saturated `libero_goal`, `libero_object`, and `libero_spatial` identity `20260726`.
- `libero_goal/task_9` identity `20260727` was second-prior-solved by Quantized OpenVLA-OFT INT4 and is closed.
- `libero_object` identity `20260727` was saturated by X-VLA and creates no Ours target.

## Current Target Chain

Target: `libero_spatial/task_5`, reset identity `20260727`, initial-state index `16`.

Instruction: `pick up the black bowl on the ramekin and place it on the plate`

Initial-state SHA-256: `7230223d3b36c289be0dc4cfbfe916bfe65e2b20c4755b123504b97f9db19e76`

### First Prior

Report: `reports/post_secondprior_libero_spatial_20260727_prior_scan_result.json`

- X-VLA official prior completed 10/10 tasks; 9/10 succeeded.
- Task `5` failed cleanly; 0 infrastructure failures.
- Summary SHA `768171a6406a3e15d8c47f3a36a3b20f992721316e234f0cb6d8c5525a242e91`; task-5 result SHA `9a6da411db84298748e5a35d23aa5784339f6bc14cdbe24f6842e6a5e6ce40be`.
- No training/Ours/checkpoint/optimizer step.

### Base

Report: `reports/post_secondprior_libero_spatial_20260727_base_gate_result.json`

- SmolVLA Base failed cleanly: 0/1 success, 0 infrastructure failures, 280 env steps, 6 action chunks.
- Result SHA `353e3d66bd98696f2a5d64e86f3eb72295b61b18091aba56fdda09da0b3e0941`; video SHA `b06bed4febfc09e6891e56e677297683e74f25992d29ac1dfe1282d47aa2ff59`.
- No training/Ours/checkpoint/optimizer step.

### Headroom

Report: `reports/post_secondprior_libero_spatial_20260727_headroom_result.json`

- Decision: `TASK5_TASK_LEVEL_EXPERT_HEADROOM_POSITIVE_SAME_RESET_UNAVAILABLE`.
- Nearest HDF5 demo `demo_9`, L2 `2.984242906`, init SHA `0d599c208cb9d95b4e724e2a883c651a720276cd8e15e754cf6f3a7527ae497f`.
- Exact selected-demo replay succeeded at index `93`; zero-action control failed.
- No same-reset HDF5 init-state hash matched the residual.
- Result SHA `42c0b9e287904a7781cf077397c64578a3a5fb7ab651f30f85f810f18eb44fb9`.

### Second Prior

Report: `reports/post_secondprior_libero_spatial_20260727_second_prior_result.json`

- Quantized OpenVLA-OFT INT4 had valid `libero_spatial_no_noops` support and no proxy.
- It failed cleanly: 0/1 success, 0 infrastructure failures, final reward `0.0`, 230 steps, 28 chunks.
- Result SHA `ac550a1cf3c779495f645c6a9f9cf10d336d99723ddefdc872b803e19a69b0f1`; video SHA `83c8db433af3c9dfeeb030b4dbd062980c9ba8e347221019576efc91ebdbd2fb`.
- No training/Ours/checkpoint/optimizer step.

### Data Audit

Report: `reports/post_secondprior_libero_spatial_20260727_data_audit_result.json`

- PASS; candidate-generation readiness true.
- 50 demos; 40/10 train/validation; train/validation chunks `4325 / 1121`.
- Terminal reward/done demos `50 / 50`; actions are finite 7D, max abs `1.0`.
- No residual init-state overlap.
- Train source/transit/target chunks `2627 / 650 / 1048`; validation `711 / 164 / 246`.
- Result SHA `e782d0947edaee4c8eef26d36af3627bd30d787ffe64df22e3678cff9b3abda5`.
- Focused pytest: `2 passed`.

## Candidate Generation

Reports:

- `reports/post_secondprior_libero_spatial_20260727_candidate_generation_result.json`
- `reports/post_secondprior_libero_spatial_20260727_candidate_generation_result.md`

Decision: `EXACTLY_TWO_CANDIDATES_GENERATED_ONE_SELECTED`

Exactly two candidates were generated:

1. `R2P-XVLA` / Ramekin-to-Plate Phase-Balanced X-VLA Adapter — selected for a frozen no-training spec.
2. `CTR-XVLA` / Clearance-Triggered Temporal Requery X-VLA — not selected; may later serve as a simple-control threat.

No training, optimizer step, checkpoint write, implementation, LoRA/QLoRA training, or closed-loop Ours rollout happened.

## Immediate Next Action

Create a frozen no-training specification for selected candidate `R2P-XVLA`.

Required contents for the spec:

- exact training-data split and phase-label derivation from the data audit;
- deployment input policy: RGB, wrist RGB, proprioception, instruction only;
- no privileged object positions or phase labels at inference;
- uniform LoRA/OFT simple-control role;
- no-phase-balancing key ablation;
- clean-retention and validation gates before any closed-loop Ours rollout;
- explicit prohibition on tuning checkpoint selection from residual rollout reward.

Do not train or run Ours yet.

## Validation To Run Before Commit

- Parse new candidate JSON and existing target-chain JSON.
- Check `reports/autonomous_compact_handoff.md` line count is under 250.
- `git diff --check`
- `powershell -ExecutionPolicy Bypass -File .\scripts\99_tree_check.ps1`
