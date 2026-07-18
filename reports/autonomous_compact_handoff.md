# Autonomous Compact Handoff

Updated: 2026-07-18 KST

## Current State

- Branch: `codex/epoch5-official-prior-first`
- HEAD before latest optimizer-gate commit: `2226dae2520c34d1cfc6876ef53e4dd0aa2f6753`
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

## R2P-XVLA Frozen Spec

Reports:

- `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_spec_result.json`
- `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_spec_result.md`

Decision: `R2P_XVLA_FROZEN_NO_TRAINING_SPEC_CREATED`

Tracked code/tests:

- `tca_map/xvla_spatial_task5/training_spec.py`
- `tests/test_xvla_spatial_task5_training_spec.py`

Ignored runtime snapshot:

- `runs/xvla_prior/epoch5_r2p_xvla_task5_training_spec_v1.json`, SHA `d795dc72373f32d36cacd4b5b6a695607154d6f65c588d56e6bd010ef4312f78`

Frozen arms:

1. `r2p_xvla_rank8_phase_weights_lr1e4_steps64` — selected method, source/transit/target weights `1.0 / 2.0 / 1.5`.
2. `uniform_task5_xvla_rank8_lambda0_lr1e4_steps64` — uniform ablation, weights `1.0 / 1.0 / 1.0`.

Still closed: training, optimizer step, checkpoint write, closed-loop Ours rollout, residual-reward checkpoint selection, privileged inference inputs, and paper claim from one identity.

Validation: `py_compile` passed; focused pytest `3 passed`; spec snapshot written without model load/training/rollout.

## R2P-XVLA Data-Adapter Smoke

Reports:

- `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_data_adapter_smoke_result.json`
- `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_data_adapter_smoke_result.md`

Decision: `R2P_XVLA_DATA_ADAPTER_SMOKE_PASS`

Tracked code/tests:

- `tca_map/xvla_spatial_task5/data_adapter_smoke.py`
- `tests/test_r2p_xvla_data_adapter_smoke.py`

Ignored runtime result:

- `runs/xvla_prior/r2p_xvla_task5_data_adapter_smoke_20260718T0417KST/result.json`, SHA `c0e44013d31f364beeea134c7991f55fb12d3643e4746961d993eeb2f19288e6`

Smoke materialized `demo_0` and `demo_40`; combined source/transit/target phase coverage `128 / 37 / 89`. Official X-VLA reader returned action `[30,20]`, proprio `[20]`, image `[3,3,224,224]`, and `domain_id` int64. No model load, training, backward, optimizer, checkpoint, simulator, or Ours rollout happened.

## R2P-XVLA Gradient Smoke

Reports:

- `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_gradient_smoke_result.json`
- `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_gradient_smoke_result.md`

Decision: `R2P_XVLA_GRADIENT_SMOKE_PASS`

Tracked code/tests:

- `tca_map/xvla_spatial_task5/gradient_smoke.py`
- `tests/test_r2p_xvla_gradient_smoke.py`

Ignored runtime result:

- `runs/xvla_prior/r2p_xvla_gradient_smoke_offline_20260718T0425KST/result.json`, SHA `c170d52cbbc01974ff51c8b3ad6e8d68136abc8cf90d9a3eb6580d72302b1f76`

WSL/offline gradient smoke loaded cached X-VLA from local snapshot, attached PEFT LoRA, ran one forward/backward, and passed with finite gradients: trainable params `11868760`, grad tensors finite/total `537/537`, nonzero `271`, gradient norm `2372.1450494696983`, weighted loss `12.958698272705078`, max CUDA allocated `5260.354` MiB. No optimizer, checkpoint, training loop, simulator, downloads, or Ours rollout happened.

## R2P-XVLA Optimizer Gate

Reports:

- `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_optimizer_gate_result.json`
- `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_optimizer_gate_result.md`

Decision: `R2P_XVLA_OPTIMIZER_GATE_FROZEN_TRAINING_NOT_LAUNCHED`

The optimizer-step contract is frozen but not armed. Exact arms and output dirs are fixed under `runs/xvla_prior/epoch5_r2p_xvla_task5_training/`; required pre-step writes are `worker.pid`, `training_status.json`, `heartbeat.json`, and `frozen_spec_snapshot.json`; offline flags must be enforced; no third task5 config, downloads, residual-reward checkpoint selection, privileged inference input, or closed-loop rollout is allowed.

Reason not armed at this point: task5 `train_lora` and offline-validation runners were not implemented/tested yet. No training, optimizer, checkpoint, or rollout happened.

## R2P-XVLA Train LoRA Runner

Reports:

- `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_train_lora_runner_result.json`
- `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_train_lora_runner_result.md`

Decision: `R2P_XVLA_TRAIN_LORA_RUNNER_IMPLEMENTED_TESTED_NOT_LAUNCHED`

Tracked code/tests:

- `tca_map/xvla_spatial_task5/train_lora.py`
- `tests/test_r2p_xvla_train_lora.py`

The runner now loads the frozen spec, accepts only the two frozen arms, rejects downloads, enforces the exact output root, rejects max steps above 64, writes worker/status/heartbeat/spec/log/exit/result artifacts when launched, materializes official X-VLA reader clips without residual-reset sampling, uses frozen phase-weighted loss, and limits checkpoints to steps `16/32/64`. It does not launch training by itself and it performs no closed-loop rollout or residual-reward checkpoint selection.

Validation: WSL `.venv` `py_compile` passed; focused pytest `6 passed`; task5 bundle pytest `14 passed` with the existing SciPy/NumPy warning. No model load, optimizer, checkpoint, simulator, download, or Ours rollout happened in this implementation gate.

## R2P-XVLA Offline Validation Runner

Reports:

- `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_offline_validation_runner_result.json`
- `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_offline_validation_runner_result.md`

Decision: `R2P_XVLA_OFFLINE_VALIDATION_RUNNER_IMPLEMENTED_TESTED_NOT_LAUNCHED`

Tracked code/tests: `tca_map/xvla_spatial_task5/offline_validate.py`, `tests/test_r2p_xvla_offline_validate.py`.

The runner fixes validation to demos `40..49`, rejects downloads and output-path drift, expects frozen step-64 primary/uniform adapters, uses a common R2P phase-weighted metric for Primary vs Uniform, checks source degradation/action delta/CUDA bounds, and writes worker/status/heartbeat/log/exit/result artifacts when launched. It performs no closed-loop rollout, residual-reward checkpoint selection, or privileged inference-state use.

Validation: WSL `.venv` `py_compile` passed; focused pytest `6 passed`; task5 bundle pytest `20 passed` with the existing SciPy/NumPy warning. No model/adapters were loaded and no offline validation runtime, optimizer, checkpoint, simulator, download, or Ours rollout happened.

Sequential gate runner: `tca_map/xvla_spatial_task5/training_gate.py` with `tests/test_r2p_xvla_training_gate.py`; report `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_training_gate_runner_result.json`; decision `R2P_XVLA_SEQUENTIAL_TRAINING_GATE_IMPLEMENTED_TESTED_NOT_LAUNCHED`; expanded task5 bundle pytest `23 passed`. No training/runtime happened.

Arming report: `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_gate_arming_result.json`; decision `R2P_XVLA_OPTIMIZER_GATE_ARMED_TRAINING_LAUNCH_AUTHORIZED`; prelaunch clean except pre-existing untracked rollout dirs.

Attempt 1 failed before training due missing `peft` in repo `.venv`; archived at `runs/xvla_prior/epoch5_r2p_xvla_task5_training_failed_peft_missing_20260718T0456KST`; report `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_training_launch_attempt1_result.json`. Re-armed with `/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python`.

Attempt 2 failed before training due Windows X-VLA root in WSL; archived at `runs/xvla_prior/epoch5_r2p_xvla_task5_training_failed_xvla_root_20260718T0459KST`; fixed defaults to `/mnt/c/assets/repos/X-VLA`, report `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_wsl_root_fix_rearming_result.json`; task5 pytest `25 passed`.

Attempt 3 completed training/offline selection: both arms reached `64/64`, but frozen offline selection failed (`primary 0.9418842308` vs uniform `0.9418841700` weighted loss; delta `6.08e-08`); report `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_training_gate_result.json`. No closed-loop rollout happened.

## Current Immediate Next Action

R2P archive: `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_archive_decision.json`; latest confirmed shared residual remains `libero_goal/task_3`, identity `20260728` (Base/X-VLA/OpenVLA-INT4 clean failures; task-level headroom positive).

Repeated-residual screen `20260729..33`: goal/task3 `20260729` Base-solved; object/spatial `20260728` saturated; goal `20260729` no shared residual; object `20260729` saturated; spatial/task4 `20260729` second-prior-solved; goal `20260730` no shared residual; object `20260730` saturated; spatial/task5 shared failures (`20260727`,`20260730`,`20260733`) with headroom; `20260731/32` task5 X-VLA-solved. Candidate cap exhausted: SGL simple-control-equivalent; OCR needs observability data, not method kill. Post-task5 scans: `20260731` goal/object saturated and spatial task4 second-prior-solved; `20260732` all saturated; `20260733` goal/object saturated and spatial repeats closed task5; `20260734` goal/object saturated and spatial task5 closed-family failure; `20260735` goal/object saturated. Cap: only final spatial `20260735` scan remains, then return convergence decision; no training/Ours.
