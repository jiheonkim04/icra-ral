# Autonomous Compact Handoff

Updated: 2026-07-17 KST

## Current State

- Branch: `codex/epoch5-official-prior-first`
- Pre-update HEAD: `330230f96f208ba1aeb4879904552d65bd42fd21`
- Current epoch: 5
- Current cycle: 0
- Current stage: `epoch_5_br_xvla_gradient_smoke_complete`
- Current decision: `BR_XVLA_GRADIENT_SMOKE_PASS_TRAINING_LAUNCHER_PENDING`
- Selected current method: `BR-XVLA`
- Previous method: `MCI-VLA`
- Previous decision: `MCI_STAGE_0_IMPLEMENTATION_FAILURE`
- Do not rescue or retune MCI/CSPR/R2R/CR-LightVLA/ATCD.
- Cycle 39 ordinary local-method search is superseded by the official-prior-first epoch.

## Audit Anchor

- Audit report: `reports/autonomous_research_full_history_audit.md`
- Audit accepted as evidence; the embedded generic Cycle 39 resume prompt is not the active plan.
- Audit refresh commit on this branch: `330230f96f208ba1aeb4879904552d65bd42fd21`
- Audit established 0 valid PROTOTYPE_GO methods, 0 official-prior Ours wins, and no second-backbone Ours result.

## Epoch 5 Report Set

- Ecosystem selection: `reports/epoch5_prior_ecosystem_selection.md`
- Reproduction plan: `reports/epoch5_prior_reproduction_plan.md`
- Reproduction result: `reports/epoch5_prior_reproduction_result.md`
- Reproduction result JSON: `reports/epoch5_prior_reproduction_result.json`
- Task-1 candidate design: `reports/epoch5_task1_ours_candidate_design.md`

## Prior Sequence So Far

OpenVLA-OFT INT4:

- Selected first because official code/checkpoints were accessible.
- Hard-slice diagnostic: OpenVLA-OFT INT4 20/20, SmolVLA base 11/20.
- Residual diagnostic: OpenVLA-OFT INT4 14/16 vs SmolVLA base 7/16.
- R2R-OFT was tried under a bounded spec, but offline validation did not pass; no closed-loop Ours GO.

LightVLA:

- Official LightVLA-Libero10 checkpoint loaded and ran locally.
- It was complementary on task 8 but did not produce a usable Ours route.
- CR-LightVLA Stage 0 and ATCD teacher-signal audit were closed; do not rescue.

X-VLA:

- X-VLA-Libero official prior loaded from cached HF assets.
- Task 8 residual was solved by the prior, so no Ours target there.
- Task 1 exposed the current shared residual used for `BR-XVLA`.

## Current Residual: LIBERO-10 Task 1

Task: put both the cream cheese box and the butter in the basket.

Matched diagnostic over identities `20260724..20260731`:

- X-VLA-Libero: 6/8, failures `20260725`, `20260727`.
- SmolVLA frozen base: 3/8, failures `20260724`, `20260727`, `20260728`, `20260729`, `20260730`.
- Shared valid target: `20260727`.
- `20260725` is not an Ours target because base solved it while X-VLA failed.

Headroom:

- Artifact: `runs/xvla_prior/diagnostic_task1_expert_headroom_20260727_20260717T180914KST/result.json`
- Positive task-level HDF5 expert replay exists.
- Caveat: nearest HDF5 demo init-state was selected by L2; local HDF5 init-state hashes do not match the frozen benchmark reset hash.

## Basket Data and Method Selection

Task-1 basket data audit:

- Artifact: `runs/xvla_prior/diagnostic_task1_basket_data_audit_20260727_20260717T181823KST/result.json`
- 50 demos, 13,021 steps, 12,671 total 8-step chunks.
- Train one-target-remaining chunks: 4,607 across 40 demos.
- Validation one-target-remaining chunks: 1,079 across 10 demos.
- Labels use HDF5 simulator state for training/validation only, not inference.

Candidate design:

- Exactly two candidates were generated around the task-1 residual.
- Selected: `BR-XVLA`, score 86/100.
- Not selected: `OCB-XVLA`, score 73/100.
- Mechanism: LoRA/QLoRA-adapt X-VLA-Libero with phase-balanced imitation, upweighting chunks where exactly one target is already in/near the basket and the remaining target still needs completion.
- LoRA/QLoRA is infrastructure, not the contribution.

Frozen BR-XVLA spec:

- Artifact: `runs/xvla_prior/epoch5_br_xvla_training_spec_v1.json`
- Primary arm: `br_xvla_rank8_alpha16_phase3_lr2e4_steps64`
- Control arm: `uniform_xvla_rank8_alpha16_phase1_lr2e4_steps64`
- Max total arms: 2; max optimizer steps per arm: 64.
- At freeze time: no optimizer step, no checkpoint, no closed-loop Ours.

## BR-XVLA Data-Adapter Smoke

- Module: `tca_map/xvla_task1/data_adapter_smoke.py`
- Test: `tests/test_br_xvla_data_adapter_smoke.py`
- Artifact: `runs/xvla_prior/br_xvla_data_adapter_smoke_20260717T183355KST/result.json`
- Decision: `BR_XVLA_DATA_ADAPTER_SMOKE_PASS`
- Meaning: a tiny converted task-1 dataset satisfies X-VLA's official LIBERO reader contract.
- Caveat: a minimal local `mmengine.fileio` shim was used for the smoke only.

## BR-XVLA One-Batch Gradient Smoke

- Module: `tca_map/xvla_task1/gradient_smoke.py`
- Test: `tests/test_br_xvla_gradient_smoke.py`
- Artifact: `runs/xvla_prior/br_xvla_gradient_smoke_20260717T190919KST/result.json`
- Reported artifact SHA-256: `d661576639c86fd4657abe983968b8aa3969e934d8de082de3337cb56e7802cd`
- Decision: `BR_XVLA_GRADIENT_SMOKE_PASS`
- Model: `2toINF/X-VLA-Libero`, revision `129e71460678b7236cee6fc9707f09d9fa0c3590`
- `local_files_only`: true
- Clip: `demo_0`, start 115, end 211, 96 steps, one-target fraction 1.0
- Loss total: 2.6243011951446533
- Weighted loss: 7.872903823852539
- Phase weight: 3.0
- Trainable parameters: 11,868,760
- Grad tensors finite/nonzero: 537/271
- Gradient global norm: 1239.7495099257394
- CUDA peak allocated: 5,260.354 MiB
- Policy booleans: model loaded true, PEFT LoRA attached true, forward true, backward true.
- Policy booleans: optimizer created false, optimizer step false, checkpoint false, training false, closed-loop Ours false.

Runtime repairs for the smoke:

- Installed X-VLA requirements dependency `timm==1.0.12` in `official-smolvla-libero`.
- Added import-only optional server shims for `fastapi`, `uvicorn`, and `json_numpy`.
- Added compatibility patches for X-VLA under Transformers 4.57.6:
  - `Florence2ForConditionalGeneration._supports_sdpa = False`
  - safe `get_output_embeddings` handling for missing `lm_head`

## Validation Status

- `py_compile` for `gradient_smoke.py` and `test_br_xvla_gradient_smoke.py`: pass.
- Focused pytest suite: `9 passed`.
- JSON/report validation must be rerun after the handoff line-count replacement.
- Keep this file under 250 lines.

## Immediate Next Gate

Prepare the bounded two-arm BR-XVLA training launcher and offline-validation path.

Allowed next:

- Implement launcher/offline validation under the frozen spec.
- Optimizer steps only inside that bounded training gate.
- Preserve artifacts, logs, heartbeats, exit codes, and resume commands for long jobs.

Still disallowed:

- Closed-loop Ours evaluation before offline validation passes.
- New generic method candidates.
- Retuning/rescuing MCI/CSPR/R2R/CR-LightVLA/ATCD.
- Treating X-VLA, OpenVLA-OFT INT4, or any prior diagnostic success as an Ours result.
- Full-model fine-tuning or making LoRA/QLoRA the headline contribution.

## Commit Scope To Finish This Turn

Stage and commit only:

- `tca_map/xvla_task1/gradient_smoke.py`
- `tests/test_br_xvla_gradient_smoke.py`
- `reports/epoch5_prior_reproduction_result.md`
- `reports/epoch5_prior_reproduction_result.json`
- `reports/epoch5_task1_ours_candidate_design.md`
- `reports/autonomous_compact_handoff.md`

Do not add `rollouts/2026_07_17/` or ignored run directories.
