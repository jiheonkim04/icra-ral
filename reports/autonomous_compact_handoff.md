# Autonomous Compact Handoff

Updated: 2026-07-17 KST

## Current State

- Branch: `codex/epoch5-official-prior-first`
- Latest implementation commit before this report update: `b91a49d8f66253ac85815fdde366a41824397232`
- Current epoch: 5
- Current cycle: 0
- Current stage: `epoch_5_br_xvla_closed_loop_residual_screen_complete`
- Current decision: `BR_XVLA_CLOSED_LOOP_RESIDUAL_NOT_PASSED_ABLATION_SUCCEEDED`
- Selected current method: `BR-XVLA` is now a validation no-pass; do not retune it.
- Previous method: `MCI-VLA`
- Previous decision: `MCI_STAGE_0_IMPLEMENTATION_FAILURE`
- Do not rescue or retune MCI/CSPR/R2R/CR-LightVLA/ATCD.
- Cycle 39 ordinary local-method search is superseded by the official-prior-first epoch.

## Audit Anchor

- Audit report: `reports/autonomous_research_full_history_audit.md`
- Audit accepted as evidence; the embedded generic Cycle 39 prompt is not active.
- Audit established 0 valid PROTOTYPE_GO methods, 0 official-prior Ours wins, and no second-backbone Ours result.

## Epoch 5 Report Set

- Ecosystem selection: `reports/epoch5_prior_ecosystem_selection.md`
- Reproduction plan: `reports/epoch5_prior_reproduction_plan.md`
- Reproduction result: `reports/epoch5_prior_reproduction_result.md`
- Reproduction result JSON: `reports/epoch5_prior_reproduction_result.json`
- Task-1 candidate design: `reports/epoch5_task1_ours_candidate_design.md`

## Prior Sequence

OpenVLA-OFT INT4:

- First selected official prior because code/checkpoints were accessible.
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
- Mechanism: phase-balanced LoRA/QLoRA adaptation of X-VLA-Libero, upweighting chunks where exactly one target is already in/near the basket and the remaining target still needs completion.
- LoRA/QLoRA is infrastructure, not the contribution.

## BR-XVLA Pre-Training Gates

Frozen spec:

- Artifact: `runs/xvla_prior/epoch5_br_xvla_training_spec_v1.json`
- Primary arm: `br_xvla_rank8_lambda2_lr1e4_steps64`
- Control arm: `uniform_xvla_rank8_lambda0_lr1e4_steps64`
- Max total arms: 2; max optimizer steps per arm: 64.
- At freeze time: no optimizer step, no checkpoint, no closed-loop Ours.

Data-adapter smoke:

- Artifact: `runs/xvla_prior/br_xvla_data_adapter_smoke_20260717T183355KST/result.json`
- Decision: `BR_XVLA_DATA_ADAPTER_SMOKE_PASS`
- Meaning: a tiny converted task-1 dataset satisfies X-VLA's official LIBERO reader contract.
- Caveat: a minimal local `mmengine.fileio` shim was used.

Gradient smoke:

- Artifact: `runs/xvla_prior/br_xvla_gradient_smoke_20260717T190919KST/result.json`
- SHA-256: `d661576639c86fd4657abe983968b8aa3969e934d8de082de3337cb56e7802cd`
- Decision: `BR_XVLA_GRADIENT_SMOKE_PASS`
- Model: `2toINF/X-VLA-Libero`, revision `129e71460678b7236cee6fc9707f09d9fa0c3590`
- `local_files_only`: true
- Trainable parameters: 11,868,760
- Grad tensors finite/nonzero: 537/271
- Gradient global norm: 1239.7495099257394
- CUDA peak allocated: 5,260.354 MiB
- No optimizer, checkpoint, training run, or closed-loop Ours evaluation happened in this smoke.

## BR-XVLA Bounded Training / Offline Gate

Full gate:

- Launch manifest: `runs/xvla_prior/epoch5_br_xvla_training/gate_launch_manifest.json`
- Launch manifest SHA-256: `abe8cafc194e42bfec7462f9f2825d2158dd6e9a53fd8efa8d20fdc65631eebc`
- Gate result: `runs/xvla_prior/epoch5_br_xvla_training/gate_result.json`
- Gate result SHA-256: `3af1afc6a152aae8d8fafe5dfc43a19fe9ff2174236d2f263067b1f3cace2a76`
- Decision: `BR_XVLA_OFFLINE_PASS_BEATS_ABLATION`
- Success: true
- Gate exit code: 0
- Closed-loop Ours evaluation happened: false
- Gate commit: `06d03d8147df53e54837605e079427eb4f66adfa`
- Elapsed: 150.81154718000005 seconds

Training arms:

- Primary result: `runs/xvla_prior/epoch5_br_xvla_training/br_xvla_rank8_lambda2_lr1e4_steps64/result.json`
- Primary SHA-256: `e6f8c641c4f8c931ff769bf6da11b7cfc9cdc62e90bab985896d6f9870d3ee05`
- Primary steps/checkpoint: 64 / `checkpoints/step_0064/adapter`
- Uniform result: `runs/xvla_prior/epoch5_br_xvla_training/uniform_xvla_rank8_lambda0_lr1e4_steps64/result.json`
- Uniform SHA-256: `da9492c600a84a6742f3b10e2d414c0b910da0f8434cc0b41211c76f15c1c4f0`
- Uniform steps/checkpoint: 64 / `checkpoints/step_0064/adapter`

Offline validation:

- Result: `runs/xvla_prior/epoch5_br_xvla_offline_validation_step0064.json`
- SHA-256: `119723e76e769589442fd0e04d4c26e2fe1b9fc4d825ab47ce7abd6e56ec747a`
- Fixed chunks: 24; phase counts `{0: 6, 1: 12, 2: 6}`; denoise steps 10.
- X-VLA prior mean/phase-1 loss: 3.495260993639628 / 3.107213238875071.
- BR-XVLA mean/phase-1 loss: 1.258300895492236 / 1.0268368770678837.
- Uniform mean/phase-1 loss: 1.2583009228110313 / 1.0268369267384212.
- Primary-vs-uniform phase-1 margin: `4.967053751942781e-8`.
- Interpretation: offline gate passes numerically, but the ablation margin is tiny; do not treat this as robust closed-loop superiority.

Launcher note:

- Full gate result was complete, but the launcher initially wrote a newline-only `gate_exit_code.txt`.
- The run artifact has been corrected to `0`.

## BR-XVLA Closed-Loop Residual Screen

Implementation:

- Commit: `b91a49d8f66253ac85815fdde366a41824397232`
- Evaluator: `tca_map/xvla_task1/closed_loop_residual_eval.py`
- Launcher: `tca_map/xvla_task1/launch_closed_loop_residual.py`
- Tests: `tests/test_br_xvla_closed_loop_residual_eval.py`, `tests/test_br_xvla_launch_closed_loop_residual.py`
- WSL dry-run launch passed before full run.

Frozen screen:

- Scope: `libero_10/task_1`, identity `20260727`, initial-state index `16`.
- Policies: same-run X-VLA prior, BR-XVLA primary, uniform-weight ablation.
- Frozen manifest: `runs/xvla_prior/epoch5_br_xvla_closed_loop_residual_20260727/closed_loop_manifest.json`
- Frozen manifest SHA-256: `ea222a6014e2cda6a8f7428bdf2d0f0105e1773e0f7a0c6ba3ce5bb74f01dc63`
- Result: `runs/xvla_prior/epoch5_br_xvla_closed_loop_residual_20260727/closed_loop_result.json`
- Result SHA-256: `472904b03472c8b1017aad2080c57e49c0b1064816b430670051330dd970b64f`
- Decision: `BR_XVLA_CLOSED_LOOP_RESIDUAL_NOT_PASSED`
- Python result status: `COMPLETE`; elapsed 126.246165868 seconds.
- Training/optimizer/checkpoint during screen: false/false/false.
- Closed-loop Ours evaluation happened: true, only within the frozen one-identity screen.

Policy outcomes:

- X-VLA prior: failed, 900 steps, reward 0.0.
- BR-XVLA primary: failed, 900 steps, reward 0.0.
- Uniform ablation: succeeded, 479 steps, reward 1.0.
- Interpretation: residual reproduced, but selected BR weighting failed and the key ablation solved it. This invalidates the BR-XVLA mechanism-specific claim for this frozen configuration.

Launcher caveat:

- `closed_loop_exit_code.txt` is newline-only because unescaped `$` variables were eaten in Windows-to-WSL `bash -lc`.
- The Python result artifact is complete and authoritative for policy outcomes.
- `launch_training_gate.py` and `launch_closed_loop_residual.py` are now patched to escape `\$?` and `\$status`; validation must be rerun before commit.

## Validation Status

- JSON parse: pass via `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m json.tool`.
- Handoff line count: 239, under the 250-line cap.
- Windows focused pytest: `27 passed`.
- Official WSL launcher/residual tests: `10 passed`.
- `git diff --check`: pass with LF/CRLF warnings only.
- `scripts/99_tree_check.ps1`: pass.

## Immediate Next Gate

Archive BR-XVLA as a validation no-pass and pivot under official-prior-first governance.

Allowed next:

- Validate current reports/code and launcher wrapper fix, then commit/push.
- Do a short BR-XVLA postmortem/synthesis if needed.
- Choose the next official-prior-first residual/condition or prior-anchored route without using BR-XVLA closed-loop failure for retuning.

Still disallowed:

- Retuning BR-XVLA from the failed residual screen.
- Treating the uniform ablation's single-identity success as an Ours result.
- Broad confirmatory evaluation for BR-XVLA.
- New generic method candidates.
- Retuning/rescuing MCI/CSPR/R2R/CR-LightVLA/ATCD.
- Treating X-VLA, OpenVLA-OFT INT4, or any prior diagnostic success as an Ours result.
- Treating the tiny offline primary-vs-uniform margin as evidence after closed-loop ablation domination.
- Full-model fine-tuning or making LoRA/QLoRA the headline contribution.

## Commit Scope To Finish This Turn

Stage and commit only:

- `tca_map/xvla_task1/launch_training_gate.py`
- `tca_map/xvla_task1/launch_closed_loop_residual.py`
- `tca_map/xvla_task1/closed_loop_residual_eval.py`
- `tests/test_br_xvla_closed_loop_residual_eval.py`
- `tests/test_br_xvla_launch_closed_loop_residual.py`
- `reports/epoch5_prior_reproduction_result.md`
- `reports/epoch5_prior_reproduction_result.json`
- `reports/epoch5_task1_ours_candidate_design.md`
- `reports/autonomous_compact_handoff.md`

Do not add `rollouts/2026_07_17/` or ignored run directories.
