# Epoch 5 Prior Reproduction Plan

Selected ecosystem: OpenVLA-OFT on LIBERO.

## Boundary

This plan reproduces or validates the selected external prior before any Ours design. It does not create a new method candidate, does not rescue MCI-VLA, does not run training, and does not claim a full-precision paper reproduction from an INT4 run.

## Preferred Evidence Order

1. Official execution: run official OpenVLA-OFT code and checkpoint under its intended environment.
2. Mechanism-faithful local port: allowed only if official code is inspected and the computational graph, inputs, action semantics, and inference mechanism are preserved.
3. Existing validated local execution: acceptable as recovered evidence for the branch transition only if the artifact was produced by the official stack, is result-file backed, and passes focused validation now.

## Local Artifact State

| Item | Status |
|---|---|
| Official code checkout | present at `C:\assets\repos\openvla-oft`; checkout is dirty from prior local compatibility changes and must not be cleaned destructively |
| Official checkpoint | present in WSL at `/home/jiheon/assets/checkpoints/openvla-oft/moojink_openvla-7b-oft-finetuned-libero-spatial-object-goal-10` |
| Checkpoint visible size | about 15G local disk; state records 14.845 GiB visible size |
| Prior local execution | `reports/openvla_oft_quantized_hard_slice_result.json` and `runs/openvla_oft_int4/hard_slice_openvla_int4.json` |
| Matched Base artifact | `runs/openvla_oft_int4/hard_slice_smolvla_exact.json` |
| Validation test | `tests/test_openvla_oft_int4_gate.py` |

## Risk Assessment

| Risk | Decision |
|---|---|
| Download | no new download in this step |
| GPU | no new rollout or model load in this step; focused test only reads artifacts |
| Simulator | no new simulator rollout in this step |
| Full BF16 OpenVLA-OFT | not attempted; prior local preflight forbids full BF16 on this 16GB GPU |
| Quantization | INT4 result is valid as a quantized local prior diagnostic, not as numerical full-precision reproduction |
| External checkout dirtiness | record it; do not reset or clean external repo |

## Reproduction Validation Command

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests\test_openvla_oft_int4_gate.py -q
```

Expected result: all focused OpenVLA-OFT gate tests pass.

## Matched Base/Prior Diagnostic Already Available

Existing matched hard-slice diagnostic:

- OpenVLA-OFT INT4: 20/20 successful episodes.
- SmolVLA frozen-base exact-init: 11/20 successful episodes.
- Hard-slice failures in SmolVLA were not reproduced by OpenVLA-OFT INT4.

This satisfies prior-positive reproduction for the selected hard-slice condition but does not yet satisfy residual-gap discovery because OpenVLA-OFT saturated that condition.

## Residual-Gap Next Step

Do not design Ours yet. The next scientific step is to preregister and run a small residual-condition diagnostic for OpenVLA-OFT. Candidate residual conditions must come from the selected prior's known limits or benchmark stressors, for example:

- LIBERO-PRO-style object/instruction/environment perturbations if assets are locally accessible without risky downloads;
- official LIBERO tasks where OpenVLA-OFT is not saturated under a matched small manifest;
- language-grounding or visual-feedback cases motivated by the OpenVLA-OFT paper's own qualitative discussion.

If no residual remains, move to pi0.5/OpenPI as the second-ranked ecosystem rather than inventing a proxy-only local method.

## Preregistered Residual Diagnostic: `epoch5_libero10_residual_v1`

Execution status: `COMPLETE`

The first residual diagnostic is a bounded LIBERO-10 long-horizon expansion
using official LIBERO tasks that were not in the saturated hard-slice result.

Rationale:

- OpenVLA-OFT reports very strong but non-perfect LIBERO performance, so a
  residual, if locally accessible, is most likely to appear on long-horizon
  LIBERO-10 tasks rather than already-saturated spatial/control tasks.
- Task IDs `8` and `9` are official LIBERO-10 tasks, share the selected
  OpenVLA-OFT checkpoint's action/observation semantics, and were not evaluated
  in the recovered hard-slice condition.
- Reset labels `20260716..20260723` map to official initial-state indices
  `5..12`, disjoint from the recovered hard-slice indices `0..4`.

Frozen manifest:

| Field | Value |
|---|---|
| Label | `epoch5_libero10_residual_v1` |
| Tasks | `libero_10/task_8` "put both moka pots on the stove"; `libero_10/task_9` "put the yellow and white mug in the microwave and close it" |
| Reset identities | `20260716,20260717,20260718,20260719,20260720,20260721,20260722,20260723` |
| Episodes per policy | `16` |
| Policies | SmolVLA frozen-base exact-init; Quantized OpenVLA-OFT INT4 |
| Matched row SHA-256 | `13642c7bed5e7d5944f7377e9848aeec1b9090be96d110362b53bc9cd9a3b3b2` |
| OpenVLA manifest | `runs/openvla_oft_int4/epoch5_libero10_residual_openvla_manifest.json` |
| SmolVLA manifest | `runs/openvla_oft_int4/epoch5_libero10_residual_smolvla_manifest.json` |

Execution boundaries:

- no new download;
- no training or fine-tuning;
- no full-BF16 OpenVLA-OFT load;
- no Ours method, method candidate, or local proxy;
- stop after two identical infrastructure failures, matching the existing
  runner safety behavior.

Decision rules:

- `RESIDUAL_FOUND_PRIOR_POSITIVE`: Base has meaningful failure, OpenVLA-OFT
  improves over Base, and OpenVLA-OFT leaves at least one measured failure.
- `PRIOR_SATURATED_NEXT_CONDITION`: OpenVLA-OFT succeeds on all 16 episodes;
  do not design Ours for this condition.
- `PRIOR_NOT_POSITIVE_ON_CONDITION`: OpenVLA-OFT does not improve over Base;
  do not design Ours from this condition.
- `INFRASTRUCTURE_BLOCKED`: simulator/model execution fails under the safety
  rules; repair only the runner or move to the next selected prior ecosystem.

If residual is found, run the smallest available upper-bound/headroom check
before any Ours design. If no residual remains, preregister the next
claim-specific condition or fall back to pi0.5/OpenPI.

## Completed Residual Outcome

The frozen `epoch5_libero10_residual_v1` condition completed after
preregistration:

| Policy | Successes | Episodes | Infrastructure failures |
|---|---:|---:|---:|
| SmolVLA frozen-base exact-init | 7 | 16 | 0 |
| Quantized OpenVLA-OFT INT4 | 14 | 16 | 0 |

OpenVLA-OFT INT4 improves over Base and leaves a residual on
`libero_10/task_8`:

- task 8: Base 3/8, OpenVLA-OFT 6/8;
- task 9: Base 4/8, OpenVLA-OFT 8/8.

The two OpenVLA-OFT residual failures are task 8 reset identities `20260721`
and `20260722`, corresponding to official initial-state indices `10` and `11`.

## Completed Upper/Headroom Diagnostic

The smallest available headroom check was a task-level HDF5 expert exact-init
teacher replay for task 8:

- artifact:
  `runs/openvla_oft_int4/epoch5_libero10_residual_expert_headroom_task8_demo10.json`;
- task: `libero_10/task_8`;
- demo: `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5::demo_10`;
- result: success, reward `1.0`, done/success at step `377`;
- exact demo init-state proof: `after_set_state_l2_to_hdf5_init = 0.0`;
- training/download/VLA load: none.

Caveat: this is not a same-reset upper bound. Local HDF5 demo init-state hashes
did not match the frozen benchmark initial-state hashes for residual reset
identities `20260721`/`20260722`. The result is therefore task-level
recoverability evidence, sufficient to avoid `CONDITION_TOO_SEVERE`, but the
identity mismatch must be carried into Ours design and claims.

## Current Gate Result

Decision: `R2R_OFT_OFFLINE_SELECTION_NOT_PASSED`.

Ours design is now permitted only for the exact task-8 residual limitation and
must generate at most two candidates. Do not broaden into a generic method
search.

## Completed Ours Candidate Selection

Execution status: `COMPLETE`

Exactly two candidates were generated after the Base/Prior/headroom gate:

1. `R2R-OFT`: Residual Remaining-object Reweighted OFT.
2. `MPC-OFT`: Moka-pair Counterfactual Phase OFT.

Selected method: `R2R-OFT`.

The selected mechanism is a narrow prior extension of OpenVLA-OFT: preserve the
same two-image, proprioceptive, continuous action-chunk inference path, and use
LoRA/QLoRA only as implementation infrastructure for a phase-balanced imitation
objective focused on the one-pot-complete / one-pot-remaining task-8 phase.

Next frozen pre-training step: run a CPU/local data-health audit for phase
labels, record counts, split integrity, action validity, and non-privileged
deployment inputs before any training.

## Completed `R2R-OFT` Data-Health Audit

Execution status: `COMPLETE_PASS`

Artifact:
`runs/openvla_oft_int4/epoch5_r2r_oft_pretraining_data_audit.json`.

Summary:

- HDF5 demos: 50.
- Train/validation split: 40 / 10 demos.
- Total action steps: 20,794.
- Total 8-step chunks: 20,444.
- Train one-pot-remaining chunks: 9,152.
- Validation one-pot-remaining chunks: 2,332.
- Action shape/range: 7D, finite, within [-1.0, 1.0].
- Residual failure init-state hash overlap: 0.
- Privileged simulator/HDF5 state is used only for training phase labels, not
  deployment inputs.

Decision: `R2R_OFT_DATA_HEALTH_PASS_PRETRAINING_READY`.

## Completed `R2R-OFT` One-Batch QLoRA Gradient Smoke

Execution status: `COMPLETE_PASS`

Artifact:
`runs/openvla_oft_int4/epoch5_r2r_oft_qlora_gradient_smoke.json`.

Scope: mechanism/feasibility only. This loaded the quantized OpenVLA-OFT prior,
attached a rank-4 LoRA adapter, selected one audited one-pot-remaining HDF5
chunk, computed the phase-weighted chunk L1 loss, and ran backward. It did not
take an optimizer step and did not write a checkpoint.

Summary:

- LoRA rank/alpha: 4 / 8.
- Phase-weight lambda: 2.0.
- Sample: `demo_0`, timestep 147.
- Weighted loss: 0.99609375.
- Trainable LoRA parameters: 13,853,536.
- Nonzero-gradient parameter tensors: 425.
- Gradient global norm: 4.082890925442449.
- CUDA allocated/peak allocated: 5,917.196 / 8,121.43 MiB.
- Training run happened: false.
- Optimizer step happened: false.
- Checkpoint written: false.

Decision: `R2R_OFT_QLORA_GRADIENT_SMOKE_PASS`.

## Frozen `R2R-OFT` Bounded Training Configuration

Execution status: `FROZEN_PASS`

Artifact:
`runs/openvla_oft_int4/epoch5_r2r_oft_training_spec_v1.json`.

SHA-256:
`1875b93f9249597c026f20b0bea32b13751a2df366612b209d6df96eb6870ddb`.

Frozen matrix:

1. `r2r_oft_rank4_lambda2_lr2e4_steps64`: primary selected method.
2. `uniform_oft_rank4_lambda0_lr2e4_steps64`: uniform-weight ablation.

Both arms use rank-4 / alpha-8 LoRA, the same train/validation split, the same
deterministic phase cycle `[1, 0, 1, 2]`, INT4 prior loading, no full-BF16
OpenVLA-OFT load, and a 64-step optimizer limit. Only VLA LoRA adapters are
trainable; the prior action head and proprio projector are frozen.

Selection is offline-first. The residual reset identities `20260721` and
`20260722` are disallowed for model selection or retuning. Closed-loop
evaluation on the frozen residual manifest may happen only after the offline
gate and may not create a third configuration.

Decision: `R2R_OFT_TRAINING_CONFIG_FROZEN`.

## `R2R-OFT` Trainer/Launcher Validation

Execution status: `VALIDATED_NO_TRAINING`

Implementation:

- `tca_map/r2r_oft/train_qlora.py`;
- `tca_map/r2r_oft/launch_training.py`;
- `tests/test_r2r_oft_train_qlora.py`.

Validation:

- `py_compile`: pass;
- focused tests: `14 passed`;
- dry-run launch manifest:
  `runs/openvla_oft_int4/epoch5_r2r_oft_training/r2r_oft_rank4_lambda2_lr2e4_steps64/launch_manifest.json`;
- dry-run status: `DRY_RUN`;
- training/optimizer step at manifest write: false / false.
- runtime correction after first launch attempt: use
  `/home/jiheon/venvs/openvla-oft-int4-rtx5080/bin/python`, not
  `/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python`.

First launch attempt failed before training:

- artifact:
  `runs/openvla_oft_int4/epoch5_r2r_oft_training/r2r_oft_rank4_lambda2_lr2e4_steps64/result_failed_missing_rich_20260717T1341KST.json`;
- cause: wrong WSL runtime environment missing optional OpenVLA logging
  dependency `rich`;
- training happened / optimizer steps / checkpoint written: false / 0 / false.

Decision: `R2R_OFT_TRAINER_LAUNCHER_VALIDATED`.

## Frozen Two-Arm Training Completed

Execution status: `COMPLETE`

Both frozen arms completed 64/64 optimizer steps and wrote checkpoints at
steps 16, 32, and 64. Peak CUDA allocation was 8,370.589 MiB for both arms.

Artifacts:

- primary:
  `runs/openvla_oft_int4/epoch5_r2r_oft_training/r2r_oft_rank4_lambda2_lr2e4_steps64/result.json`;
- uniform ablation:
  `runs/openvla_oft_int4/epoch5_r2r_oft_training/uniform_oft_rank4_lambda0_lr2e4_steps64/result.json`.

## Offline Validation / Selection Gate

Execution status: `COMPLETE_NO_PASS`

Fixed validation chunks: 24, from validation demos `40..49`, with phase counts
`{0: 6, 1: 12, 2: 6}`. Closed-loop rollout happened: false.

| Checkpoint | Primary phase-1 L1 | Ablation phase-1 L1 | Primary max action delta | Gate |
|---|---:|---:|---:|---|
| step 16 | 0.3845006649692853 | 0.38416459163029987 | 1.002685546875 | FAIL |
| step 32 | 0.2876454995324214 | 0.29051600955426693 | 1.010009765625 | FAIL |
| step 64 | 0.2626503886034091 | 0.2789938536783059 | 1.0028839111328125 | FAIL |

Decision: `R2R_OFT_OFFLINE_SELECTION_NOT_PASSED`.

## Simple Control: Shorter OpenVLA-OFT Requery

The predeclared no-training simple alternative was evaluated after the
`R2R-OFT` offline no-pass:

- artifact:
  `runs/openvla_oft_int4/epoch5_task8_short_requery4_openvla_int4.json`;
- SHA-256:
  `6864e691b1ad5dbfe371b309468b9f107806d12b41fe6bd5b51fd99ab00bf37e`;
- configuration: Quantized OpenVLA-OFT INT4 with `num_open_loop_steps=4`;
- task/resets: `libero_10/task_8`, reset identities `20260716..20260723`;
- result: 5/8, with failures on `20260718`, `20260720`, and `20260721`;
- comparison: original OpenVLA-OFT INT4 8-step prior was 6/8, with failures on
  `20260721` and `20260722`.

Decision: `SHORT_REQUERY4_SIMPLE_CONTROL_NOT_SELECTED`.

## Fallback Prior Ecosystem Preflight

The two preselected fallback ecosystems are not immediately runnable with the
current local assets:

- pi0.5 / OpenPI LIBERO: source cloned at
  `C:\assets\repos\openpi`, main
  `15a9616a00943ada6c20a0f158e3adb39df2ccac`. A Python 3.11 + uv bootstrap
  and `/home/jiheon/venvs/openpi-uv` environment were created. OpenPI import,
  `pi05_libero` config load, and JAX CUDA detection passed. The public
  `gs://openpi-assets/checkpoints/pi05_libero` checkpoint was downloaded
  locally, but random-input policy restore/inference exited `137` under current
  WSL memory before result JSON.
- PCD / PCD-LeRobot: source cloned/inspected at
  `C:\assets\repos\PCD` and `C:\assets\repos\PCD-LeRobot`; official setup
  still requires TensorFlow CUDA, JAX CUDA 11, PyTorch CUDA 11.8,
  Grounded-SAM2, SAM2, GroundingDINO, Inpaint-Anything/big-lama, Octo,
  OpenVLA-7B, SigLIP, T5, and extra/manual checkpoints. Official default
  evaluation uses `num_gpus=8`.

Decision: `ALL_THREE_PRIOR_ECOSYSTEMS_EXECUTION_BLOCKED_OR_NO_GO`.

Next step: strategic decision. A fair continuation needs larger-memory/remote
OpenPI, substantial PCD setup on adequate GPU resources, or a new
prior-ecosystem selection beyond the initial three. Closed-loop Ours rollout is
disallowed for the trained `R2R-OFT` checkpoints.

## Second-Pass Selected Prior: LightVLA on LIBERO-10

Decision: `SECOND_PASS_SELECTED_LIGHTVLA_LIBERO10_PRIOR_PREFLIGHT`.

This is still prior reproduction/preflight, not Ours design.

Selected official prior:

- paper: `The Better You Learn, The Smarter You Prune`, arXiv `2509.12594`;
- official repository: `https://github.com/LiAutoAD/LightVLA`;
- local checkout: `C:\assets\repos\LightVLA`, HEAD
  `a4680fda5ffe73029190ac97328aa34b0e87a45a`;
- official checkpoint selected for first execution:
  `TTJiang/LightVLA-libero-10`, Hugging Face revision
  `d40628fe49fbbca841e1ae9c7b17e2fb6abe7aa7`;
- metadata size: `15,454,705,546` bytes (`14.393` GiB);
- local download target:
  `/home/jiheon/assets/checkpoints/lightvla/TTJiang_LightVLA-libero-10`;
- official evaluation entry point:
  `experiments/robot/libero/run_libero_eval.py`;
- bounded run directory:
  `runs/lightvla_prior/download_lightvla_libero10_20260717T1520KST`.

Preflight gates:

1. Source/import gate: official eval config imports in the existing
   `/home/jiheon/venvs/openvla-oft-int4-rtx5080` environment after installing
   lightweight missing package `joblib`. The stack is not a numerically exact
   official environment because local PyTorch is `2.10.0+cu128` while LightVLA
   reports Python `3.10.14`, PyTorch `2.2.0`, and a custom transformers
   `4.40.1` fork on NVIDIA H20.
2. Checkpoint gate: complete. The selected checkpoint downloaded to
   `/home/jiheon/assets/checkpoints/lightvla/TTJiang_LightVLA-libero-10`;
   the download run exited `0`.
3. Load gate: complete. `load_in_4bit=True` loaded on the local RTX 5080 with
   about 4.99 GB peak CUDA allocation.
4. Bounded diagnostic gate: complete. LightVLA was run on the matched
   `libero_10/task_8` reset identities `20260716..20260723`.

Bounded diagnostic result:

| Policy | Task-8 successes | Failed reset identities |
|---|---:|---|
| SmolVLA frozen base | 3/8 | `20260716`, `20260717`, `20260721`, `20260722`, `20260723` |
| OpenVLA-OFT INT4 | 6/8 | `20260721`, `20260722` |
| LightVLA 4-bit | 6/8 | `20260716`, `20260723` |

Decision: `LIGHTVLA_PRIOR_DIAGNOSTIC_COMPLEMENTARY_RESIDUAL_FOUND`.

Do not train LightVLA locally, do not retune OpenVLA-OFT, and do not write a
LightVLA-inspired method as though this were already Ours. The measured
second-pass residual is cross-prior complementarity: LightVLA solves the two
OpenVLA-OFT failures, while OpenVLA-OFT solves the two LightVLA failures.

## First Selected Method Around the Complementarity Residual

Decision: `SELECTED_CR_LIGHTVLA_FOR_STAGE0`.

Exactly two candidates were considered:

1. `CR-LightVLA`: a no-training collision-rescue extension of LightVLA's
   token-pruner. It keeps the original first-choice unique tokens and rescues
   second-choice tokens only for dynamic-query collisions.
2. `ATCD`: a QLoRA action-teacher distillation method using LightVLA/OpenVLA
   proposal complementarity on training demos.

`CR-LightVLA` was selected first because it is the narrowest direct extension
of LightVLA's actual mechanism and can be evaluated without pseudo-label
training or a generic policy gate.

Stage-0 result: `CR-LightVLA` completed the same 8 reset identities with 6/8
successes, failures `20260718` and `20260723`. It fixed `20260716`, preserved
`20260721`/`20260722`, regressed `20260718`, and failed to fix `20260723`.

Decision: `CR_LIGHTVLA_STAGE0_NO_PROTOTYPE_GO`.

## Completed ATCD Teacher-Signal Audit

Execution status: `COMPLETE_NO_GO`

The deferred `ATCD` candidate was audited before any QLoRA training. The audit
used 24 fixed task-8 HDF5 validation chunks from demos `40..49` and compared
normalized 8x7 action proposals from OpenVLA-OFT INT4 and LightVLA. It did not
train, take optimizer steps, write checkpoints, or run closed-loop simulation.

Artifacts:

- runner: `scripts/epoch5_atcd_teacher_signal_audit.py`;
- result:
  `runs/lightvla_prior/atcd_teacher_signal_20260717T1620KST/atcd_teacher_signal_result_v2.json`.

Summary:

| Signal | Value |
|---|---:|
| OpenVLA-OFT wins | 9/24 |
| LightVLA wins | 15/24 |
| OpenVLA-OFT mean L1 | 0.4338486312578122 |
| LightVLA mean L1 | 0.41920601141949493 |
| Oracle mean L1 | 0.4083502360930045 |
| Oracle absolute gain vs best single | 0.010855775326490402 |
| Oracle relative gain vs best single | 0.025896039252230916 |
| Phase-1 oracle absolute gain | 0.013157747685909271 |

Decision: `ATCD_TEACHER_SIGNAL_NOT_ENOUGH`.

ATCD has measurable policy complementarity, but it missed the frozen relative
oracle-gain threshold of `0.03`. Therefore no ATCD training or rollout is
authorized from this audit. The next plan step is a new bounded method-selection
cycle around the same cross-prior complementarity, still without retuning on
the tested reset identities.

## Completed Second-Pass Fallback Prior Preflight

Execution status: `COMPLETE_BLOCKED`

The remaining second-pass exact-three prior ecosystems were checked after the
LightVLA-derived method attempts did not produce prototype-go evidence.

| Ecosystem | Local result | Decision |
|---|---|---|
| RIPT-VLA | Official source cloned at `C:\assets\repos\ript-vla`, HEAD `440990e8864e12e4578b490ff6359e4f2c49ae3e`; HF model metadata is 32 files / 6.180 GiB. Source imports in the existing OpenVLA runtime. OpenVLA-OFT RIPT assets cover LIBERO Goal/Spatial/Object/Long, not the current `libero_10/task_8` residual, and new RIPT training is interactive RL with official 4-GPU recommendation. QueST checkpoints are small but not an exact both-moka `libero_10` prior. | `RIPT_VLA_FALLBACK_NOT_COMPARABLE_OR_RESOURCE_BLOCKED` |
| VLA-GSE | Official source cloned at `C:\assets\repos\VLA-GSE`, HEAD `200cdc245880322f2bef7b24ec506063a0f35e8c`. No local trained checkpoint. Official setup uses `Qwen/Qwen3-VL-4B-Instruct`, LeRobot-format LIBERO data, 8-GPU training, and a policy-server evaluation stack. | `VLA_GSE_SOURCE_ONLY_RESOURCE_BLOCKED` |

No checkpoint download, training, or rollout was run for these fallbacks.

Decision: `SECOND_PASS_PRIOR_FALLBACKS_BLOCKED_AFTER_LIGHTVLA_NO_GO`.

Next step: start a third exact-three official-prior ecosystem selection pass.
Do not claim RIPT-VLA or VLA-GSE prior results from these source-only/resource
preflights.

## Completed Third-Pass Official Prior Diagnostic

Execution status: `COMPLETE_SOLVED_BY_OFFICIAL_PRIOR`

Third-pass exact-three candidates were X-VLA, VLA-0, and VLA-JEPA. X-VLA was
selected first because it had the lightest executable LIBERO-relevant official
checkpoint (`2toINF/X-VLA-Libero`, 15 files / 3.280 GiB) plus a LIBERO-long
adapter option, while VLA-0 and VLA-JEPA model repositories were about 21.459
GiB and 22.961 GiB respectively.

Local X-VLA preflight passed:

- source: `C:\assets\repos\X-VLA`, HEAD
  `6bc2513f5f1cbec715cc668b414392a6cae5c671`;
- load artifact:
  `runs/xvla_prior/load_xvla_libero_20260717T1649KST/result.json`;
- action-smoke artifact:
  `runs/xvla_prior/action_smoke_xvla_libero_20260717T1654KST/result.json`;
- runner: `scripts/epoch5_xvla_libero10_task8_eval.py`.

The matched task-8 official-prior diagnostic then completed:

| Policy | Task | Episodes | Successes | Infrastructure failures | Artifact |
|---|---|---:|---:|---:|---|
| X-VLA-Libero | `libero_10/task_8` | 8 | 8 | 0 | `runs/xvla_prior/diagnostic_xvla_task8_all_20260717T1705KST/result.json` |

Decision: `X_VLA_SOLVES_CURRENT_TASK8_RESIDUAL_NO_OURS_TARGET`.

Planning consequence: do not design Ours on the current task-8 residual. The
residual is solved by an executable official prior, including the original
OpenVLA-OFT failures `20260721` and `20260722` and the LightVLA failures
`20260716` and `20260723`. A fair continuation must select a new residual
condition against the latest executable official-prior set, with X-VLA included
as a prior baseline.

## Completed Post-X-VLA Residual Search Scan

Execution status: `COMPLETE_NO_CANDIDATE`

The X-VLA runner was generalized for residual search and used for a fresh
single-identity scan over all LIBERO-10 tasks:

| Scan | Suite | Tasks | Reset identity | Initial-state index | Successes | Infrastructure failures |
|---|---|---:|---:|---:|---:|---:|
| `runs/xvla_prior/failure_scan_libero10_identity20260724_20260717T1716KST` | `libero_10` | 10 | `20260724` | 13 | 10/10 | 0 |

Decision: `NO_NEW_XVLA_LIBERO10_SINGLE_IDENTITY_RESIDUAL_FOUND`.

This is not a benchmark claim. It only says the first fresh X-VLA residual
mining seed did not expose a failure. Next search must broaden to more
identities or additional suites before any Ours design is allowed.

## New Residual Candidate: X-VLA LIBERO-10 Task 1

Execution status: `OFFICIAL_PRIOR_RESIDUAL_FOUND`

Focused X-VLA sweep:

| Policy | Task | Identities | Successes | Failures | Infrastructure failures | Artifact |
|---|---|---|---:|---|---:|---|
| X-VLA-Libero | `libero_10/task_1` | `20260724..20260731` | 6/8 | `20260725`, `20260727` | 0 | `runs/xvla_prior/diagnostic_xvla_libero10_task1_id20260724_20260731_20260717T1729KST/result.json` |

Decision: `X_VLA_LIBERO10_TASK1_RESIDUAL_FOUND_MATCHED_BASE_PENDING`.

Next step: run the matched Base/Prior diagnostic on this exact task/reset
window. Do not design, train, or retune Ours until the matched base result and
residual headroom are verified.
