# Epoch 5 Prior Reproduction Result

Selected prior ecosystem: OpenVLA-OFT on LIBERO.

## Result

Decision: `R2R_OFT_OFFLINE_SELECTION_NOT_PASSED`

Epoch 5 completed the selected-prior-first diagnostic sequence before Ours
design, then generated exactly two Ours candidates and selected `R2R-OFT`.
Bounded optimizer-step training happened only after the training spec was
frozen. No new download, full-BF16 OpenVLA-OFT attempt, or closed-loop Ours
evaluation has happened.
After the `R2R-OFT` offline no-pass, the preregistered simple alternative
`shorter OpenVLA-OFT action-chunk requery without training` was run on the
task-8 residual resets. It did not beat the original 8-step OpenVLA-OFT prior:
4-step requery was 5/8 versus the original 6/8.

The recovered hard-slice condition established that the selected prior is
locally runnable and positive, but saturated. The preregistered
`epoch5_libero10_residual_v1` condition then produced the required matched
Base/Prior residual structure:

- SmolVLA frozen-base exact-init: 7/16.
- Quantized OpenVLA-OFT INT4: 14/16.
- OpenVLA-OFT still fails 2/16, both on `libero_10/task_8`.
- No infrastructure failures occurred in either matched run.

The smallest available upper/headroom check was then run as a task-level HDF5
expert replay on `libero_10/task_8`. It succeeded, but it is not a same-reset
upper bound because local HDF5 demo init-state hashes do not match the frozen
benchmark initial-state hashes used in the Base/Prior diagnostic.

## Validation Commands

Focused OpenVLA artifact validation:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests\test_openvla_oft_int4_gate.py -q
```

Observed after manifest-control patches: `5 passed`.

## Recovered Hard-Slice Evidence

| Evidence | Value |
|---|---|
| Prior result | `reports/openvla_oft_quantized_hard_slice_result.json` |
| Prior result summary | OpenVLA-OFT INT4 completed 20/20 and succeeded 20/20 |
| Matched Base result | `runs/openvla_oft_int4/hard_slice_smolvla_exact.json` |
| Matched Base summary | SmolVLA frozen-base exact-init completed 20/20 and succeeded 11/20 |
| Matched manifest | `reports/openvla_oft_quantized_hard_slice_manifest.json` |
| Policy-load evidence | `reports/openvla_oft_int4_policy_load_result.md` |
| Memory preflight | `reports/openvla_oft_int4_memory_preflight.md` |
| Quantization caveat | INT4 is not claimed numerically identical to full-precision OpenVLA-OFT |
| Local checkpoint | `/home/jiheon/assets/checkpoints/openvla-oft/moojink_openvla-7b-oft-finetuned-libero-spatial-object-goal-10` |
| Local checkpoint size | 15G by WSL `du -sh`; result metadata visible size 14.845 GiB |
| Local official repo | `C:\assets\repos\openvla-oft`, HEAD `e4287e94541f459edc4feabc4e181f537cd569a8`, dirty from prior local compatibility changes |

The recovered hard slice is prior-positive but unusable for Ours design because
OpenVLA-OFT INT4 saturates it at 20/20.

## Matched Residual Diagnostic: `epoch5_libero10_residual_v1`

Frozen condition:

- tasks: `libero_10/task_8` and `libero_10/task_9`;
- reset identities: `20260716..20260723`, mapping to official initial-state
  indices `5..12`;
- planned episodes: `16` SmolVLA frozen-base exact-init and `16` Quantized
  OpenVLA-OFT INT4;
- matched row SHA-256:
  `13642c7bed5e7d5944f7377e9848aeec1b9090be96d110362b53bc9cd9a3b3b2`.

| Policy | Completed | Successes | Failures | Infrastructure failures | Result |
|---|---:|---:|---:|---:|---|
| SmolVLA frozen-base exact-init | 16 | 7 | 9 | 0 | `runs/openvla_oft_int4/epoch5_libero10_residual_smolvla_exact.json` |
| Quantized OpenVLA-OFT INT4 | 16 | 14 | 2 | 0 | `runs/openvla_oft_int4/epoch5_libero10_residual_openvla_int4.json` |

Per-task result:

| Task | SmolVLA Base | OpenVLA-OFT INT4 | Interpretation |
|---|---:|---:|---|
| `libero_10/task_8` put both moka pots on the stove | 3/8 | 6/8 | prior improves but leaves residual |
| `libero_10/task_9` put the yellow and white mug in the microwave and close it | 4/8 | 8/8 | prior improves and saturates this task |

The two OpenVLA-OFT residual failures are:

| Task | Reset identity | Initial-state index | Initial-state SHA-256 |
|---|---:|---:|---|
| `libero_10/task_8` | `20260721` | 10 | `098c331d6cad1772de3e8ee22a7f983b4c109493f657735e7e7e78319ac1f455` |
| `libero_10/task_8` | `20260722` | 11 | `7753c014bd3caf96ff9694b20b5ea40358f64730fa10607312183377f69fb305` |

Manifest/result integrity:

| Artifact | SHA-256 |
|---|---|
| OpenVLA residual manifest | `b2de1d683d7ab0c5aff7462857f0366bd72c9208c98b2e8566e6a42a296b5adf` |
| SmolVLA residual manifest | `6defb7769a75b595bc8456e6938254d7185d2b03fd94a4bda4fd0a95464a837c` |
| OpenVLA residual result | `29cddfb319df9f3ffa19bd34f8b571e69199118783338423eca25e94ee16f1e9` |
| SmolVLA residual result | `24569154c305ef2dbfe25d71ba2ea8d9c5de5b7c1d85851596ba93671a1e38c1` |

The manifests have identical `(suite, task_id, reset_identity,
initial_state_index, initial_state_sha256)` rows. This satisfies the matched
Base/Prior part of the required structure.

## Headroom Diagnostic

Artifact:
`runs/openvla_oft_int4/epoch5_libero10_residual_expert_headroom_task8_demo10.json`.

| Field | Value |
|---|---|
| Diagnostic | task-level HDF5 expert exact-init replay |
| Task | `libero_10/task_8` |
| Demo | `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5::demo_10` |
| Same benchmark reset as residual failure? | No |
| HDF5 demo init-state SHA-256 | `3ebe5ab024c6896e57dee59422f47ef631355a9e20f10a082fae7ad7f533f81a` |
| Nearest residual reset index represented by label | OpenVLA residual failure index 10, but hash differs |
| Expert replay result | success, reward 1.0, done at step 377 |
| Exact demo init-state set proof | `after_set_state_l2_to_hdf5_init = 0.0` |
| Training/download/VLA load | none |

Interpretation: task-level recoverable headroom is positive for the residual
task, but this is weaker than a same-reset oracle. It is sufficient to avoid
classifying the condition as floor/too-severe at this stage, while the exact
identity mismatch must remain visible in any Ours design.

## Required Structure

| Required structure | Status |
|---|---|
| Base has meaningful failure | COMPLETE: SmolVLA 7/16 |
| Prior improves | COMPLETE: OpenVLA-OFT INT4 14/16 |
| Prior leaves residual gap | COMPLETE: OpenVLA-OFT fails 2/16 |
| Condition neither floor nor saturated | COMPLETE for task 8; prior saturated task 9 only |
| OpenVLA-OFT does not fully solve it | COMPLETE on task 8 |
| Upper/headroom indicates recoverability | PARTIAL-COMPLETE: task-level expert replay positive; same-reset expert unavailable |

## Ours Candidate Generation

Status: `EXACTLY_TWO_CANDIDATES_GENERATED_ONE_SELECTED`

Candidate generation happened only after the Base/Prior/headroom gate above.
The residual limitation is not generic LIBERO weakness: visual review of the
two OpenVLA-OFT failures shows a repeated second-object completion pattern.
OpenVLA-OFT places or reaches the stove phase for one moka pot, then times out
while trying to recover the remaining moka pot. This matches the quantitative
residual: both failures are `libero_10/task_8` reset identities `20260721`
and `20260722`.

Local supervision exists for this exact task: the task-8 HDF5 file has 50
successful demos, 20,794 total action steps, terminal reward/done in every
demo, and actions in valid 7D LIBERO range. Object/phase labels still require a
pre-training data-health audit because object poses are not directly named in
the HDF5 `obs`; they must be decoded from simulator state or bounded replay.

| Candidate | Contribution type | Core mechanism | Score | Decision |
|---|---|---|---:|---|
| `R2R-OFT`: Residual Remaining-object Reweighted OFT | `PRIOR_EXTENSION` | Keep OpenVLA-OFT's two-image + proprio + 8x7 continuous action chunk path, but LoRA/QLoRA-finetune with a phase-balanced imitation objective that upweights successful expert chunks where exactly one moka pot is already on/near the stove and the remaining pot still needs pickup/placement. | 84/100 | SELECTED |
| `MPC-OFT`: Moka-pair Counterfactual Phase OFT | `PRIOR_EXTENSION` | Build object-order/paraphrase augmentation for two visually similar moka pots so the policy sees balanced "remaining pot" and left/right object-role variants during LoRA/QLoRA fine-tuning. | 75/100 | NOT SELECTED |

Selected method: `R2R-OFT`.

Selection rationale: `R2R-OFT` is the narrowest extension of the selected
prior and directly targets the observed residual phase. `MPC-OFT` is plausible
but adds a broader object-role augmentation hypothesis and depends more heavily
on reliable object identity synthesis.

## Selected Method Sketch: `R2R-OFT`

Prior mechanism being extended:

- OpenVLA-OFT fine-tunes OpenVLA via LoRA.
- The selected checkpoint uses two image inputs, proprioception, continuous L1
  action regression, LIBERO no-noop action normalization, and 8-step action
  chunks.
- The local prior is Quantized OpenVLA-OFT INT4, not a full-precision
  reproduction.

Scientific method:

- Learn a residual-state-biased OFT update for the exact task-8 second-object
  completion phase.
- Training labels may use simulator/HDF5 state to identify phase, but
  deployment must use only the same OpenVLA-OFT inputs: RGB, wrist RGB,
  proprio, and instruction.
- LoRA/QLoRA is implementation infrastructure only. The scientific claim is
  phase-balanced remaining-object supervision for strong VLA residual failures.

Mathematical form, to be audited before training:

Let `x_t = (I_t, W_t, p_t, l)` be the deployment input, `a*_{t:t+7}` the expert
7D action chunk, and `pi_theta` the OpenVLA-OFT action head. Let
`m_t = 1` when training-only phase labels indicate exactly one moka pot is
already on/near the stove and at least one moka pot remains off-target.

The primary objective is weighted chunk imitation:

`L(theta) = mean_t (1 + lambda * m_t) * ||pi_theta(x_t) - a*_{t:t+7}||_1`.

Identity preservation comes from zero/near-zero initialized LoRA/QLoRA
updates, bounded validation action-delta checks, and retaining non-residual
task-8 phases in the sampler. A clean-retention regularizer may be added only
if the data audit shows the weighted sampler alone causes global action drift.

Key ablation:

- Same LoRA/QLoRA scaffold and same task-8 data, but with `m_t = 0` for all
  samples or uniform weights. This tests whether residual-phase weighting, not
  generic extra task-8 adaptation, explains any gain.

Strongest simple alternative explanation:

- Shorter OpenVLA-OFT action-chunk requery on task 8, with no new training.
  This tests whether the failure is merely stale 8-step open-loop execution.

Second-backbone path:

- Apply the same phase-weighted sampler/objective to the SmolVLA adapter/QLoRA
  path using the same task-8 HDF5 phase labels and the same held-out reset
  manifest, then compare SmolVLA versus SmolVLA+`R2R-OFT`.

Pre-training gate:

- Run a CPU/local data-health audit before any training: records, phase counts,
  positive/negative phase balance, demo coverage, train/validation split, no
  overlap with confirmatory reset identities, action ranges, chunk validity,
  and whether phase labels are decodable without privileged deployment inputs.

## `R2R-OFT` Pre-Training Data-Health Audit

Status: `PASS`

Artifact:
`runs/openvla_oft_int4/epoch5_r2r_oft_pretraining_data_audit.json`.

The audit used the verified HDF5 state layout:

- `moka_pot_1_pos = state[10:13]`;
- `moka_pot_2_pos = state[17:20]`;
- verification source: exact-init LIBERO observation for `demo_0`, where the
  decoded positions matched `moka_pot_1_pos` and `moka_pot_2_pos`.

The stove/target region was inferred only from training-demo final pot
positions. The resulting phase labels are training labels only; inference
remains RGB, wrist RGB, proprioception, and instruction.

| Audit item | Result |
|---|---:|
| HDF5 demos | 50 |
| Train/validation demos | 40 / 10 |
| Total action steps | 20,794 |
| Total 8-step chunks | 20,444 |
| Terminal reward demos | 50 |
| Terminal done demos | 50 |
| Action dimension | 7 |
| Action range | [-1.0, 1.0] |
| Train one-pot-remaining chunks | 9,152 |
| Validation one-pot-remaining chunks | 2,332 |
| Residual failure init-state hash overlap | 0 |

Gate result: `R2R_OFT_DATA_HEALTH_PASS_PRETRAINING_READY`.

## `R2R-OFT` One-Batch QLoRA Gradient Smoke

Status: `PASS`

Artifact:
`runs/openvla_oft_int4/epoch5_r2r_oft_qlora_gradient_smoke.json`.

This was a mechanism/feasibility smoke only, not a training run. It loaded the
quantized OpenVLA-OFT prior, attached a rank-4 LoRA adapter, selected one
audited one-pot-remaining HDF5 chunk, computed the phase-weighted chunk L1
loss, and ran backward to verify that LoRA parameters receive finite nonzero
gradients within local VRAM.

| Smoke item | Result |
|---|---:|
| LoRA rank / alpha | 4 / 8 |
| Phase-weight lambda | 2.0 |
| Sample | `demo_0`, timestep 147 |
| Sample phase count on stove | 1 |
| Base L1 | 0.33203125 |
| Phase weight | 3.0 |
| Weighted loss | 0.99609375 |
| Trainable LoRA parameters | 13,853,536 |
| Nonzero-gradient parameter tensors | 425 |
| Gradient global norm | 4.082890925442449 |
| CUDA allocated / peak allocated | 5,917.196 / 8,121.43 MiB |
| Optimizer step happened | false |
| Checkpoint written | false |
| Training run happened | false |

Gate result: `R2R_OFT_QLORA_GRADIENT_SMOKE_PASS`.

## `R2R-OFT` Bounded Training Configuration Freeze

Status: `FROZEN_PASS`

Artifact:
`runs/openvla_oft_int4/epoch5_r2r_oft_training_spec_v1.json`.

SHA-256:
`1875b93f9249597c026f20b0bea32b13751a2df366612b209d6df96eb6870ddb`.

The first optimizer-step training attempt is now constrained to exactly two
arms:

| Arm | Role | Rank / alpha | Lambda | Max optimizer steps |
|---|---|---:|---:|---:|
| `r2r_oft_rank4_lambda2_lr2e4_steps64` | primary selected method | 4 / 8 | 2.0 | 64 |
| `uniform_oft_rank4_lambda0_lr2e4_steps64` | uniform-weight ablation | 4 / 8 | 0.0 | 64 |

Shared constraints:

- single local CUDA 16 GB, INT4 prior load only;
- no full-BF16 OpenVLA-OFT load;
- two OpenVLA-OFT images, proprioception, continuous L1, 8-step 7D chunks;
- train demos `0..39`, validation demos `40..49`;
- deterministic phase cycle `[1, 0, 1, 2]`;
- trainable components: VLA LoRA adapters only;
- frozen components: prior action head and prior proprio projector;
- save/evaluate at steps `16`, `32`, and `64`;
- maximum CUDA peak memory: 14,500 MiB;
- maximum wall time: 90 minutes per arm.

Selection is offline-first. The residual reset identities `20260721` and
`20260722` must not be used for model selection or retuning. Closed-loop
evaluation on the frozen residual manifest is allowed only after an offline
gate; if it fails, no new configuration may be generated from those resets.

Gate result: `R2R_OFT_TRAINING_CONFIG_FROZEN`.

## `R2R-OFT` Trainer/Launcher Validation

Status: `VALIDATED_NO_TRAINING`

New implementation files:

- `tca_map/r2r_oft/train_qlora.py`;
- `tca_map/r2r_oft/launch_training.py`;
- `tests/test_r2r_oft_train_qlora.py`.

Validation:

- `py_compile` passed for `train_qlora.py`, `launch_training.py`, and
  `training_spec.py`;
- focused tests passed: `14 passed`;
- dry-run launch manifest:
  `runs/openvla_oft_int4/epoch5_r2r_oft_training/r2r_oft_rank4_lambda2_lr2e4_steps64/launch_manifest.json`;
- dry-run status: `DRY_RUN`;
- training happened at launch-manifest write: false;
- optimizer step happened at launch-manifest write: false;
- dry-run command targets `tca_map.r2r_oft.train_qlora`.
- runtime correction after first launch attempt: the WSL launcher default was
  switched from `/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python`
  to `/home/jiheon/venvs/openvla-oft-int4-rtx5080/bin/python`, the local venv
  that imports OpenVLA-OFT/Prismatic dependencies and exposes CUDA.

First launch attempt:

- artifact:
  `runs/openvla_oft_int4/epoch5_r2r_oft_training/r2r_oft_rank4_lambda2_lr2e4_steps64/result_failed_missing_rich_20260717T1341KST.json`;
- commit: `c25fc46792eda395a2af5167306fb8c4f071744a`;
- status: `FAILED`;
- cause: wrong WSL runtime environment missing optional OpenVLA logging
  dependency `rich`;
- training happened: false;
- optimizer steps completed: 0;
- checkpoint written: false.

Gate result: `R2R_OFT_TRAINER_LAUNCHER_VALIDATED`.

## `R2R-OFT` Frozen Two-Arm Training Result

Status: `COMPLETE`

Both frozen arms completed 64/64 optimizer steps under the committed spec.

| Arm | Artifact | SHA-256 | Steps | Peak CUDA MiB | Checkpoints |
|---|---|---|---:|---:|---|
| `r2r_oft_rank4_lambda2_lr2e4_steps64` | `runs/openvla_oft_int4/epoch5_r2r_oft_training/r2r_oft_rank4_lambda2_lr2e4_steps64/result.json` | `a03a3703a3ee61c420e082e41bdb22b6ab4105e507edb21f0e323be078b2326d` | 64 | 8,370.589 | 16 / 32 / 64 |
| `uniform_oft_rank4_lambda0_lr2e4_steps64` | `runs/openvla_oft_int4/epoch5_r2r_oft_training/uniform_oft_rank4_lambda0_lr2e4_steps64/result.json` | `0b741c1f1fbbae5fb8daa623287d319ae548be1ea114945930625de6e41b0cd6` | 64 | 8,370.589 | 16 / 32 / 64 |

The repaired launch used commit
`71de9562ef467b1f65e481d393ffdaad0547a7a0` and the local OpenVLA runtime
`/home/jiheon/venvs/openvla-oft-int4-rtx5080/bin/python`.

## `R2R-OFT` Offline Validation / Selection Gate

Status: `COMPLETE_NO_PASS`

Validation used 24 fixed validation chunks from demos `40..49`, with phase
counts `{0: 6, 1: 12, 2: 6}`. No closed-loop rollout was run.

Prior phase-1 validation L1: `0.4049811102449894`.

| Checkpoint | Primary phase-1 L1 | Ablation phase-1 L1 | Primary beats ablation | Primary max action delta vs prior | Gate |
|---|---:|---:|---|---:|---|
| step 16 | 0.3845006649692853 | 0.38416459163029987 | false | 1.002685546875 | FAIL |
| step 32 | 0.2876454995324214 | 0.29051600955426693 | true | 1.010009765625 | FAIL |
| step 64 | 0.2626503886034091 | 0.2789938536783059 | true | 1.0028839111328125 | FAIL |

Artifacts:

- `runs/openvla_oft_int4/epoch5_r2r_oft_offline_validation_step0016.json`,
  SHA-256 `e0c35f01f95d2b8101ddfc9f9d4ba93af09db84b5bff03cdb8f31ec0cc8b1974`;
- `runs/openvla_oft_int4/epoch5_r2r_oft_offline_validation_step0032.json`,
  SHA-256 `125b99249c9e1f2ac5058a1f1bcc43714a0b6311f8e9edb85433d16609254cda`;
- `runs/openvla_oft_int4/epoch5_r2r_oft_offline_validation_step0064.json`,
  SHA-256 `a66a02bfc873ea26596321a1147c7b51f8511d5f793abe479bc4f4adc7dbb0fc`.

Interpretation: the selected mechanism improved the residual one-pot validation
phase, especially at step 64, but all primary checkpoints violated the frozen
max action-delta bound. Under the preregistered offline-selection rule,
closed-loop evaluation is therefore disallowed.

Gate result: `R2R_OFT_OFFLINE_SELECTION_NOT_PASSED`.

## Simple Control: Shorter OpenVLA-OFT Requery

After the offline gate disallowed closed-loop Ours rollout, the predeclared
simple alternative from the selected-method sketch was evaluated as a
no-training prior/control variant, not as Ours.

Artifact:
`runs/openvla_oft_int4/epoch5_task8_short_requery4_openvla_int4.json`,
SHA-256 `6864e691b1ad5dbfe371b309468b9f107806d12b41fe6bd5b51fd99ab00bf37e`.

Configuration: Quantized OpenVLA-OFT INT4, task `libero_10/task_8`, reset
identities `20260716..20260723`, `num_open_loop_steps=4`, no training, no new
downloads, no CPU/disk offload detected.

| Variant | Successes | Failures | Failed reset identities |
|---|---:|---:|---|
| Original OpenVLA-OFT INT4, 8-step chunks | 6/8 | 2 | `20260721`, `20260722` |
| Short-requery OpenVLA-OFT INT4, 4-step chunks | 5/8 | 3 | `20260718`, `20260720`, `20260721` |

Interpretation: shorter requery fixed one original residual failure
(`20260722`) but introduced two regressions (`20260718`, `20260720`) and still
failed `20260721`. This is a useful control but not a rescue route and not a
selected method. It reinforces that the task-8 residual is sensitive to action
chunking and cannot be claimed solved by a trivial requery change.

Gate result: `SHORT_REQUERY4_SIMPLE_CONTROL_NOT_SELECTED`.

## Fallback Prior Ecosystem Preflight

Because the selected OpenVLA-OFT residual route did not produce a selectable
Ours checkpoint or simple-control rescue, the two preselected fallback
ecosystems from `reports/epoch5_prior_ecosystem_selection.md` were checked
before any download, install, or rollout.

| Ecosystem | Local readiness result | Decision |
|---|---|---|
| pi0.5 / OpenPI LIBERO | Official source cloned at `C:\assets\repos\openpi`, main `15a9616a00943ada6c20a0f158e3adb39df2ccac`. A user-local Python 3.11 + uv bootstrap was created, then OpenPI dependencies synced to `/home/jiheon/venvs/openpi-uv` using an `evdev-binary` workaround after source-built `evdev` failed on missing WSL Linux headers. JAX sees the RTX 5080. The public `pi05_libero` checkpoint was downloaded, but random-input policy restore/inference was killed with exit code `137`. | source/env/checkpoint present; local memory/resource blocker before usable prior rollout; not a scientific kill |
| PCD / PCD-LeRobot | Source cloned/inspected at `C:\assets\repos\PCD` and `C:\assets\repos\PCD-LeRobot`; PCD requires Simpler/OpenVLA/Octo/pi0 plus TensorFlow CUDA, JAX CUDA 11, PyTorch CUDA 11.8, Grounded-SAM2, SAM2, GroundingDINO, Octo, OpenVLA-7B, SigLIP, T5, and big-lama/Inpaint-Anything assets. Official default evaluation uses `num_gpus=8`; contrast OpenVLA uses `n_trajs=100` and tracking/Grounded-SAM search. | local single-GPU/dependency/checkpoint blocked before fair prior run; not a scientific kill |

OpenPI artifacts:

- source: `C:\assets\repos\openpi`;
- bootstrap env: `/home/jiheon/miniconda3-official/envs/openpi-py311`;
- OpenPI env: `/home/jiheon/venvs/openpi-uv`, about 7.8 GiB;
- uv cache: `/home/jiheon/.cache/uv`, about 7.9 GiB;
- checkpoint cache: `/home/jiheon/assets/checkpoints/openpi`, about 12 GiB,
  16 files under `openpi-assets/checkpoints/pi05_libero`;
- initial `uv sync` artifact:
  `runs/openpi_pi05_setup/uv_sync_20260717/exit_code.txt`, exit `1`, missing
  Linux input headers for source-built `evdev`;
- workaround `uv sync --no-install-package evdev` plus `evdev-binary==1.9.2`:
  `runs/openpi_pi05_setup/uv_sync_evdev_binary_20260717/exit_code.txt`,
  `sync_exit=0`, `evdev_binary_exit=0`;
- policy smoke rerun:
  `runs/openpi_pi05_setup/policy_smoke_rerun_20260717/exit_code.txt`, exit
  `137`, no result JSON.

PCD artifacts:

- source: `C:\assets\repos\PCD`, main
  `cec18b820daeadfdaf080c030a1b5eb080ff75cd`;
- LeRobot source/object DB: `C:\assets\repos\PCD-LeRobot`, main
  `519b4a814e85bf9b786677d90b0ff07218729bb2`;
- official install script:
  `C:\assets\repos\PCD\scripts\install_dependencies.sh`;
- official checkpoint script:
  `C:\assets\repos\PCD\scripts\download_pretrained_weights.sh`;
- default evaluation scripts under
  `C:\assets\repos\PCD\scripts\inference\default\`.

Gate result: `ALL_THREE_PRIOR_ECOSYSTEMS_EXECUTION_BLOCKED_OR_NO_GO`.

## Next Decision

The next action is strategic: do not claim a pi0.5 or PCD prior result, and do
not add a third local OpenVLA task-8 candidate. A fair continuation needs either
a larger-memory/remote OpenPI runtime, substantial PCD dependency/checkpoint
setup on adequate GPU resources, or a new prior-ecosystem selection beyond the
initial three. Closed-loop Ours rollout is still disallowed for the trained
`R2R-OFT` checkpoints.

## Second-Pass Prior Preflight: LightVLA on LIBERO-10

Decision: `SECOND_PASS_SELECTED_LIGHTVLA_LIBERO10_PRIOR_PREFLIGHT`.

The second official-prior selection pass chose LightVLA as the next executable
prior ecosystem after the initial OpenVLA-OFT/OpenPI/PCD set was exhausted
locally.

Local source/import status:

- official source cloned to `C:\assets\repos\LightVLA`;
- local source HEAD: `a4680fda5ffe73029190ac97328aa34b0e87a45a`;
- official eval module:
  `C:\assets\repos\LightVLA\experiments\robot\libero\run_libero_eval.py`;
- existing WSL env:
  `/home/jiheon/venvs/openvla-oft-int4-rtx5080/bin/python`;
- import smoke: `GenerateConfig` imports after installing lightweight missing
  package `joblib`;
- CUDA detected: `NVIDIA GeForce RTX 5080`;
- caveat: local stack uses PyTorch `2.10.0+cu128`, not the exact LightVLA
  reported PyTorch `2.2.0` / H20 environment.

Selected checkpoint:

- repo: `TTJiang/LightVLA-libero-10`;
- revision: `d40628fe49fbbca841e1ae9c7b17e2fb6abe7aa7`;
- metadata size: `15,454,705,546` bytes (`14.393` GiB);
- local target:
  `/home/jiheon/assets/checkpoints/lightvla/TTJiang_LightVLA-libero-10`;
- download run:
  `runs/lightvla_prior/download_lightvla_libero10_20260717T1520KST`.

Execution status: `COMPLETE_BOUNDED_PRIOR_DIAGNOSTIC`.

Download and load evidence:

- checkpoint download run:
  `runs/lightvla_prior/download_lightvla_libero10_20260717T1520KST`;
- download exit code: `0`;
- local checkpoint directory:
  `/home/jiheon/assets/checkpoints/lightvla/TTJiang_LightVLA-libero-10`;
- local checkpoint disk use after official loader modifications: about `15G`,
  `49` files;
- selected checkpoint metadata before load: 21 files,
  `15,454,705,546` bytes (`14.393` GiB), revision
  `d40628fe49fbbca841e1ae9c7b17e2fb6abe7aa7`;
- 4-bit load artifact:
  `runs/lightvla_prior/load_lightvla_libero10_20260717T1528KST/result_4bit.json`,
  SHA-256
  `5bc9ab4d45f99775433576d95397e97c80d4c89cc9f09dcc8bcd9945f5ff4312`;
- loaded classes: `OpenVLAForActionPrediction`, `L1RegressionActionHead`,
  `ProprioProjector`, `PrismaticProcessor`;
- peak CUDA allocation during load: `4,993,455,616` bytes.

Caveat: LightVLA's official local-path loader copies repository code into the
checkpoint directory after creating timestamped backups. This happened during
the load/episode diagnostics; the checkpoint path should therefore be treated
as a local execution directory, not a pristine Hugging Face snapshot.

The official `run_task` scheduler waits for more than 20 GB free VRAM, which is
impossible on the local 16 GB RTX 5080. The bounded diagnostic bypassed only
that scheduler and called the official LightVLA functions directly:
`GenerateConfig`, `initialize_model`, `get_libero_env`, and `run_episode`.

One-episode residual checks:

| Reset identity | Prior context | Result artifact | Success |
|---:|---|---|---|
| `20260721` | OpenVLA-OFT INT4 failed this reset | `runs/lightvla_prior/diagnostic_lightvla_libero10_task8_20260721_20260717T1530KST/result.json` | true |
| `20260722` | OpenVLA-OFT INT4 failed this reset | `runs/lightvla_prior/diagnostic_lightvla_libero10_task8_20260722_20260717T1533KST/result.json` | true |

Full matched task-8 diagnostic:

- artifact:
  `runs/lightvla_prior/diagnostic_lightvla_libero10_task8_all_20260717T1535KST/result.json`;
- SHA-256:
  `9f272655ec89f7504328e6a1e148ee538a44ce95dcfece6d253a668568ab2dcf`;
- imported module:
  `/mnt/c/assets/repos/LightVLA/experiments/robot/libero/run_libero_eval.py`;
- completed episodes: `8/8`;
- successes: `6/8`;
- failures: `20260716`, `20260723`;
- peak CUDA allocation: `5,350,705,152` bytes;
- load time in batch run: `41.993` seconds.

| Reset identity | Initial-state index | SmolVLA frozen base | OpenVLA-OFT INT4 | LightVLA 4-bit |
|---:|---:|---|---|---|
| `20260716` | 5 | fail | success | fail |
| `20260717` | 6 | fail | success | success |
| `20260718` | 7 | success | success | success |
| `20260719` | 8 | success | success | success |
| `20260720` | 9 | success | success | success |
| `20260721` | 10 | fail | fail | success |
| `20260722` | 11 | fail | fail | success |
| `20260723` | 12 | fail | success | fail |

Interpretation:

- LightVLA is prior-positive versus SmolVLA on the task-8 matched slice:
  `3/8 -> 6/8`.
- LightVLA solves both original OpenVLA-OFT INT4 residual failures
  (`20260721`, `20260722`).
- LightVLA does not dominate OpenVLA-OFT INT4 on this slice: both are `6/8`,
  but their failures are disjoint.
- The residual is therefore a cross-prior complementarity condition, not a
  simple LightVLA-only residual that OpenVLA-OFT cannot solve.
- An oracle that selects the successful policy between OpenVLA-OFT INT4 and
  LightVLA on each reset would be `8/8`, providing recoverable headroom for a
  future mechanism, but no Ours design or training has happened in this
  second-pass prior stage.

Two invalid batch-launch attempts are preserved in the same run directory:

- `*.invalid_initialize_model_signature_20260717T1539KST`: wrong call
  signature during runner setup;
- `*.invalid_imported_openvla_oft_20260717T1542KST`: Python imported the older
  `/mnt/c/assets/repos/openvla-oft` evaluation module because the runner was
  executed by absolute path before `PYTHONPATH` was pinned to LightVLA.

Decision: `LIGHTVLA_PRIOR_DIAGNOSTIC_COMPLEMENTARY_RESIDUAL_FOUND`.

Next action: update the structured report state and compact handoff. Do not
call the LightVLA result an Ours result. If method design begins later, it must
target the measured cross-prior complementarity and preserve held-out reset
discipline.

## First Method After LightVLA: Collision-Rescue LightVLA

Decision before rollout: `SELECTED_CR_LIGHTVLA_FOR_STAGE0`.

Two candidates were considered around the exact measured residual:

| Candidate | Core mechanism | Training? | Decision |
|---|---|---|---|
| `CR-LightVLA`: Collision-Rescue LightVLA | Extend LightVLA's own token-pruning rule: keep all original first-choice unique visual tokens, and only when multiple dynamic queries collide on the same first-choice token also keep each collided query's second-choice token. | no | selected |
| `ATCD`: Action-Teacher Complementarity Distillation | QLoRA-distill the better of LightVLA/OpenVLA-OFT action proposals on training demos using expert L1 as a teacher signal. | yes | not selected for first attempt; higher pseudo-labeling/training complexity |

Why `CR-LightVLA` was selected first: it directly modifies the selected
LightVLA prior mechanism, adds no parameters, does not use a generic verifier
or policy gate, and has a fixed rule before rollout.

Tracked runner:
`scripts/epoch5_lightvla_collision_rescue_eval.py`, SHA-256
`7b15ba7293ef8ae25f34c383a1d0f07036611122619af088500bb6ffdaec50d4`.

Run artifact:
`runs/lightvla_prior/cr_lightvla_task8_all_20260717T1600KST/result.json`,
SHA-256
`6c604b1493d10d8a8c54afcbecc3d8d6b87bd393483293cbf93f3a0d6474b9e6`.

Configuration:

- same LightVLA checkpoint and same matched `libero_10/task_8` reset identities
  `20260716..20260723`;
- no training, no optimizer steps, no checkpoint written;
- closed-loop rollout happened;
- official LightVLA scheduler bypassed for the same local 16 GB VRAM reason;
- official functions still used for load/env/episode execution.

Result:

| Policy | Successes | Failures |
|---|---:|---|
| OpenVLA-OFT INT4 | 6/8 | `20260721`, `20260722` |
| LightVLA 4-bit | 6/8 | `20260716`, `20260723` |
| `CR-LightVLA` | 6/8 | `20260718`, `20260723` |

Per-reset interpretation:

- fixed one LightVLA failure: `20260716`;
- preserved LightVLA's wins on OpenVLA-OFT failures: `20260721`, `20260722`;
- introduced one regression: `20260718`;
- did not fix LightVLA's other failure: `20260723`.

Telemetry: mean retained tokens increased from roughly `69..73` original
first-choice unique tokens to roughly `105..112` collision-rescued tokens,
depending on reset.

Decision: `CR_LIGHTVLA_STAGE0_NO_PROTOTYPE_GO`.

`CR-LightVLA` is not a prototype GO because it ties the selected prior and
OpenVLA-OFT on total success while changing the failure set. It is useful
mechanistic evidence that token-collision rescue can fix one over-pruning
failure without destroying the two OpenVLA-residual wins, but the regression
shows the fixed rescue rule is not yet a paper candidate.

## Second Method Audit After LightVLA: ATCD Teacher Signal

Decision: `ATCD_TEACHER_SIGNAL_NOT_ENOUGH`.

After `CR-LightVLA` produced no prototype GO, the previously deferred `ATCD`
candidate was audited before any QLoRA training. The audit asked only whether
OpenVLA-OFT INT4 and LightVLA produce enough complementary normalized HDF5
action proposals on the fixed task-8 validation chunks to justify a later
distillation run. It performed no training, no optimizer step, no checkpoint
write, and no simulator rollout.

Tracked runner:
`scripts/epoch5_atcd_teacher_signal_audit.py`, SHA-256
`868b235e3c200c8fd27526c531daf08e8c928b987f43d4c049a6b8aaed506d93`.

Artifacts:

- result:
  `runs/lightvla_prior/atcd_teacher_signal_20260717T1620KST/atcd_teacher_signal_result_v2.json`,
  SHA-256 `9ea69029528a0f81da480e12ea04dffab6a9e51d20a637e13c0f571aa1023921`;
- OpenVLA-OFT rows:
  `runs/lightvla_prior/atcd_teacher_signal_20260717T1620KST/openvla_oft_int4_rows_v2.json`,
  SHA-256 `86594bde77fdcd573c9c69bfdf05712dfc458c78b26b9e26d2b0fadf1cae7998`;
- LightVLA rows:
  `runs/lightvla_prior/atcd_teacher_signal_20260717T1620KST/lightvla_rows_v2.json`,
  SHA-256 `b55087d1f9e9f1ff27b0b3c13372b64be9802dea401d60179467c1e0f99dd174`.

Validation set: 24 fixed task-8 HDF5 validation chunks from demos `40..49`,
with phase counts `{0: 6, 1: 12, 2: 6}`. The comparison used normalized
8x7 continuous action chunks. OpenVLA-OFT used the existing unpruned text-mask
hidden-state extraction; LightVLA used its runtime-compatible final action-token
span because its pruner changes the hidden-state layout at inference.

| Metric | Value |
|---|---:|
| OpenVLA-OFT mean L1 | 0.4338486312578122 |
| LightVLA mean L1 | 0.41920601141949493 |
| Oracle best-of-two mean L1 | 0.4083502360930045 |
| Oracle absolute gain vs best single | 0.010855775326490402 |
| Oracle relative gain vs best single | 0.025896039252230916 |
| Phase-1 oracle absolute gain | 0.013157747685909271 |
| OpenVLA-OFT wins | 9/24 |
| LightVLA wins | 15/24 |

Predeclared pass criteria required both policies to win at least 3 chunks,
oracle absolute gain at least `0.01`, oracle relative gain at least `0.03`,
at least 6 phase-1 chunks, and phase-1 oracle absolute gain at least `0.01`.
The audit passed all criteria except relative gain: `0.025896 < 0.03`.

Interpretation: ATCD has measurable complementarity, but not enough by the
frozen threshold to justify a bounded QLoRA distillation run. Do not train or
roll out ATCD from this audit. The next scientific action is a new bounded
method-selection cycle around the same cross-prior complementarity, without
retuning on the tested reset identities.

## Second-Pass Fallback Prior Preflight

Decision: `SECOND_PASS_PRIOR_FALLBACKS_BLOCKED_AFTER_LIGHTVLA_NO_GO`.

After LightVLA produced a valid complementary prior diagnostic but the two
bounded LightVLA/OpenVLA method attempts (`CR-LightVLA`, `ATCD`) did not reach a
prototype-go path, the remaining second-pass prior ecosystems were preflighted
before any new method design.

### RIPT-VLA

Source-only status:

- official repo: `https://github.com/Ariostgx/ript-vla`;
- local clone: `C:\assets\repos\ript-vla`;
- local HEAD: `440990e8864e12e4578b490ff6359e4f2c49ae3e`;
- checkpoint repo: `tanshh97/RIPT_VLA`, revision
  `57532f4abbf81b89a8ff6a642a996fc54b6a6a10`;
- checkpoint metadata: 32 files, `6,635,348,819` bytes (`6.180` GiB);
- import smoke passed in the existing OpenVLA runtime:
  `/home/jiheon/venvs/openvla-oft-int4-rtx5080/bin/python`;
- checkpoint download, training, rollout: none.

Blocker: the official OpenVLA-OFT RIPT assets and scripts cover LIBERO
Goal/Spatial/Object/Long suites, not the current `libero_10/task_8`
both-moka residual. Training a new OpenVLA-OFT RIPT adapter is interactive RL,
and the official README recommends 4 GPUs for OpenVLA-OFT RIPT. The QueST
checkpoint path is lighter and importable, but the model zoo does not provide
an exact `libero_10` both-moka prior; LIBERO-90 contains related single-moka
tasks, not the matched residual.

Classification: `RIPT_VLA_FALLBACK_NOT_COMPARABLE_OR_RESOURCE_BLOCKED`.

### VLA-GSE

Source-only status:

- official repo: `https://github.com/YuhuaJiang2002/VLA-GSE`;
- local clone: `C:\assets\repos\VLA-GSE`;
- local HEAD: `200cdc245880322f2bef7b24ec506063a0f35e8c`;
- checkpoint download, training, rollout: none.

Blocker: VLA-GSE is an 8-GPU PEFT training framework around
`Qwen/Qwen3-VL-4B-Instruct` and LeRobot-format LIBERO data. The README's
reference setup trains for 80k steps and reports about 48 hours on 8 A100 GPUs.
Evaluation requires a trained checkpoint plus a two-process policy-server /
LIBERO-client setup. No local trained VLA-GSE checkpoint is present.

Classification: `VLA_GSE_SOURCE_ONLY_RESOURCE_BLOCKED`.

Second-pass conclusion: the exact-three second-pass ecosystem set is exhausted
locally. LightVLA was executable but no method reached prototype-go evidence;
RIPT-VLA and VLA-GSE are not fair executable prior results for the current
residual under local resources. The next action is a third exact-three
official-prior ecosystem selection pass.

## Third-Pass Official Prior Diagnostic: X-VLA

Decision: `X_VLA_SOLVES_CURRENT_TASK8_RESIDUAL_NO_OURS_TARGET`.

Third-pass exact-three selection:

| Rank | Ecosystem | Official assets | Local status |
|---|---|---|---|
| 1 | X-VLA | `https://github.com/2toinf/X-VLA`; `2toINF/X-VLA-Libero`; `2toINF/X-VLA-libero-long-peft` | selected and executed |
| 2 | VLA-0 | `https://github.com/NVlabs/vla0`; `ankgoyal/vla0-libero` | fallback not executed after X-VLA solved the residual |
| 3 | VLA-JEPA | `https://github.com/ginwind/VLA-JEPA`; `ginwind/VLA-JEPA` | fallback not executed after X-VLA solved the residual |

X-VLA source and load preflight:

- source clone: `C:\assets\repos\X-VLA`;
- source HEAD: `6bc2513f5f1cbec715cc668b414392a6cae5c671`;
- selected model: `2toINF/X-VLA-Libero`;
- model revision: `129e71460678b7236cee6fc9707f09d9fa0c3590`;
- model metadata: 15 files, `3.280` GiB;
- runner: `scripts/epoch5_xvla_libero10_task8_eval.py`;
- runner SHA-256:
  `1b0c6d43450a7c5320221308fe461f974c093d1fd2c0fd15d900a89b4f0bd077`.

Preflight artifacts:

| Artifact | Status | Key result |
|---|---:|---|
| `runs/xvla_prior/load_xvla_libero_20260717T1649KST/result.json` | pass | 879,482,456 parameters loaded on `cuda:0`; peak allocation `3,518,954,496` bytes |
| `runs/xvla_prior/action_smoke_xvla_libero_20260717T1654KST/result.json` | pass | finite dummy action tensor shape `[1, 30, 20]`; peak allocation `3,689,555,456` bytes |

Matched task-8 diagnostic:

- artifact:
  `runs/xvla_prior/diagnostic_xvla_task8_all_20260717T1705KST/result.json`;
- artifact SHA-256:
  `b13423bac4bb3f06c74611c42bf7b817cdfa586f1b5e4c3f05e9b73e270a5ef3`;
- task: `libero_10/task_8`, “put both moka pots on the stove”;
- reset identities: `20260716..20260723`;
- official initial-state indices: `5..12`;
- protocol: official X-VLA LIBERO path using `OffScreenRenderEnv`, absolute
  controller mode (`robot.controller.use_delta = False`), 10 settle steps,
  horizon 900, `domain_id=3`, and 10 denoising steps;
- training / optimizer / checkpoint / Ours design: false / false / false /
  false.

| Reset identity | Initial-state index | Success | Steps |
|---:|---:|---:|---:|
| 20260716 | 5 | true | 363 |
| 20260717 | 6 | true | 396 |
| 20260718 | 7 | true | 373 |
| 20260719 | 8 | true | 376 |
| 20260720 | 9 | true | 375 |
| 20260721 | 10 | true | 375 |
| 20260722 | 11 | true | 356 |
| 20260723 | 12 | true | 375 |

Summary: X-VLA completed 8/8 episodes with 8/8 successes and zero
infrastructure failures. It solved both original OpenVLA-OFT failure resets
(`20260721`, `20260722`) and both LightVLA failure resets (`20260716`,
`20260723`).

Interpretation: the current task-8 residual is solved by an executable official
third-pass prior. Therefore no Ours method should be designed, trained, or
reported on this residual. If Epoch 5 continues, it must select a new residual
condition against the latest executable official prior set, with X-VLA included
as a prior baseline.

## Post-X-VLA Residual Search Scan

Decision: `NO_NEW_XVLA_LIBERO10_SINGLE_IDENTITY_RESIDUAL_FOUND`.

After X-VLA solved the frozen task-8 residual, the runner was generalized for
fresh residual search. Current generalized runner SHA-256:
`262644e9a7d62834103496fd0fb7a740b5c359407af3ed1f8a647b6d155b0ff3`.

Scan scope:

- policy: `X-VLA-Libero`;
- suite: `libero_10`;
- tasks: `0..9`;
- reset identity: `20260724`;
- mapped official initial-state index: `13`;
- protocol: same X-VLA absolute-controller protocol as above;
- run directory:
  `runs/xvla_prior/failure_scan_libero10_identity20260724_20260717T1716KST`;
- benchmark claim: false; this was only failure mining for a new residual.

| Task | Success | Steps |
|---:|---:|---:|
| 0 | true | 260 |
| 1 | true | 234 |
| 2 | true | 259 |
| 3 | true | 229 |
| 4 | true | 224 |
| 5 | true | 182 |
| 6 | true | 283 |
| 7 | true | 250 |
| 8 | true | 378 |
| 9 | true | 261 |

Summary: 10/10 tasks completed and 10/10 succeeded with zero infrastructure
failures. This scan did not find a new X-VLA residual candidate at identity
`20260724`. If Epoch 5 continues, residual search must broaden to more
identities or additional suites before any Ours design is allowed.
