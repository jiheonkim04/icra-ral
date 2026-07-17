# Epoch 5 Prior Reproduction Result

Selected prior ecosystem: OpenVLA-OFT on LIBERO.

## Result

Current decision:
`TASK6_MPR_XVLA_SELECTED_AFTER_SECOND_PRIOR_RESIDUAL_SURVIVED`.

Historical task-8 method decision: `R2R_OFT_OFFLINE_SELECTION_NOT_PASSED`.

Epoch 5 completed the selected-prior-first diagnostic sequence before Ours
design, then generated exactly two task-8 Ours candidates and selected
`R2R-OFT`. Bounded optimizer-step training happened only after its training
spec was frozen. No full-BF16 OpenVLA-OFT attempt or closed-loop task-8 Ours
evaluation happened.
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

After X-VLA solved the task-8 residual, a fresh task-1 residual was found and
matched against SmolVLA base. The shared failure `20260727` now has positive
task-level expert headroom, again with same-reset HDF5 evidence unavailable.
The task1 basket data audit then passed, and exactly two narrow Ours candidates
were generated. `BR-XVLA` was selected, its no-training two-arm spec was frozen,
the tiny X-VLA-format data-adapter smoke passed, the one-batch no-optimizer
gradient smoke passed, and the bounded two-arm training/offline-validation gate
passed. The subsequent frozen closed-loop residual screen on identity
`20260727` did not pass: the same-run X-VLA prior failed, the `BR-XVLA` primary
also failed, and the uniform-weight ablation succeeded.

After archiving BR-XVLA as a validation no-pass, a detached X-VLA prior-only
scan over all LIBERO-10 tasks at reset identity `20260725` completed cleanly.
It found X-VLA failures on task 1 and task 6. Task 1 is already a known
X-VLA-regression identity where SmolVLA base succeeded, so it is not an active
Ours target. Task 6 is the fresh candidate from this scan. A matched
Base/Prior window now confirms usable residual structure: X-VLA improves over
SmolVLA base (`6/8` versus `3/8`) while leaving shared residual failures at
`20260725` and `20260731`. Both shared failures have positive task-level HDF5
expert replay headroom, but no same-reset HDF5 demo init-state match was
available. A task6 spatial data audit then passed, and the required
Quantized OpenVLA-OFT INT4 second-prior screen did not solve either shared
residual. Exactly two task6 candidates were generated; `MPR-XVLA` was selected
as the first narrow candidate. No task6 optimizer step, checkpoint, training, or
closed-loop Ours evaluation has happened.

## Validation Commands

Focused OpenVLA artifact validation:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests\test_openvla_oft_int4_gate.py -q
```

Observed after manifest-control patches: `5 passed`.

Post-BR-XVLA scan report validation:

- JSON parse: pass via
  `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m json.tool`;
- compact handoff: 134 lines, under the 250-line cap;
- scan launcher syntax: pass via WSL `bash -n`;
- X-VLA runner py-compile: pass via the official WSL environment;
- focused scan tests: none found;
- `git diff --check`: pass with LF/CRLF warnings only;
- `scripts/99_tree_check.ps1`: pass via one-shot PowerShell
  execution-policy bypass.

Task6 candidate-design validation:

- JSON parse: pass via
  `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m json.tool`;
- compact handoff: 210 lines, under the 250-line cap;
- py-compile: pass for `tca_map\openvla_oft_int4_gate.py`,
  `tca_map\xvla_task6\data_audit.py`,
  `tests\test_openvla_oft_int4_gate.py`, and
  `tests\test_xvla_task6_data_audit.py`;
- focused pytest: `8 passed`;
- `git diff --check`: pass with LF/CRLF warnings only;
- `scripts/99_tree_check.ps1`: pass via one-shot PowerShell
  execution-policy bypass.

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

## New X-VLA Residual Candidate: LIBERO-10 Task 1

Decision: `X_VLA_LIBERO10_TASK1_RESIDUAL_FOUND_MATCHED_BASE_PENDING`.

The next broadened scan tested `libero_10/task_1` across identities
`20260724..20260731`. This produced a clean official-prior residual candidate.

- policy: `X-VLA-Libero`;
- task: `libero_10/task_1`, “put both the cream cheese box and the butter in
  the basket”;
- reset identities: `20260724..20260731`;
- official initial-state indices: `13..20`;
- artifact:
  `runs/xvla_prior/diagnostic_xvla_libero10_task1_id20260724_20260731_20260717T1729KST/result.json`;
- artifact SHA-256:
  `279807f3b729032e55a921bda03d512cda243b6fbd1e4db76dd5b97384fee77d`;
- runner SHA-256:
  `262644e9a7d62834103496fd0fb7a740b5c359407af3ed1f8a647b6d155b0ff3`;
- training / optimizer / checkpoint / Ours design: false / false / false /
  false.

| Reset identity | Initial-state index | X-VLA success | Steps |
|---:|---:|---:|---:|
| 20260724 | 13 | true | 231 |
| 20260725 | 14 | false | 900 |
| 20260726 | 15 | true | 243 |
| 20260727 | 16 | false | 900 |
| 20260728 | 17 | true | 249 |
| 20260729 | 18 | true | 250 |
| 20260730 | 19 | true | 243 |
| 20260731 | 20 | true | 240 |

Summary: X-VLA completed 8/8 episodes with 6/8 successes and zero
infrastructure failures. Failures: `20260725` and `20260727`.

Interpretation: this is a valid official-prior residual candidate, but it is
not yet an Ours target. The next required step is a matched Base/Prior
diagnostic on this same task/reset window, followed by residual headroom
verification before any method design.

## Matched Base/Prior Diagnostic: LIBERO-10 Task 1

Decision: `TASK1_MATCHED_BASE_PRIOR_RESIDUAL_CONFIRMED_HEADROOM_PENDING`.

The matched Base run used the same task/reset window as X-VLA. The first attempt
in the OpenVLA-compatible runtime failed before rollout because that environment
pins `transformers 4.40.1`, which lacks `AutoModelForImageTextToText`. The
rerun in the direct official SmolVLA environment completed cleanly.

| Policy | Artifact | Completed | Successes | Infrastructure failures | Failures |
|---|---|---:|---:|---:|---|
| X-VLA-Libero | `runs/xvla_prior/diagnostic_xvla_libero10_task1_id20260724_20260731_20260717T1729KST/result.json` | 8 | 6 | 0 | `20260725`, `20260727` |
| SmolVLA frozen base | `runs/xvla_prior/diagnostic_smolvla_base_libero10_task1_id20260724_20260731_officialenv_20260717T1739KST/result.json` | 8 | 3 | 0 | `20260724`, `20260727`, `20260728`, `20260729`, `20260730` |

Base artifact SHA-256:
`bf20b5433c889e9be61ef8b6e0701ca495e4a27df1df10070f8a897a56791e83`.

Identity-level comparison:

| Identity | X-VLA | SmolVLA base | Classification |
|---:|---:|---:|---|
| 20260724 | true | false | X-VLA-only success |
| 20260725 | false | true | Base-only success / X-VLA regression |
| 20260726 | true | true | Both success |
| 20260727 | false | false | Shared residual failure |
| 20260728 | true | false | X-VLA-only success |
| 20260729 | true | false | X-VLA-only success |
| 20260730 | true | false | X-VLA-only success |
| 20260731 | true | true | Both success |

Interpretation: X-VLA is a valid stronger official prior overall, but it leaves
two residual failures. The cleanest headroom target is the shared failure
`20260727`; `20260725` must be handled separately because the base policy
already succeeds there. Do not design Ours until residual headroom is verified.

## Task-1 Headroom Diagnostic

Decision:
`TASK1_TASK_LEVEL_EXPERT_HEADROOM_POSITIVE_SAME_RESET_UNAVAILABLE`.

Artifact:
`runs/xvla_prior/diagnostic_task1_expert_headroom_20260727_20260717T180914KST/result.json`.

Script: `scripts/epoch5_task1_expert_headroom.py`.

The diagnostic targeted the matched Base/Prior shared failure `20260727`, which
maps to LIBERO-10 task-1 initial-state index `16`. The benchmark residual
initial-state SHA-256 was confirmed as
`bb8073f96294281b7008501d0b6ebdec3668f90448421c5937b58f57c1b8c5e2`.

| Check | Result |
|---|---|
| Task HDF5 demos scanned | 50 |
| Same-reset HDF5 init-state matches | 0 |
| Selected replay demo | `demo_48` |
| Selection reason | nearest HDF5 demo init-state by L2; no hash match |
| Selected demo L2 to benchmark residual init | 1.309200905 |
| Exact replay success | true |
| Exact replay reward / done / success step | 246 / 246 / 246 |
| Exact replay reward sum | 1.0 |
| `after_set_state_l2_to_selected_hdf5_init` | 0.0 |
| Zero-action exact-init control succeeded | false |
| Default-reset expert replay succeeded | false |

Interpretation: the task is not unrecoverable under expert actions, so the
shared task-1 residual is allowed to proceed to narrow Ours design. The caveat
is material: there is no same-reset HDF5 expert upper bound for `20260727`.
The base-only success / X-VLA regression identity `20260725` remains out of
scope for Ours unless separately governed.

## Task-1 Basket Data-Health Audit

Status: `PASS`.

Decision: `TASK1_BASKET_DATA_HEALTH_PASS_PREDESIGN_READY`.

Artifact:
`runs/xvla_prior/diagnostic_task1_basket_data_audit_20260727_20260717T181823KST/result.json`.

Module: `tca_map/xvla_task1/data_audit.py`.

| Audit item | Result |
|---|---:|
| HDF5 demos | 50 |
| Total steps | 13,021 |
| Total 8-step chunks | 12,671 |
| Train one-target-remaining chunks | 4,607 |
| Validation one-target-remaining chunks | 1,079 |
| Train demos with one-target-remaining chunks | 40 |
| Validation demos with one-target-remaining chunks | 10 |
| Basket XY threshold | 0.08 |
| Final target-object XY max distance to basket | 0.05548356006913492 |
| Initial target-object XY min distance to basket | 0.18621333529485404 |
| Residual init-state hash overlap | 0 |

Verified state layout:

- `cream_cheese_1_pos = state[17:20]`;
- `butter_1_pos = state[52:55]`;
- `basket_1_pos = state[59:62]`.

The phase labels use HDF5 simulator state for training/validation labels only;
inference inputs remain RGB, wrist RGB, proprioception, and instruction.

## Task-1 Ours Candidate Generation

Status: `EXACTLY_TWO_CANDIDATES_GENERATED_ONE_SELECTED_NO_TRAINING`.

Decision: `BR_XVLA_SELECTED_TRAINING_SPEC_PENDING`.

Detailed artifact: `reports/epoch5_task1_ours_candidate_design.md`.

| Candidate | Contribution type | Core mechanism | Score | Decision |
|---|---|---|---:|---|
| `BR-XVLA`: Basket-Remaining Reweighted X-VLA | `PRIOR_EXTENSION` | LoRA/QLoRA-adapt X-VLA-Libero with phase-balanced imitation, upweighting chunks where exactly one target object is in/near the basket and the other remains out. | 86/100 | SELECTED |
| `OCB-XVLA`: Object-Contrast Basket X-VLA | `PRIOR_EXTENSION` | Balance cream-cheese-first and butter-first object-order supervision. | 73/100 | NOT SELECTED |

Bounded optimizer-step training and checkpoint writes happened for `BR-XVLA`
only inside the frozen two-arm training gate. Closed-loop Ours evaluation then
happened only inside the frozen one-identity residual-manifest screen below.
The result is a validation no-pass, and BR-XVLA must not be retuned from it.

## `BR-XVLA` Training Spec Freeze

Status: `FROZEN_PASS_NO_TRAINING`.

Decision: `BR_XVLA_TRAINING_SPEC_FROZEN_PREOPT_GATES_COMPLETE`.

Artifact: `runs/xvla_prior/epoch5_br_xvla_training_spec_v1.json`.

Module: `tca_map/xvla_task1/training_spec.py`.

The interface audit found official X-VLA PEFT fine-tuning code in
`C:\assets\repos\X-VLA\peft_train.py`. The frozen spec keeps the first
adaptation attempt to exactly two arms:

| Arm | Role | Lambda | Max optimizer steps |
|---|---|---:|---:|
| `br_xvla_rank8_lambda2_lr1e4_steps64` | primary selected method | 2.0 | 64 |
| `uniform_xvla_rank8_lambda0_lr1e4_steps64` | uniform-weight ablation | 0.0 | 64 |

At freeze time: training false, optimizer step false, checkpoint false.

Important interface caveat: raw LIBERO HDF5 is not a direct official X-VLA
training input. The required tiny X-VLA-format data-adapter smoke and one-batch
no-optimizer gradient smoke have both passed before any optimizer step.

## `BR-XVLA` Data-Adapter Smoke

Status: `PASS`.

Decision: `BR_XVLA_DATA_ADAPTER_SMOKE_PASS`.

Artifact:
`runs/xvla_prior/br_xvla_data_adapter_smoke_20260717T183355KST/result.json`.

Module: `tca_map/xvla_task1/data_adapter_smoke.py`.

The smoke converted `demo_0` and `demo_48` into X-VLA's official LIBERO HDF5
contract, then instantiated X-VLA's `InfiniteDataReader` and pulled one sample.

| Reader output | Shape / value |
|---|---|
| action | `30 x 20` |
| proprio | `20` |
| image input | `3 x 3 x 224 x 224` |
| image mask | `3` |
| domain id dtype | `torch.int64` |
| local `mmengine.fileio` shim used | true |

No training, model load, backward pass, optimizer step, checkpoint, or closed-
loop Ours evaluation happened in the data-adapter smoke. This gate is now
superseded by the passing one-batch no-optimizer gradient smoke below.

## `BR-XVLA` One-Batch No-Optimizer Gradient Smoke

Status: `PASS`.

Artifact:
`runs/xvla_prior/br_xvla_gradient_smoke_20260717T190919KST/result.json`.
SHA-256:
`d661576639c86fd4657abe983968b8aa3969e934d8de082de3337cb56e7802cd`.

This was a mechanism/feasibility smoke only, not a training run. It loaded the
cached `2toINF/X-VLA-Libero` checkpoint at revision
`129e71460678b7236cee6fc9707f09d9fa0c3590`, attached the official PEFT LoRA
configuration from the frozen spec, consumed the task1 X-VLA-format adapter,
computed the basket-remaining weighted supervised loss on a one-target clip,
and ran exactly one backward pass.

Runtime repairs were limited to the X-VLA environment boundary: `timm==1.0.12`
was installed from X-VLA's own requirements, import-only shims were used for
serving-only dependencies `fastapi`, `uvicorn`, and `json_numpy`, the existing
local `mmengine.fileio` reader shim was used, and two Transformers 4.57.6
compatibility patches kept the checked-out X-VLA Florence2 code on the
conservative path (`_supports_sdpa=False`, missing-`lm_head`
`get_output_embeddings` returns `None`).

| Smoke item | Result |
|---|---:|
| Exit code | 0 |
| Local files only | true |
| Weighted loss | 7.872903823852539 |
| Phase weight | 3.0 |
| Trainable PEFT parameters | 11,868,760 |
| Gradient tensors finite / total | 537 / 537 |
| Nonzero-gradient tensors | 271 |
| Gradient global norm | 1239.7495099257394 |
| CUDA max allocated | 5,260.354 MiB |
| Model loaded / PEFT attached | true / true |
| Forward / backward happened | true / true |
| Optimizer created / step | false / false |
| Checkpoint / training / closed-loop Ours | false / false / false |

Gate result: `BR_XVLA_GRADIENT_SMOKE_PASS`.

## `BR-XVLA` Frozen Two-Arm Training / Offline Validation Gate

Status: `COMPLETE_OFFLINE_PASS`.

Gate decision: `BR_XVLA_OFFLINE_PASS_BEATS_ABLATION`.

Launch manifest:
`runs/xvla_prior/epoch5_br_xvla_training/gate_launch_manifest.json`,
SHA-256 `abe8cafc194e42bfec7462f9f2825d2158dd6e9a53fd8efa8d20fdc65631eebc`.

Gate result:
`runs/xvla_prior/epoch5_br_xvla_training/gate_result.json`,
SHA-256 `3af1afc6a152aae8d8fafe5dfc43a19fe9ff2174236d2f263067b1f3cace2a76`.

Offline validation result:
`runs/xvla_prior/epoch5_br_xvla_offline_validation_step0064.json`,
SHA-256 `119723e76e769589442fd0e04d4c26e2fe1b9fc4d825ab47ce7abd6e56ec747a`.

The detached gate ran from commit
`06d03d8147df53e54837605e079427eb4f66adfa`, completed in
150.81154718000005 seconds, and wrote exit code `0`. A launcher bookkeeping bug
initially wrote a newline-only exit-code file; the run result itself completed,
the artifact now records `0`, and the launcher is fixed to write future exit
codes with `printf`.

| Arm | Result artifact | SHA-256 | Steps | Checkpoint | Last loss | Peak CUDA MiB |
|---|---|---|---:|---|---:|---:|
| `br_xvla_rank8_lambda2_lr1e4_steps64` | `runs/xvla_prior/epoch5_br_xvla_training/br_xvla_rank8_lambda2_lr1e4_steps64/result.json` | `e6f8c641c4f8c931ff769bf6da11b7cfc9cdc62e90bab985896d6f9870d3ee05` | 64 | `checkpoints/step_0064/adapter` | 0.7570973634719849 | 5,350.398 |
| `uniform_xvla_rank8_lambda0_lr1e4_steps64` | `runs/xvla_prior/epoch5_br_xvla_training/uniform_xvla_rank8_lambda0_lr1e4_steps64/result.json` | `da9492c600a84a6742f3b10e2d414c0b910da0f8434cc0b41211c76f15c1c4f0` | 64 | `checkpoints/step_0064/adapter` | 0.7570974826812744 | 5,351.867 |

Offline validation used 24 fixed validation chunks with phase counts
`{0: 6, 1: 12, 2: 6}`, `denoise_steps=10`, `local_files_only=true`, and no
closed-loop rollout.

| Policy | Mean loss | Phase-0 loss | Phase-1 loss | Phase-2 loss |
|---|---:|---:|---:|---:|
| X-VLA prior base | 3.495260993639628 | 4.755821585655212 | 3.107213238875071 | 3.0107959111531577 |
| `BR-XVLA` primary | 1.258300895492236 | 2.165877252817154 | 1.0268368770678837 | 0.8136525750160217 |
| Uniform ablation | 1.2583009228110313 | 2.1658771137396493 | 1.0268369267384212 | 0.8136527240276337 |

The predefined offline screen passes because the primary beats the uniform
ablation numerically on phase-1 loss and does not degrade the clean phase
relative to the prior. The phase-1 margin versus uniform is only
`4.967053751942781e-8`, so this should be treated as a narrow offline validation
pass, not as robust evidence that the BR weighting mechanism beats uniform
adaptation. Both trained adapters are much lower-loss than the unadapted X-VLA
prior on the fixed offline chunks.

## `BR-XVLA` Frozen Closed-Loop Residual-Manifest Screen

Status: `COMPLETE_NOT_PASSED`.

Decision: `BR_XVLA_CLOSED_LOOP_RESIDUAL_NOT_PASSED`.

This was a one-identity validation screen, not a broad confirmatory experiment.
The frozen manifest evaluated `libero_10/task_1`, reset identity `20260727`,
initial-state index `16`, with denoise steps 10 and horizon 900. Policies were
same-run X-VLA prior, `BR-XVLA` primary, and uniform-weight ablation.

Launch manifest:
`runs/xvla_prior/epoch5_br_xvla_closed_loop_residual_20260727/closed_loop_launch_manifest.json`,
SHA-256 `0996517829549c07ac64009ce8e4be3457da5248f64100cfcf3ad449d359f6fd`.

Frozen manifest:
`runs/xvla_prior/epoch5_br_xvla_closed_loop_residual_20260727/closed_loop_manifest.json`,
SHA-256 `ea222a6014e2cda6a8f7428bdf2d0f0105e1773e0f7a0c6ba3ce5bb74f01dc63`.

Result:
`runs/xvla_prior/epoch5_br_xvla_closed_loop_residual_20260727/closed_loop_result.json`,
SHA-256 `472904b03472c8b1017aad2080c57e49c0b1064816b430670051330dd970b64f`.

The run used commit `b91a49d8f66253ac85815fdde366a41824397232`, completed its
Python result in 126.246165868 seconds, used local files only, and performed no
training, optimizer step, or checkpoint write.

| Policy | Adapter | Success | Steps | Final reward | Chunks | Action range |
|---|---|---:|---:|---:|---:|---|
| X-VLA prior base | none | false | 900 | 0.0 | 30 | [-0.3528921604156494, 1.011316180229187] |
| `BR-XVLA` primary | `br_xvla_rank8_lambda2_lr1e4_steps64/checkpoints/step_0064/adapter` | false | 900 | 0.0 | 30 | [-0.2577773928642273, 1.2225242853164673] |
| Uniform ablation | `uniform_xvla_rank8_lambda0_lr1e4_steps64/checkpoints/step_0064/adapter` | true | 479 | 1.0 | 16 | [-0.2553746700286865, 1.2284687757492065] |

Interpretation: the residual failure was reproduced for the same-run X-VLA
prior, but the selected BR-XVLA weighting did not fix it. Worse for the
mechanism-specific claim, the uniform-weight ablation solved the same identity.
This archives the frozen BR-XVLA configuration as a validation no-pass. Do not
retune BR-XVLA from this result.

Launcher caveat: the Python result artifact is complete, but the detached
wrapper wrote a newline-only `closed_loop_exit_code.txt` and then printed
`exit: : numeric argument required`. The cause was unescaped dollar variables
in the Windows-to-WSL `bash -lc` wrapper. The launcher code is patched after
this run to escape `$?` and `$status` for future exit-code writes; this caveat
does not change the closed-loop policy outcomes above.

## Post-BR-XVLA X-VLA Prior Residual-Mining Scan

Status: `COMPLETE_FAILURES_FOUND_BASE_MATCH_PENDING`.

Decision:
`X_VLA_POST_BRXVLA_RESIDUAL_SCAN_FOUND_FAILURES_BASE_MATCH_PENDING`.

This scan was official-prior residual mining only. It did not train, step an
optimizer, write a checkpoint, design Ours, or run closed-loop Ours evaluation.
It only executed the unmodified X-VLA prior across LIBERO-10 tasks at reset
identity `20260725`.

Repaired completed run:
`runs/xvla_prior/failure_scan_libero10_identity20260725_post_brxvla_repaired2_20260717T2022KST`.

| Artifact | SHA-256 |
|---|---|
| Scan manifest | `5adbc60144dde3f49a1c8cd82f5bcdc2f82c184447d5fb799843a0fbeef3eacc` |
| Scan summary | `c2ff073b74efb5e9af9db0bc6254aaa9dd735aaaf0c6635fcf93dfe35d07a16a` |
| Launcher script | `abf19670014523ccc704d34a632e035784c0dc25810a7cffd9bfe6fe3f562059` |
| Runner script | `2dbd93bb062913c5e07f211f49669a224b3a1f6777ef7c30c8570cd4c200edf6` |

Run metadata:

- commit: `5835ef3bafad1027e9e4ed6dcf5943383d2a9714`;
- policy: `X-VLA-Libero`;
- suite: `libero_10`;
- reset identity: `20260725`, initial-state index `14`;
- horizon: `900`, settle steps `10`, denoise steps `10`;
- exit code: `0`;
- finished: `2026-07-17T20:22:49+09:00`;
- completed tasks: `10/10`;
- successes: `8/10`;
- infrastructure failures: `0`.

| Task | Success | Steps | Interpretation |
|---:|---:|---:|---|
| 0 | true | 273 | prior succeeds |
| 1 | false | 900 | known X-VLA regression; SmolVLA base previously succeeded on this identity |
| 2 | true | 238 | prior succeeds |
| 3 | true | 221 | prior succeeds |
| 4 | true | 219 | prior succeeds |
| 5 | true | 180 | prior succeeds |
| 6 | false | 900 | fresh prior failure; matched Base/Prior and headroom pending |
| 7 | true | 270 | prior succeeds |
| 8 | true | 368 | prior succeeds |
| 9 | true | 244 | prior succeeds |

Invalid infrastructure attempts preserved for audit:

| Run | Classification | Cause |
|---|---|---|
| `runs/xvla_prior/failure_scan_libero10_identity20260725_post_brxvla_20260717T2010KST` | `INVALID_INFRASTRUCTURE_BLOCKED` | X-VLA serving-only dependency import reached `fastapi` before import shims. |
| `runs/xvla_prior/failure_scan_libero10_identity20260725_post_brxvla_repaired_20260717T2018KST` | `INVALID_INFRASTRUCTURE_BLOCKED` | Script-by-path execution lacked repo root on `sys.path`, causing `No module named 'tca_map'`. |

## Task-6 Matched Base/Prior Residual Diagnostic

Status: `COMPLETE_RESIDUAL_STRUCTURE_CONFIRMED`.

Decision at this gate:
`TASK6_MATCHED_BASE_PRIOR_RESIDUAL_CONFIRMED_HEADROOM_PENDING_AT_RUN_TIME`.

Task: `libero_10/task_6`, "put the white mug on the plate and put the chocolate
pudding to the right of the plate".

Window: reset identities `20260724..20260731`, initial-state indices `13..20`.
This was not training and not an Ours evaluation.

| Policy | Completed | Successes | Failures | Infrastructure failures | Artifact | SHA-256 |
|---|---:|---:|---|---:|---|---|
| X-VLA prior | 8/8 | 6/8 | `20260725`, `20260731` | 0 | `runs/xvla_prior/diagnostic_xvla_libero10_task6_id20260724_20260731_20260717T2043KST/result.json` | `d18356bf1a18e4f2053596142d9af13983ffc1ed0ccc74fa525ad4d802ac25aa` |
| SmolVLA frozen base | 8/8 | 3/8 | `20260724`, `20260725`, `20260727`, `20260730`, `20260731` | 0 | `runs/xvla_prior/diagnostic_smolvla_base_libero10_task6_id20260724_20260731_officialenv_20260717T2047KST/result.json` | `749fbc0f25f075902de9e2172c602e99cde020d4b4be735accedbb80c45556c8` |

SmolVLA base manifest:
`runs/xvla_prior/diagnostic_smolvla_base_libero10_task6_id20260724_20260731_officialenv_20260717T2047KST/manifest.json`,
SHA-256 `19733d8a5490350beba7d4444810e73c90af48c47e21471dca2b5257e0874f89`.

Per-identity interpretation:

| Reset identity | Base | X-VLA | Interpretation |
|---:|---:|---:|---|
| `20260724` | fail | success | X-VLA-only success |
| `20260725` | fail | fail | shared residual failure |
| `20260726` | success | success | both succeed |
| `20260727` | fail | success | X-VLA-only success |
| `20260728` | success | success | both succeed |
| `20260729` | success | success | both succeed |
| `20260730` | fail | success | X-VLA-only success |
| `20260731` | fail | fail | shared residual failure |

Interpretation: task 6 now satisfies the official-prior-first structure:
Base has meaningful failures, the official prior improves the condition, and a
measurable residual gap remains. This is not an Ours result.

Launcher caveat: an earlier `20260717T2040KST` attempt used a WSL background
child that was torn down when the shell session exited. It produced no
`result.json`, no exit code, and no simulator rows, so it is classified as
`INVALID_LAUNCHER_NO_RESULT_NO_ROLLOUT`, not a scientific run.

## Task-6 Expert Headroom Diagnostics

Status: `COMPLETE_TASK_LEVEL_HEADROOM_POSITIVE_SAME_RESET_UNAVAILABLE`.

Decision:
`TASK6_MATCHED_BASE_PRIOR_RESIDUAL_CONFIRMED_HEADROOM_POSITIVE_SAME_RESET_UNAVAILABLE`.

Script: `scripts/epoch5_expert_headroom.py`, SHA-256
`7339d16a9b70665064b437eb7d007d81f6bc99246f0fe28a46b2e33ee321b8b0`.

The diagnostic checked the two shared task-6 residual failures. For both
identities, the benchmark residual init-state SHA matched the expected value,
but no HDF5 demo had the same init-state SHA. The nearest-demo exact expert
replay succeeded for both; zero action and default-reset expert replay did not.

| Reset identity | Residual init SHA-256 | Same-reset HDF5 matches | Selected demo | L2 to residual init | Exact expert replay | First success | Zero action | Default-reset replay | Artifact SHA-256 |
|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| `20260725` | `47a0a589a343a89446f23421036719e5afd5bfd6fb1fc975c9a3546d867c3c82` | 0 | `demo_24` | 0.286668634 | success | 235 | false | false | `68b61e5802f6d672d44ab58ee26170cad724fce6c8cc4870065e2b4b2dc7cccd` |
| `20260731` | `4f63fc206bad261b4721178ee1859e47c3111c119b2ef428e8d296ae7c0069e3` | 0 | `demo_6` | 0.283710624 | success | 217 | false | false | `5dac493d0443bb1237b69ca0c0d5c69b2a00259c697de39fe2364550b9d9f49d` |

Interpretation: this avoids a no-headroom/no-recoverability stop for task 6,
but it is weaker than a same-reset oracle. It authorizes narrow task-6 residual
characterization and at most two candidates around the exact X-VLA task-6
residual; it does not authorize a broad method search or BR-XVLA retuning.

## Task-6 Spatial Data Audit

Status: `COMPLETE_DATA_HEALTH_PASS`.

Decision: `TASK6_SPATIAL_DATA_HEALTH_PASS_PREDESIGN_READY`.

Artifact:
`runs/xvla_prior/diagnostic_task6_spatial_data_audit_20260717T2115KST/result.json`,
SHA-256 `71178809c5290ae6b4083e34fdf3aa49a4b259bb42f26b2561628acaeb3800fd`.

Source:
`tca_map/xvla_task6/data_audit.py`, SHA-256
`0accb6887839178fca565b18d8a691ed78fa2b515b90f8cf5d986085e1b779c8`.

The audit is CPU/HDF5-only: no model load, no training, no optimizer step, no
checkpoint write, and no simulator rollout.

| Split | Demos | Chunks | Phase-0 chunks | Mug done / pudding remaining | Phase-2 chunks |
|---|---:|---:|---:|---:|---:|
| Train | 40 | 9,929 | 3,803 | 5,518 | 608 |
| Validation | 10 | 2,477 | 950 | 1,372 | 155 |

Dataset checks:

- 50 demos, 12,756 steps, 12,406 chunks.
- Actions are finite 7D values in the LIBERO controller range `[-1, 1]`.
- All 50 demos have terminal reward and done signals.
- All 50 demos complete the mug-on-plate subgoal before the pudding-right
  subgoal.
- The red mug remains off-plate as a distractor.
- Residual init-state overlap is zero for hashes
  `47a0a589a343a89446f23421036719e5afd5bfd6fb1fc975c9a3546d867c3c82` and
  `4f63fc206bad261b4721178ee1859e47c3111c119b2ef428e8d296ae7c0069e3`.
- Privileged simulator state is used only to create training labels; inference
  remains X-VLA RGB/proprio/instruction only.

## Task-6 Quantized OpenVLA-OFT INT4 Second-Prior Screen

Status: `COMPLETE_SECOND_PRIOR_RESIDUAL_SURVIVED`.

Decision: `TASK6_NOT_SOLVED_BY_OPENVLA_OFT_INT4`.

The valid screen used the OpenVLA-compatible runtime
`/home/jiheon/venvs/openvla-oft-int4-rtx5080/bin/python` after two invalid
wrong-runtime attempts.

Valid artifact:
`runs/openvla_oft_int4/diagnostic_task6_residual_openvla_int4_20260725_20260731_openvlaenv_20260717T2114KST/result.json`,
SHA-256 `c897000b299d2d8fd356bb467a574971dd8d11843c0d06ecdd7698d765cd233b`.

| Policy | Completed | Successes | Infrastructure failures | Elapsed seconds |
|---|---:|---:|---:|---:|
| Quantized OpenVLA-OFT INT4 | 2/2 | 0/2 | 0 | 208.769 |

| Reset identity | Initial-state index | Success | Steps | Final reward | Video |
|---:|---:|---:|---:|---:|---|
| `20260725` | 14 | false | 530 | 0.0 | `./rollouts/2026_07_17/2026_07_17-21_14_55--openvla_oft--episode=110000--success=False--task=put_the_white_mug_on_the_plate_and_put_the_chocola.mp4` |
| `20260731` | 20 | false | 530 | 0.0 | `./rollouts/2026_07_17/2026_07_17-21_14_55--openvla_oft--episode=110001--success=False--task=put_the_white_mug_on_the_plate_and_put_the_chocola.mp4` |

Invalid attempts preserved for audit:

| Run | Classification | Cause |
|---|---|---|
| `runs/openvla_oft_int4/diagnostic_task6_residual_openvla_int4_20260725_20260731_20260717T2130KST` | `INVALID_RUNTIME_ENV_MISSING_OPENVLA_DEPENDENCY` | Wrong WSL runtime lacked `json_numpy`; no `result.json`. |
| `runs/openvla_oft_int4/diagnostic_task6_residual_openvla_int4_20260725_20260731_repaired_20260717T2135KST` | `INVALID_RUNTIME_ENV_MISSING_OPENVLA_DEPENDENCY` | Same wrong runtime then lacked TensorFlow; no `result.json`. |

Interpretation: task6 is not already fully solved by Quantized OpenVLA-OFT INT4
on the two shared X-VLA/SmolVLA Base residual identities. Candidate design is
therefore authorized by the second-prior screen.

## Task-6 Ours Candidate Design

Status: `MPR_XVLA_CANDIDATE_SELECTED_PRETRAINING_SPEC_PENDING`.

Decision: `TASK6_MPR_XVLA_SELECTED_AFTER_SECOND_PRIOR_RESIDUAL_SURVIVED`.

Detailed artifact: `reports/epoch5_task6_ours_candidate_design.md`.

Exactly two candidates were considered around the exact measured task6
residual:

| Candidate | Contribution type | Core mechanism | Score | Decision |
|---|---|---|---:|---|
| `MPR-XVLA`: Mug-placed / Pudding-right Reweighted X-VLA | `PRIOR_EXTENSION` | LoRA/QLoRA-adapt X-VLA-Libero with phase-balanced imitation, upweighting chunks where the white mug is already on the plate and the chocolate pudding still needs the right-of-plate relation. | 88/100 | SELECTED |
| `PRC-XVLA`: Pudding-Right Contrast X-VLA | `PRIOR_EXTENSION` | Add relation/distractor contrast around pudding-right-of-plate versus red-mug/plate distractor geometry during adaptation. | 74/100 | NOT SELECTED |

`MPR-XVLA` is selected because it is the narrowest mechanism supported by the
data audit and residual gates. The first frozen training spec, if created, must
include exactly two arms: primary `MPR-XVLA` and a uniform-weight X-VLA
LoRA/QLoRA ablation. The uniform ablation is mandatory because the task1
BR-XVLA screen showed that uniform adaptation can explain an apparent residual
fix.

No task6 optimizer step, checkpoint write, training run, or closed-loop Ours
evaluation has happened.
