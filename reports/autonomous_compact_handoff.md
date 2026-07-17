# Autonomous Compact Handoff

Updated: 2026-07-17 KST

## Current State

- Branch: `codex/epoch5-official-prior-first`
- Current epoch: 5
- Current cycle: 0
- Current stage: `epoch_5_second_pass_fallback_preflight_complete`
- Current decision: `SECOND_PASS_PRIOR_FALLBACKS_BLOCKED_AFTER_LIGHTVLA_NO_GO`
- Previous method: `MCI-VLA`
- Previous decision: `MCI_STAGE_0_IMPLEMENTATION_FAILURE`
- MCI rescue/retune: prohibited and not performed
- Cycle 39 ordinary local-method search: superseded by strategy reset, not a scientific kill

## Audit Anchor

- Full audit: `reports/autonomous_research_full_history_audit.md`
- Audit commit: `b0ecb6ea5f6eba2953b5bd842883c0474d634dff`
- Refreshed audit commit on this branch: `541c82259db2b37adfe3894d2776235302e1c536`
- Audit totals: 73 routes, 47 formal methods, 31 trained/checkpointed routes, 17 Stage A, 10 Stage B, 0 GO, 0 second-backbone Ours, 0 formal Ours official-prior wins.

## Epoch 5 Artifacts

- Ecosystem selection: `reports/epoch5_prior_ecosystem_selection.md`
- Reproduction plan: `reports/epoch5_prior_reproduction_plan.md`
- Reproduction result: `reports/epoch5_prior_reproduction_result.md`
- Reproduction result JSON: `reports/epoch5_prior_reproduction_result.json`

## Selected Prior Ecosystem

Selected: OpenVLA-OFT on LIBERO.

Why: public primary paper, MIT official code, official LIBERO checkpoints,
local checkout, local 15G checkpoint, and existing validated INT4 hard-slice
run.

Quantization caveat: INT4 is a local prior diagnostic, not a full-precision
OpenVLA-OFT reproduction.

## Completed Prior Diagnostics

Recovered hard slice:

- OpenVLA-OFT INT4: 20/20.
- SmolVLA frozen-base exact-init: 11/20.
- Interpretation: prior positive but saturated; unusable for Ours.

Preregistered residual diagnostic `epoch5_libero10_residual_v1`:

- tasks: `libero_10/task_8`, `libero_10/task_9`;
- reset identities: `20260716..20260723` -> official initial-state indices
  `5..12`;
- SmolVLA frozen-base exact-init: 7/16;
- Quantized OpenVLA-OFT INT4: 14/16;
- infrastructure failures: 0/32;
- task 8: Base 3/8, OpenVLA-OFT 6/8;
- task 9: Base 4/8, OpenVLA-OFT 8/8;
- OpenVLA residual failures: task 8 reset `20260721` index 10 and reset
  `20260722` index 11.

Headroom diagnostic:

- artifact:
  `runs/openvla_oft_int4/epoch5_libero10_residual_expert_headroom_task8_demo10.json`;
- type: task-level HDF5 expert exact-init teacher replay;
- task: `libero_10/task_8`;
- demo: `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5::demo_10`;
- result: success, reward 1.0, done/success at step 377;
- proof: `after_set_state_l2_to_hdf5_init = 0.0`;
- caveat: not same benchmark reset; local HDF5 demo init-state hashes do not
  match residual benchmark initial-state hashes.

## Current Scientific Meaning

The required Base/Prior residual structure is present:

Base fails -> OpenVLA-OFT improves -> residual remains on task 8.

The condition is not classified as too severe because task-level expert
headroom is positive. However, same-reset upper-bound evidence is unavailable,
so Ours claims must stay narrow and explicitly carry that caveat.

## Selected Method and Completed Checks

Selected method:

- `R2R-OFT`: Residual Remaining-object Reweighted OFT.
- Prior extended: OpenVLA-OFT two-image + proprio + continuous L1 8x7 action
  chunks, LoRA fine-tuning.
- Exact residual: task 8 second-object completion after one moka pot is on/near
  the stove; failures are reset `20260721` index 10 and reset `20260722` index
  11.
- Core mechanism: phase-balanced imitation objective that upweights successful
  expert chunks where exactly one moka pot is already on/near the stove and a
  remaining moka pot still needs pickup/placement.
- Deployment inputs: same OpenVLA-OFT RGB, wrist RGB, proprio, instruction.
- Training-only labels may use HDF5/simulator state; privileged state is not an
  inference input.
- Key ablation: same LoRA/QLoRA scaffold with uniform task-8 weighting.
- Simple alternative: shorter OpenVLA-OFT action-chunk requery without
  training.
- Second-backbone path: same phase-weighted sampler/objective on SmolVLA
  adapter/QLoRA path.

Audit result:

- artifact:
  `runs/openvla_oft_int4/epoch5_r2r_oft_pretraining_data_audit.json`;
- pass: true;
- demos: 50;
- steps/chunks: 20,794 / 20,444;
- train/validation demos: 40 / 10;
- train one-pot chunks: 9,152;
- validation one-pot chunks: 2,332;
- action range: [-1.0, 1.0], 7D finite;
- residual init-state hash overlap: 0.

R2R-OFT execution summary:

- data audit and one-batch QLoRA gradient smoke passed;
- frozen spec:
  `runs/openvla_oft_int4/epoch5_r2r_oft_training_spec_v1.json`;
- primary and uniform-ablation arms both completed 64 optimizer steps;
- fixed offline validation at steps 16/32/64 failed the action-delta gate;
- closed-loop Ours rollout was disallowed;
- simple 4-step OpenVLA-OFT requery control scored 5/8, not selected.

## Second-Pass LightVLA Prior

Initial exact-three status: OpenVLA route executed but no-go; OpenPI/pi0.5
resource-blocked after source/env/checkpoint setup; PCD/PCD-LeRobot
source-inspected but dependency/checkpoint/multi-GPU blocked.

Second exact-three selected LightVLA first:

- source/checkpoint: `C:\assets\repos\LightVLA` HEAD `a4680fda...`;
  `TTJiang/LightVLA-libero-10` revision `d40628fe...`; download run exited `0`;
- caveat: official loader copied source files/backups into the checkpoint
  directory, so local file count is now 49 and the directory is not pristine;
- load artifact:
  `runs/lightvla_prior/load_lightvla_libero10_20260717T1528KST/result_4bit.json`;
- load result: 4-bit pass, `OpenVLAForActionPrediction` +
  `L1RegressionActionHead`, peak CUDA about 4.99 GB.

LightVLA task-8 matched diagnostic:

- artifact:
  `runs/lightvla_prior/diagnostic_lightvla_libero10_task8_all_20260717T1535KST/result.json`;
- status: completed, 8/8 episodes, no infrastructure failure;
- SmolVLA frozen base: 3/8, failures `20260716`, `20260717`, `20260721`,
  `20260722`, `20260723`;
- OpenVLA-OFT INT4: 6/8, failures `20260721`, `20260722`;
- LightVLA 4-bit: 6/8, failures `20260716`, `20260723`;
- LightVLA solved both OpenVLA failures, while OpenVLA solved both LightVLA
  failures; oracle OpenVLA-or-LightVLA would be 8/8.

## First Method After LightVLA

Selected method: `CR-LightVLA`, collision-rescue token pruning.

- runner: `scripts/epoch5_lightvla_collision_rescue_eval.py`;
- artifact:
  `runs/lightvla_prior/cr_lightvla_task8_all_20260717T1600KST/result.json`;
- rule: keep LightVLA first-choice unique tokens; for dynamic-query collisions,
  also keep collided queries' second-choice token;
- training / optimizer / checkpoint: false / false / false;
- result: 6/8, failures `20260718`, `20260723`;
- fixed one LightVLA failure: `20260716`;
- preserved LightVLA wins on OpenVLA failures: `20260721`, `20260722`;
- regression vs LightVLA/OpenVLA: `20260718`;
- still failed: `20260723`;
- decision: `CR_LIGHTVLA_STAGE0_NO_PROTOTYPE_GO`.

ATCD teacher-signal audit:

- runner: `scripts/epoch5_atcd_teacher_signal_audit.py`;
- result:
  `runs/lightvla_prior/atcd_teacher_signal_20260717T1620KST/atcd_teacher_signal_result_v2.json`;
- scope: 24 fixed task-8 HDF5 validation chunks, normalized 8x7 action chunks;
- OpenVLA-OFT wins 9/24, LightVLA wins 15/24;
- mean L1: OpenVLA 0.4338486312578122, LightVLA 0.41920601141949493,
  oracle 0.4083502360930045;
- oracle gain: absolute 0.010855775326490402, relative 0.025896039252230916;
- phase-1 oracle absolute gain: 0.013157747685909271;
- no training / optimizer / checkpoint / rollout;
- decision: `ATCD_TEACHER_SIGNAL_NOT_ENOUGH` because relative gain < 0.03.

Second-pass fallback prior preflight:

- RIPT-VLA: official source cloned at `C:\assets\repos\ript-vla`, HEAD
  `440990e8864e12e4578b490ff6359e4f2c49ae3e`; HF repo
  `tanshh97/RIPT_VLA` revision `57532f4...`, 32 files / 6.180 GiB metadata;
  source import smoke passed in the OpenVLA runtime; no checkpoint download,
  training, or rollout.
- RIPT blocker: OpenVLA-OFT RIPT assets cover LIBERO Goal/Spatial/Object/Long,
  not the current `libero_10/task_8` both-moka residual; new OpenVLA-OFT RIPT is
  interactive RL with official 4-GPU recommendation. QueST checkpoints are
  lighter but not an exact both-moka prior.
- VLA-GSE: official source cloned at `C:\assets\repos\VLA-GSE`, HEAD
  `200cdc245880322f2bef7b24ec506063a0f35e8c`; no trained checkpoint present; no
  checkpoint download, training, or rollout.
- VLA-GSE blocker: Qwen3-VL + LeRobot-format LIBERO PEFT framework with
  8-GPU/80k-step reference training and two-process policy-server evaluation.
- decision: `SECOND_PASS_PRIOR_FALLBACKS_BLOCKED_AFTER_LIGHTVLA_NO_GO`.

Next: start a third exact-three official-prior ecosystem selection pass. Do not
train or roll out ATCD; do not claim RIPT-VLA or VLA-GSE prior results from
source-only/resource-blocked preflights; do not retune on tested task-8
identities.

## Prohibitions

- Do not design outside the task-8 residual.
- Do not generate three local method candidates.
- Do not reopen CAVM, CALA, RAR, MCI, CSPR, or governance-closed routes.
- Do not create generic cached-feature residuals, gates, history heads,
  verifiers, memory lookups, or proxy-only prior methods.
- Do not claim INT4 OpenVLA-OFT is a full-precision reproduction.
