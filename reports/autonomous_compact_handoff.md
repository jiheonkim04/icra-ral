# Autonomous Compact Handoff

Updated: 2026-07-17 KST

## Current State

- Branch: `codex/epoch5-official-prior-first`
- Current epoch: 5
- Current cycle: 0
- Current stage: `epoch_5_r2r_oft_data_audit_passed`
- Current decision: `R2R_OFT_DATA_HEALTH_PASS_PRETRAINING_READY`
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

## Next Action

Freeze bounded `R2R-OFT` training configuration, resumability, and
validation-selection rules before any optimizer-step training.

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

Audit must report phase counts, demo coverage, split integrity, action ranges,
chunk validity, and whether phase labels are decodable without privileged
deployment inputs.

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

QLoRA gradient smoke result:

- artifact:
  `runs/openvla_oft_int4/epoch5_r2r_oft_qlora_gradient_smoke.json`;
- pass: true;
- scope: one-batch backward-pass feasibility only;
- LoRA rank/alpha: 4 / 8;
- phase-weight lambda: 2.0;
- weighted loss: 0.99609375;
- trainable LoRA parameters: 13,853,536;
- nonzero-gradient parameter tensors: 425;
- gradient global norm: 4.082890925442449;
- CUDA allocated/peak: 5,917.196 / 8,121.43 MiB;
- training run / optimizer step / checkpoint written: false / false / false.

## Prohibitions

- Do not design outside the task-8 residual.
- Do not generate three local method candidates.
- Do not reopen CAVM, CALA, RAR, MCI, CSPR, or governance-closed routes.
- Do not create generic cached-feature residuals, frozen-policy gates,
  history heads, verifiers, visual canonicalizers, memory lookups, or proxy-only
  prior methods.
- Do not claim INT4 OpenVLA-OFT is a full-precision reproduction.
