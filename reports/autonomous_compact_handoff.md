# Autonomous Compact Handoff

Updated: 2026-07-17 KST

## Current State

- Branch: `codex/epoch5-official-prior-first`
- Current epoch: 5
- Current cycle: 0
- Current stage: `epoch_5_prior_residual_headroom_complete`
- Current decision: `RESIDUAL_FOUND_PRIOR_POSITIVE_TASK_LEVEL_HEADROOM_POSITIVE`
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

Generate at most two Ours method candidates around the exact task-8 residual
limitation. Do not restart broad search. Each candidate must specify:

- the precise OpenVLA-OFT residual limitation;
- the actual prior mechanism being extended;
- one core new mechanism;
- supervision and deployment inputs;
- LoRA/QLoRA role as infrastructure only;
- key ablation;
- strongest simple alternative explanation;
- second-backbone path.

Select exactly one method before implementation.

## Prohibitions

- Do not design outside the task-8 residual.
- Do not generate three local method candidates.
- Do not reopen CAVM, CALA, RAR, MCI, CSPR, or governance-closed routes.
- Do not create generic cached-feature residuals, frozen-policy gates,
  history heads, verifiers, visual canonicalizers, memory lookups, or proxy-only
  prior methods.
- Do not claim INT4 OpenVLA-OFT is a full-precision reproduction.
