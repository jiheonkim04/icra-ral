# S2C-VLA Preregistration

Date: 2026-07-16 KST

Decision: `S2C_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`

Proposal: `reports/s2c_vla/researcher_proposal.md`

Proposal SHA-256:
`399A3960F9FF9AFA8EDA7C3F743A95C3FD4DC711644C2398630F1E68486DC5B3`

Reviewer attack: `reports/s2c_vla/reviewer_attack.md`

Researcher rebuttal: `reports/s2c_vla/researcher_rebuttal.md`

Mathematical audit: `reports/s2c_vla/mathematical_mechanism_audit.md`

No S2C implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this preregistration.

## Frozen Claim

S2C tests whether a Base-preserving, identity-initialized overlap edit layer can
reduce SmolVLA adjacent-chunk boundary inconsistency while leaving unselected
overlap cells and all future-zone cells exactly at Base.

The method is not a generic smoothness penalty, not LoRA as the scientific
mechanism, and not a renamed ChunkFlow adapter.

## Frozen Mechanism

Use the mathematical audit values:

- `H = 50`;
- replanning stride `s = 10`;
- overlap edit length `K = 10`;
- action dimension `D = 7`;
- editable zone `0..K-1`;
- future passthrough zone `K..H-1`;
- deterministic bridge target from current Base head and previous committed
  tail;
- learned effective edit mask with exact Base initialization;
- group caps `0.02` translation, `0.05` rotation, `0.25` gripper;
- gripper event default cap `0`;
- no deterministic-action KL.

Deployment previous tail is always the unexecuted overlap slice of the previous
committed Base or S2C chunk. Expert future actions are labels or diagnostics
only and never inference inputs.

## Evidence Partitions

Discovery and validation are development-only partitions. Confirmatory testing
is not used in Stage 0, preregistration, prototype protocol, implementation,
or bounded validation search.

Development task set:

- `libero_spatial/task_3`;
- `libero_object/task_3`;
- `libero_goal/task_5`;
- `libero_10/task_5`.

Discovery demonstrations: demo IDs `0..7`.

Validation demonstrations: demo IDs `8..9`.

Confirmatory identities: not enumerated or touched in Stage 0.

No train/validation/test identity may overlap.

## First Serious Comparison

The first serious comparison remains exactly:

1. `smolvla_base`
2. `chunkflow_overlap_proxy` or official ChunkFlow if installed and verified
3. `s2c_full`
4. `s2c_no_learned_overlap_mask_ablation`
5. `standard_lora`

The local ChunkFlow proxy must be strong and transparent: frozen/editable/future
zones, deterministic overlap blending, first-order continuity, second-order
continuity, and official action-validity semantics where feasible.

## Stage 0 Purpose

Stage 0 is a development-only audit. It is not a closed-loop scientific result
and not a paper claim.

Stage 0 determines whether S2C has enough adjacent-chunk boundary headroom and
safe policy-integration evidence to justify bounded validation.

## Stage 0 Required Artifacts

Stage 0 must produce:

- `reports/s2c_vla/stage_0_preflight.json`;
- `reports/s2c_vla/stage_0_manifest.json`;
- `reports/s2c_vla/stage_0_partial.json`;
- `reports/s2c_vla/stage_0_result.json`;
- `reports/s2c_vla/stage_0_result.md`;
- `reports/s2c_vla/stage_0_adjudication.md`;
- `reports/s2c_vla/stage_0_status.json`;
- `reports/s2c_vla/stage_0_heartbeat.json`;
- `reports/s2c_vla/stage_0_pid.txt`;
- `reports/s2c_vla/stage_0_exit_code.txt`;
- `reports/s2c_vla/stage_0_action_semantics.json`;
- `reports/s2c_vla/stage_0_official_prior_asset_check.json`;
- `reports/s2c_vla/stage_0_serializer_preflight.json`.

## Stage 0 Metrics

Required metrics:

- planned and completed row counts;
- exception count;
- duplicate/missing/extra/split-overlap key counts;
- proposal hash match;
- no reward/success/done/confirmatory reads;
- Base boundary headroom mean and p75;
- ChunkFlow residual headroom;
- identity reload max absolute error;
- effective mask positive fraction;
- future-zone drift max absolute error;
- official action validity;
- S2C versus ChunkFlow boundary-retention relative improvement;
- S2C versus no-mask ablation boundary-retention relative improvement;
- standard LoRA boundary-retention score;
- gripper event destruction count;
- objective term magnitudes and gradient norm ratios.

## Stage 0 Pass Gates

All must pass:

- Base boundary Huber disagreement mean `>= 0.0025` or p75 `>= 0.005`;
- ChunkFlow proxy leaves at least `2%` residual boundary or clean-retention
  headroom;
- identity reload max absolute error `<= 1e-6`;
- effective mask positive fraction in `[0.02, 0.80]`;
- future-zone drift max absolute error `0`;
- action validity `1.0`;
- S2C full beats ChunkFlow proxy by at least `2%` on the preregistered
  boundary-retention score;
- S2C full beats no-mask ablation by at least `5%`;
- standard LoRA does not explain the same score;
- gripper event destruction count `0`;
- no duplicate, missing, extra, or split-overlap keys;
- no exceptions.

## Stage 0 Stop Classes

Stage 0 must stop as one of:

- `S2C_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `S2C_STAGE_0_NO_ADJACENT_BOUNDARY_HEADROOM`;
- `S2C_STAGE_0_DESIGN_FAILURE`;
- `S2C_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- `S2C_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Stop before validation search for collapsed masks, equivalence to ChunkFlow,
equivalence to no-mask ablation, global smoothing, future-zone edits, gripper
event destruction, invalid action semantics, identity reload failure, expert
future-tail inference, or any reward/success/done/confirmatory read.

## Bounded Validation Search If Stage 0 Passes

If and only if Stage 0 passes, a bounded validation-only search may use at most:

- `6` total configurations;
- `2` random seeds for lightweight fits;
- `2` architecture choices;
- `3` values for one critical coefficient;
- no combinatorial grid.

Candidate factors may include edit-mask hidden size, mask sparsity coefficient,
bridge cap scale, or event-preservation coefficient. The selected
configuration must be chosen only on validation score combining boundary
retention, clean retention, mechanism activation, action validity, and overhead.

## Worker And Resume Rules

Before any expensive command, inspect PID, heartbeat, status, partial, result,
log, and exit-code artifacts. If an S2C worker is alive, monitor it only. If it
completed, adjudicate the result and do not rerun. If it died with valid partial
rows, resume only missing keys:

`(split, task_suite, task_id, demo_id, window_start, stride, previous_policy_source, policy)`

Completed rows must not repeat. Duplicate-key and manifest checks must run
before accepting the result.

## Current Authorization

This preregistration authorizes prototype protocol drafting next. It does not
authorize implementation, validation search, training, rollout, or
confirmatory testing until the prototype protocol is frozen.
