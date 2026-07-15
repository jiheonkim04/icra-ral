# PCAV-VLA Preregistration

Date: 2026-07-15 KST

Decision: `PCAV_PREREGISTRATION_FROZEN_STAGE_0A_PENDING`

Proposal hash:
`E8B23C755C6D4E450FD193101CC0B15F88AAFE20E137A0F86830ED6D421E12AA`.

## Frozen Claim

PCAV tests whether hard successful-support eligibility followed by
action-conditioned progress preference and Base-relative abstention improves
frozen VLA candidate selection over Base, support-only TACO, progress-only
selection, and matched standard LoRA.

No claim is made for support estimation, progress estimation, candidate
sampling, or abstention individually.

## Authoritative Documents

In order:

1. `reports/current_research_governance.md`
2. `reports/epoch_4_cycle_18_candidate_generation.md`
3. `reports/pcav_vla/researcher_proposal.md`
4. `reports/pcav_vla/reviewer_attack.md`
5. `reports/pcav_vla/researcher_rebuttal.md`
6. `reports/pcav_vla/mathematical_mechanism_audit.md`
7. this preregistration
8. `reports/pcav_vla/prototype_protocol.md`

Later documents may narrow execution or fix implementation bugs. They may not
change the method, evidence partitions, tasks, first policy list, objective,
search budget, thresholds after use, or confirmatory interpretation.

## Backbone, Data, And Tasks

Backbone:

`/mnt/c/assets/checkpoints/smolvla_libero`

Target source:

`/mnt/c/assets/data/libero/libero_90`

Tasks:

1. `KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf`
2. `LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray`
3. `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy`

The official 40-task checkpoint identities and these target identities must
have normalized intersection zero.

The FAMR endpoint checkpoint is forbidden. PCAV starts from the untouched Base
checkpoint and independently created heads only.

## Evidence Partitions

Target demonstration episodes per task:

- `DISCOVERY`: `0..29`;
- `VALIDATION`: `30..39`;
- `CONFIRMATORY_OFFLINE`: `40..49`.

Stage 0 may read only discovery episodes. Validation is opened only after Stage
0 mechanism and capacity pass. Confirmatory-offline is opened exactly once
after one configuration, checkpoint, policy list, metrics, thresholds, and
closed-loop manifest are frozen.

Closed-loop reset identities are generated and materialized before the first
closed-loop validation action. Discovery, validation, confirmatory, and clean
retention resets must be disjoint. A
`(partition, policy, task, reset_identity)` key must be unique.

No result from one partition selects or retunes an earlier partition.

## Frozen Stage 0A Row Construction

Valid row population:

- task is one of the three frozen tasks;
- episode is in discovery `0..29`;
- frame and frame `+10` exist before terminal padding;
- current/future images, state, and first 10 actions are finite;
- frame is assigned to early `[0,1/3)`, middle `[1/3,2/3)`, or late
  `[2/3,1)` using valid preterminal position.

For each task and phase, sort rows by
`SHA256(proposal_hash|task|episode|frame)`.

Initial `24` rows use per-task quotas:

- early: `3`;
- middle: `2`;
- late: `3`.

The single unresolved expansion to `96` uses per-task quotas:

- early: `11`;
- middle: `10`;
- late: `11`.

The 24 rows are a prefix subset of the 96-row manifest. Row identity is
`(task_identity, episode, frame)` and duplicates are forbidden.

## Frozen Candidate Construction

Stage 0A candidate count is `4`. For row identity `r` and candidate index `i`,
the Gaussian noise seed is the first 63 bits of:

`SHA256(proposal_hash|stage0a|r|i)`.

All noise tensors have native shape `1 x 50 x 32`. Candidate zero is direct
Base. All candidates use the same raw observation, processor, postprocessor,
solver steps, action dimensions, and horizon. Only noise differs.

Each row persists:

- candidate index and seed;
- noise hash;
- native policy `50 x 7` output metadata and `50 x 32` flow-noise metadata;
- postprocessed `50 x 7` action chunk;
- chunk hash;
- first-10 translation, rotation, and gripper summaries;
- direct Base identity error;
- candidate-versus-Base deltas;
- candidate-versus-demonstration errors;
- validity gates.

## Stage 0A Required Audits

### Provenance And Mapping

- proposal hash matches;
- all three HDF5 and BDDL files exist and hash;
- each task has 50 demonstrations;
- all 150 demonstrations end with source success;
- task identity overlap with checkpoint tasks is zero;
- official raw-to-policy image, state, and action mappings pass;
- discovery/validation/confirmatory episode and frame hashes have zero overlap;
- confirmatory observations decoded and actions computed both equal zero.

### Label And Sequence Health

- episode length min/max/mean/std by task;
- terminal padding and repeat frequencies;
- action/state finite fractions exactly `1.0`;
- normalized-time labels have nonzero variance;
- early/middle/late quotas are complete;
- no duplicate row, frame, action-prefix, or partition identity caused by the
  manifest construction.

### Candidate Observability

- checkpoint persists and disk reloads;
- candidate zero equals direct Base with max error `0.0`;
- all four candidates are generated for every completed row;
- more than half the rows have at least two unique action chunk hashes;
- median nonzero pairwise postprocessed action L2 exceeds `1e-4`;
- candidate and direct Base action semantics are identical;
- exception count is zero.

### Action Validity

For every candidate, compute:

- finite fraction `=1.0`;
- absolute maximum `<=1.25`;
- outside-`[-1,1]` fraction `<= direct Base + 0.01`;
- p99 exceedance `<= direct Base + 0.02`.

Candidate zero must pass. A failing alternative is recorded and made ineligible;
it is not silently clipped. More than half the rows must retain at least one
valid unique alternative, and diversity/headroom use valid alternatives only.

Simulator action-space acceptance is checked before the first rollout, not
required to decode Stage 0A offline rows.

### Candidate-Oracle Headroom

Use the standardized first-10-step error in the mathematical audit.

Pass requires:

- at least `25%` of rows contain an alternative at least `5%` better than Base;
- median oracle relative reduction over improvable rows at least `5%`.

If no row contains a strictly better unique alternative, stop as
`PCAV_STAGE_0A_NO_USABLE_HEADROOM`.

If some rows improve but the pass threshold is not met, expand once to the
frozen 96-row manifest. No other row count, seed, noise distribution, task, or
threshold is allowed.

## Stage 0A Decisions

`PCAV_STAGE_0A_PASS_STAGE_0B_ALLOWED` only when all provenance, mapping, label,
Base action-validity, valid-alternative coverage, candidate, duplicate,
exception, and headroom gates pass.

`PCAV_STAGE_0A_UNRESOLVED_EXPANSION_REQUIRED` only for the defined 24-to-96
headroom case. Expansion is automatic and uses only missing row keys.

`PCAV_STAGE_0A_NO_USABLE_HEADROOM` for exact no-headroom or failed expanded
headroom with valid candidates and mapping.

`PCAV_STAGE_0A_DESIGN_FAILURE_CANDIDATES_COLLAPSED` for exact or near-exact
candidate equivalence under verified distinct noise and correct implementation.

`PCAV_STAGE_0A_IMPLEMENTATION_OR_DATA_FAILURE` for missing/corrupt data,
mapping mismatch, reload failure, invalid action semantics, duplicates,
exceptions, partial corruption, or failed validity caused by the implementation.

No closed-loop scientific kill is possible in Stage 0A.

## Stage 0B Minimum Head Capacity

Only after Stage 0A pass:

1. extract frozen visual, language, action-expert, state, action, and future
   features from discovery only;
2. create deterministic projection matrices and CFN targets;
3. fit CFN, consequence, and progress heads for exactly 20 optimizer steps;
4. persist optimizer, model, projection, normalization, and row states;
5. audit term magnitudes and per-term gradients;
6. verify checkpoint reload and initial Base identity;
7. report noncollapse and acting differences among Base, support-only,
   progress-only, and full PCAV.

Stage 0B establishes implementation capacity only. A weak 20-step result is
`UNDERPOWERED_OR_UNRESOLVED`, not a scientific kill.

## Full Development Training

After Stage 0B implementation pass:

- CFN: 2,000 optimizer steps;
- consequence/progress: 2,000 joint optimizer steps;
- batch 16;
- AdamW learning rate `3e-4`, weight decay `1e-4`;
- seed `1801`;
- no architecture, learning-rate, horizon, feature, or objective search;
- atomic optimizer-step resume;
- all negative and partial results retained.

Required before validation:

- finite nonzero gradients;
- loss improvement over fixed subsets;
- consequence beats persistence and shuffled-action diagnostics;
- progress beats strongest trivial baseline on held-out discovery episodes;
- CFN separates demonstration-near and perturbed support;
- heads persist and reload exactly;
- Base hash unchanged;
- action validity retained;
- no confirmatory decode.

## Bounded Validation Search

At most six configurations:

1. support percentile `50`, margin `0.00`, `N=4`;
2. support percentile `50`, margin `0.05`, `N=4`;
3. support percentile `70`, margin `0.00`, `N=4`;
4. support percentile `70`, margin `0.05`, `N=4`;
5. selected support/margin with `N=8`;
6. selected configuration's second lightweight seed.

Selection score, fixed before search:

`0.50 * paired validation success or closest closed-loop proxy`
`+ 0.20 * clean retention`
`+ 0.15 * mechanism activation quality`
`+ 0.10 * action validity`
`+ 0.05 * normalized compute score`.

Activation quality is zero for exact equivalence and penalizes intervention
above `50%`; it does not maximize intervention. Ties choose lower intervention,
then lower compute, then smaller `N`.

Save every attempted configuration, metric, checkpoint identity, and negative
result. Freeze one configuration before confirmatory evaluation.

## First Serious Comparison

Exactly five policies in one paired manifest:

1. `smolvla_base`
2. `taco_support_proxy`
3. `pcav_full`
4. `pcav_progress_only`
5. `standard_lora_new_task`

No sixth policy enters Stage A without a new concrete alternative explanation
that could change the decision and costs less than proceeding.

## Stage A

Approximately `10` paired episodes per policy. Purpose:

- detect catastrophic harm;
- detect obvious closest-prior or ablation dominance;
- verify real-rollout support/progress activation;
- estimate direction.

Stage A may permanently stop only for mechanism invalidity, no usable headroom,
catastrophic degradation, clear prior/ablation dominance, or exact trivial
equivalence after adequate capacity. Small differences advance to Stage B.

## Stage B

At least `40` paired episodes per key policy. Report:

- success counts and task-balanced success;
- paired wins/losses/ties;
- paired bootstrap confidence interval;
- effect size and failure-rate reduction;
- per-task breakdown;
- support and progress activation;
- clean retention;
- latency and memory outside resource-contention intervals.

One expansion to `80` is allowed only when the frozen Stage B result is
genuinely unresolved.

## Prototype GO

Required:

- `pcav_full` beats Base;
- `pcav_full` beats `taco_support_proxy`;
- `pcav_full` beats `pcav_progress_only`;
- standard LoRA does not explain the gain;
- clean success and action validity are retained;
- support and progress mechanism evidence agree with the claim;
- novelty remains defensible.

After GO, immediately verify Quantized OpenVLA-OFT INT4, add one second
claim-specific condition, measure compute, and prepare figure/table-ready
evidence.

## Result Classification

- `PROTOTYPE_GO`
- `GENUINE_METHOD_KILL`
- `SIMPLE_BASELINE_EXPLAINS_METHOD`
- `KEY_COMPONENT_NOT_USEFUL`
- `NO_USABLE_HEADROOM`
- `DATA_OR_SUPERVISION_FAILURE`
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`
- `UNDERPOWERED_OR_UNRESOLVED`

Only the first four are scientific formulation decisions. The others inform a
new method cycle and do not establish broad negative claims.

## Durable Execution

Every nontrivial job writes PID, heartbeat, status, atomic partial JSON, final
JSON, log, and exit code. A living worker is monitored, never duplicated. A
stale heartbeat requires PID and log verification. A dead worker resumes only
missing keys after partial JSON validation.

Before accepting results:

- completed count equals planned count;
- exception count is zero;
- duplicate keys are zero;
- manifest coverage is exact;
- task/reset/action semantics are unchanged;
- simulator is synchronous;
- checkpoint and result hashes verify.

## Resource-Contention Quarantine

The existing Windows gaming/Efficiency Mode interval remains recorded in
`reports/resource_contention_intervals.json`. No overlapping latency,
throughput, wall-clock efficiency, or utilization measurement enters final
paper evidence. Valid synchronous success rows may remain only under the
exception, identity, semantics, timeout, duplicate, and manifest conditions in
the governance.

## Current Boundary

Only Stage 0A implementation, tests, audit, and execution are authorized. Stage
0B and every later stage remain blocked until the prior stage's frozen decision
allows them.
