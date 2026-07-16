# DCCG-VLA Preregistration

Date: 2026-07-16 KST

Decision: `DCCG_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`

Method: `DCCG-VLA`, Demonstration-Calibrated Coherence Guidance for SmolVLA.

Proposal SHA-256:
`AE5DBB13F0B4C19E3DD8BD054433DCFBCC301F4C4293D7B98883D76CA4A1390E`

Prerequisite documents:

- proposal: `reports/dccg_vla/researcher_proposal.md`
- Reviewer B attack: `reports/dccg_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/dccg_vla/researcher_rebuttal.md`
- mathematical audit: `reports/dccg_vla/mathematical_mechanism_audit.md`

No DCCG implementation, validation search, rollout, simulator evaluation, or
confirmatory-test access has happened before this preregistration.

## Frozen Claim

DCCG tests whether a frozen-SmolVLA, demonstration-calibrated action-coherence
energy can guide flow action chunks more usefully than ACG-style generic
incoherence guidance and simple action smoothing while preserving exact Base
behavior at zero guidance.

The claim is not:

- official ACG reproduction unless official compatible ACG assets run locally;
- a new VLA backbone;
- ordinary LoRA as the method;
- generic smoothing or temporal ensembling;
- use of demonstration phase, reset identity, reward, success, done, object
  pose, future observation, or future expert action at inference;
- or a rescue of MHS, NICE, S2C, HEST, LCG, AFID, BRID, or any closed method.

## Evidence Partitions

`DISCOVERY / TRAINING`

- LIBERO demonstration actions and cached frozen SmolVLA Base chunks only;
- discovery tasks:
  - `libero_10/task_1`
  - `libero_10/task_3`
  - `libero_goal/task_1`
  - `libero_goal/task_3`
  - `libero_object/task_1`
  - `libero_spatial/task_1`
- demonstrations `0..29`;
- used for source checks, feature variance, robust statistics, Stage 0A
  gradients, ACG proxy diagnostics, and implementation debugging.

`VALIDATION`

- validation tasks:
  - `libero_10/task_5`
  - `libero_goal/task_5`
  - `libero_object/task_3`
  - `libero_spatial/task_3`
- demonstrations `30..39`;
- used for Stage 0B gates and bounded validation search only after Stage 0
  passes.

`CONFIRMATORY TEST`

- confirmatory tasks remain sealed:
  - `libero_10/task_7`
  - `libero_goal/task_7`
  - `libero_object/task_5`
  - `libero_spatial/task_5`
- no confirmatory reset identity, reward, success, done, object pose, future
  observation, policy action, failure, partial outcome, or threshold may be
  read before preregistered final configuration freeze;
- confirmatory outcomes may not retune DCCG.

## Fixed Mechanism

Use exactly the mathematical audit object:

- `H = 50`;
- `D = 7`;
- action chunk shape `[B, 50, 7]`;
- differentiable feature count `K_s = 10`;
- `lambda_grip = 1.0`;
- no deterministic-action KL;
- hard gripper counts are gates and reports, not unverified gradient sources;
- deployment bin uses only legal task family, legal queue/chunk index when
  available, stop-gradient current action-regime features, and legal
  nonprivileged action history when available.

Fixed Stage 0 constants:

- `epsilon_scale = 1e-6`;
- `tail_tau = 0.02`;
- `tau_pause = 0.005`;
- `epsilon_pause = 0.0025`;
- `tau_grip = 0.05`;
- `c_trans = 0.02`;
- `c_rot = 0.05`;
- `c_grip = 0.25`;
- default `alpha_u = 1.0` for the diagnostic one-step flow hook unless the
  implementation audit verifies a legal solver step index and freezes a
  schedule before rollout.

These constants may not change after Stage 0 begins.

## First Serious Comparison

The first serious comparison remains exactly:

1. `smolvla_base`
2. `acg_official_proxy`
3. `dccg_full`
4. `dccg_no_demo_calibration_ablation`
5. `action_smoothing_simple_killer`

Policy 2 must first attempt official ACG assets and code compatibility. If
exact execution is unavailable, it is a transparent local proxy and must
document every mismatch from published ACG. It may not be a smoothing-only
stand-in.

Policy 4 removes demonstration calibration while keeping the same DCCG feature
families, integration, action caps, and compute budget.

Policy 5 is the strongest simple smoothing baseline under the same action
shape, official postprocessor, action caps, and gripper-event constraints.

## Stage 0 Purpose

Stage 0 is a development-only source, data, mathematical, implementation, and
mechanism audit. It is not a closed-loop scientific result and not a paper
claim.

Stage 0 determines whether:

- legal demonstration action chunks and cached Base chunks are usable;
- DCCG features and bins are noncollapsed;
- `grad_A E(A,b)` is finite and nonzero where the gate activates;
- exact Base passthrough holds at `gamma = 0`;
- Base and ACG leave meaningful coherence/action-validity headroom;
- DCCG acts distinctly from ACG, no-demo-calibration, and smoothing;
- gripper events, normalized actions, and postprocessed actions remain valid.

## Stage 0 Sources And Minimum Rows

Stage 0A source smoke:

- at most two discovery tasks:
  - `libero_10/task_1`
  - `libero_goal/task_1`
- demonstrations `0..1`;
- at most `32` windows per demonstration.

Stage 0B development audit:

- discovery tasks and validation tasks listed above;
- minimum `384` discovery windows;
- minimum `128` validation windows;
- every validation task must contribute rows;
- no validation task may contribute more than `40%` of validation rows.

If these row counts cannot be produced without duplicate keys, split overlap,
or confirmatory reads, Stage 0 stops as `DCCG_STAGE_0_DATA_FAILURE`.

## Stage 0 Required Artifacts

Stage 0 must produce under `reports/dccg_vla/`:

- `stage_0_preflight.json`;
- `stage_0_manifest.json`;
- `stage_0_partial.json`;
- `stage_0_result.json`;
- `stage_0_result.md`;
- `stage_0_adjudication.md`;
- `stage_0_status.json`;
- `stage_0_heartbeat.json`;
- `stage_0_pid.txt`;
- `stage_0_exit_code.txt`;
- `stage_0_action_semantics.json`;
- `stage_0_official_prior_asset_check.json`;
- `stage_0_serializer_preflight.json`;
- stdout/stderr logs if launched detached.

## Stage 0 Required Metrics

Required metrics:

- planned and completed row counts;
- exception count;
- duplicate manifest keys, duplicate partial keys, missing keys, extra keys,
  and split-overlap keys;
- proposal hash match;
- no reward/success/done/object-pose/future-observation/confirmatory reads;
- action shape `[50,7]` and finite fraction;
- official action postprocessor validity;
- ACG official/proxy status and mismatch list;
- bin counts and per-feature variance;
- noncollapsed `E(A,b)` on discovery and validation;
- gate activation by task, bin, and gripper context;
- `grad_A E` finite/nonzero rate and gradient norm by action group;
- exact Base passthrough at `gamma = 0`;
- DCCG versus Base, ACG, no-demo-calibration, and smoothing distinction;
- normalized and postprocessed action deltas from Base;
- hard gripper transition count, reversal count, sign-change timing, and
  gripper delta statistics;
- clean-retention proxy;
- action validity rate.

## Stage 0 Pass Gates

All must pass:

- proposal hash matches;
- no privileged or confirmatory input access;
- manifest and partial row keys are unique and complete;
- split overlap is zero;
- features and bins are noncollapsed;
- validation gate activation is neither all-zero nor all-one;
- `grad_A E` is finite and nonzero when active;
- exact Base passthrough at `gamma = 0`;
- DCCG differs from Base, ACG, no-demo-calibration, and smoothing on diagnostic
  incoherent chunks;
- gripper-event hard gate passes;
- normalized and postprocessed action validity pass;
- Base and ACG leave meaningful headroom.

Stop classes:

- `DCCG_STAGE_0_DATA_FAILURE`;
- `DCCG_STAGE_0_NO_HEADROOM`;
- `DCCG_STAGE_0_IMPLEMENTATION_FAILURE`;
- `DCCG_STAGE_0_DESIGN_FAILURE`;
- `DCCG_STAGE_0_PASS_TO_VALIDATION_SEARCH`.

## Bounded Validation Search

Allowed only after Stage 0 passes.

Maximum six configurations:

- `gamma in {0.05, 0.10, 0.20}`;
- gate quantile `theta_b in {0.90, 0.95}`.

No feature set, binning method, solver schedule, clipping cap, task split,
identity split, comparator, threshold, or stop rule may be searched outside
this budget.

Validation score:

`S_val = 0.40 * closed_loop_success_or_proxy
       + 0.20 * clean_retention
       + 0.15 * coherence_separation
       + 0.15 * action_validity
       + 0.10 * acg_and_smoothing_margin`.

Ties break by clean retention, then lower activation rate, then smaller
`gamma`, then lower gate quantile.

## Confirmatory Gate

Confirmatory Stage A/B can begin only after:

- Stage 0 passes;
- bounded validation search selects one frozen configuration;
- checkpoint/config artifacts are saved;
- policy list, metrics, thresholds, task/reset identities, and manifests are
  frozen;
- no confirmatory result has been read.

Confirmatory outcomes may not retune DCCG. A major redesign after confirmatory
access is a new method cycle.

## Immediate Next Stage

Proceed to executable prototype protocol before implementation, validation
search, rollout, or confirmatory-test access.
