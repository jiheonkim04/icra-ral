# MHS-VLA Preregistration

Date: 2026-07-16 KST

Decision: `MHS_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`

Method: `MHS-VLA`, Mamba History State for Base-preserving SmolVLA.

Proposal SHA-256:
`BBDF67AE3EC4BD9D025707A8BB3A5008BAB5EB5C691D02D44516157802A87BF3`

Prerequisite documents:

- proposal: `reports/mhs_vla/researcher_proposal.md`
- Reviewer B attack: `reports/mhs_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/mhs_vla/researcher_rebuttal.md`
- mathematical audit: `reports/mhs_vla/mathematical_mechanism_audit.md`

No MHS implementation, training, validation search, rollout, simulator access,
or confirmatory-test access has happened before this preregistration.

## Frozen Claim

MHS tests whether a frozen-SmolVLA, deployment-observable recurrent history
state can selectively gate bounded residual corrections around the frozen Base
action chunk while preserving exact Base behavior by default.

The claim is not:

- broad novelty for Mamba imitation learning;
- broad novelty for memory in VLAs;
- official MTIL reproduction unless compatible official code is installed and
  verified before confirmatory testing;
- a new VLA backbone;
- ordinary LoRA as the method;
- generic current-frame residual imitation;
- full-policy replacement;
- direct use of demonstration actions at inference;
- or a rescue of BRID or any previous closed method.

LoRA may only serve as low-compute implementation infrastructure for the
history encoder, residual head, or matched controls. Removing the word LoRA
must not remove the MHS scientific mechanism.

## Evidence Partitions

`DISCOVERY / TRAINING`

- legal LIBERO demonstrations only;
- demonstrations `0..7` for each fixed development task;
- used for Base chunk decoding, history-window construction, nearest-neighbor
  summary statistics, `m_i` and `z_i` construction, small-fit gradient checks,
  MTIL proxy diagnostics, standard-LoRA diagnostics, and implementation
  debugging;
- may not include confirmatory reset identities, labels, failures, outcomes,
  rewards, success flags, done flags, object poses, or simulator state.

`VALIDATION`

- legal LIBERO demonstrations only;
- demonstrations `8..9` for each fixed development task;
- used for Stage 0 data/mechanism gates and, only after Stage 0 pass, bounded
  validation search and final configuration selection;
- may not use confirmatory outcomes.

`CONFIRMATORY TEST`

- untouched until method, configuration, policy list, ablation, tasks, reset
  identities, metrics, thresholds, manifests, and checkpoints are frozen;
- no confirmatory task/reset identities, rewards, success flags, done flags,
  object poses, future observations, policy actions, failed rollouts, or
  partial outcomes may be read during Stage 0 or validation search;
- confirmatory outcomes may not retune MHS.

## Fixed Development Sources

Use these four source task families:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

Discovery/training demonstrations: `0..7`.

Validation demonstrations: `8..9`.

History length: `L = 8`.

Action horizon: `K = 50`.

Action dimension: `D = 7`.

Minimum Stage 0 windows:

- at least `512` discovery windows;
- at least `128` validation windows;
- every task must contribute validation rows;
- no task may contribute more than `40%` of the Stage 0 validation subset;
- every scored row must have at least `8` legal previous observation,
  proprioception, and action history steps, or must be masked and counted as
  unavailable.

If these row counts cannot be produced without duplicate keys, split overlap,
or confirmatory identity reads, Stage 0 must stop as
`MHS_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

## Frozen Mechanism

Use exactly the mathematical audit variables and constants:

- Base chunk `B in R^{N x 50 x 7}`;
- aligned demonstration chunk `E in R^{N x 50 x 7}`;
- Base residual `R = E - B`;
- history window `X_hist in R^{N x 8 x d_x}`;
- current feature `X_0 in R^{N x d_x}`;
- recurrent state `h in R^{N x 128}`;
- residual proposal `P in R^{N x 50 x 7}`;
- capped residual `Delta in R^{N x 50 x 7}`;
- scalar gate `g in R^{N x 1 x 1}`;
- output `A = B + g * Delta`;
- Huber delta `0.01`;
- Base residual threshold `tau_base = 0.02`;
- history usefulness margin `tau_hist = 0.01`;
- identity tolerance `1e-7`;
- gradient norm ratio alert `20.0`;
- no deterministic-action KL.

Action caps:

- translation residual cap `rho_trans = 0.02`;
- rotation residual cap `rho_rot = 0.05`;
- gripper residual cap `rho_grip = 0.25`.

These thresholds may not change after Stage 0 begins.

## Frozen Label And Target Construction

All labels are development-only and may use `B`, `E`, `R`, and legal history
features. They are not inference inputs.

For row `i`, define:

`e_base(i) = err(B_i, E_i)`.

Build leave-one-out neighbors within the same split and task:

- `j_cur(i)`: nearest neighbor by current-frame signature
  `c_i = summary(X_0_i, B_i, T_i)`;
- `j_hist(i)`: nearest neighbor by history signature
  `r_i = summary(X_hist_i, U_hist_i, T_i)`.

The summary function must be deterministic, serialized before Stage 0 scoring,
and must contain only legal feature means, standard deviations, first/last/mean
Base chunk statistics, and instruction/task embeddings. It may not contain
success, reward, done, object pose, future observation, or confirmatory
identity information.

Define:

`e_cur(i) = err(E_{j_cur(i)}, E_i)`

`e_hist(i) = err(E_{j_hist(i)}, E_i)`

`benefit(i) = e_cur(i) - e_hist(i)`.

The ambiguity/usefulness label is:

`m_i = 1[e_base(i) >= 0.02 and benefit(i) >= 0.01]`.

The auxiliary history target is:

`z_i = [clip(e_base(i), 0, 1), clip(benefit(i), -1, 1),
        mean_abs(R_i[:, 0:3]), grip_switch(E_i)]`.

Each component of `z_i` is normalized by discovery-set median and interquartile
range. Validation must use discovery normalization only.

Rows without valid current or history leave-one-out neighbors are masked out of
`L_gate` and reported separately. If unmasked labels are all zero, all one,
task-collapsed, or predictable only from task id, Stage 0 stops as
`MHS_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

## First Serious Comparison

The first serious comparison remains exactly:

1. `smolvla_base`
2. `mtil_history_state_proxy`
3. `mhs_full`
4. `mhs_no_history_state_ablation`
5. `standard_lora`

Policy 2 is the closest-prior proxy. If official MTIL code cannot run directly
under the existing SmolVLA/LIBERO scaffold, policy 2 must be a transparent
history-state action-chunk proxy using the same development rows, legal inputs,
action semantics, split, optimizer/parameter budget, and inference budget, but
without MHS's Base-preserving residual gate or exact Base passthrough. It must
be labeled as a proxy, not official MTIL.

The no-history ablation must keep residual/gate capacity, optimizer budget,
action caps, clean-retention terms, training rows, and selection metric
matched while replacing the recurrent state `h` with current-frame features and
Base chunk summary only.

Standard LoRA is required because MHS trains on demonstrations and ordinary
adaptation with the same data and compute is the single strongest simple
reviewer-killer explanation.

## Stage 0 Purpose

Stage 0 is a development-only audit. It is not a closed-loop scientific result
and not a paper claim.

Stage 0 determines whether:

- legal history windows exist with enough task coverage;
- ambiguity/usefulness labels are noncollapsed and not task-only;
- the history target is predictable above trivial and current-frame baselines;
- Base leaves residual headroom in the labeled history-useful rows;
- the MTIL history-state proxy leaves residual failure on the MHS claim axis;
- MHS can act distinctly from Base, MTIL proxy, no-history ablation, and
  standard LoRA while preserving action validity and clean behavior.

## Stage 0 Required Artifacts

Stage 0 must produce:

- `reports/mhs_vla/stage_0_preflight.json`;
- `reports/mhs_vla/stage_0_manifest.json`;
- `reports/mhs_vla/stage_0_partial.json`;
- `reports/mhs_vla/stage_0_result.json`;
- `reports/mhs_vla/stage_0_result.md`;
- `reports/mhs_vla/stage_0_adjudication.md`;
- `reports/mhs_vla/stage_0_status.json`;
- `reports/mhs_vla/stage_0_heartbeat.json`;
- `reports/mhs_vla/stage_0_pid.txt`;
- `reports/mhs_vla/stage_0_exit_code.txt`;
- `reports/mhs_vla/stage_0_action_semantics.json`;
- `reports/mhs_vla/stage_0_official_prior_asset_check.json`;
- `reports/mhs_vla/stage_0_serializer_preflight.json`.

If launched detached, Stage 0 must also persist stdout and stderr logs.

## Stage 0 Required Metrics

Required metrics:

- planned and completed row counts;
- exception count;
- duplicate/missing/extra/split-overlap key counts;
- proposal hash match;
- no reward/success/done/object-pose/future-observation/confirmatory reads;
- Base chunk finite and shape `[50, 7]`;
- history-window count by split, task, demo, and time index;
- masked and unmasked label counts;
- positive and negative `m_i` counts;
- label entropy, majority baseline, and task-only baseline;
- validation positive fraction and largest positive-task fraction;
- `z_i` median/IQR normalization statistics from discovery only;
- `e_base`, `e_cur`, `e_hist`, and `benefit` p50, p75, p95, and max;
- residual target statistics by task, phase proxy, and action group;
- history predictor metric versus majority, task-only, and current-frame-only
  baselines;
- history nearest-neighbor proxy versus current-frame nearest-neighbor proxy;
- MTIL proxy score and remaining residual headroom;
- MHS full versus Base, MTIL proxy, no-history ablation, and standard LoRA;
- identity initialization and disk-reload max absolute error;
- checkpoint persistence and reload status;
- finite nonzero gradients for history encoder, residual head, gate, and
  auxiliary head;
- zero gradients for frozen SmolVLA Base parameters;
- objective magnitudes for `L_res`, `L_gate`, `L_hist`, `L_clean`, and
  `L_valid`;
- weighted objective gradient-norm ratio;
- gate p50, p95, max, and intervention fraction;
- residual norm and action delta summaries by translation, rotation, and
  gripper groups;
- clean-retention delta p95 on inactive rows;
- action-validity rate after official postprocessing.

## Stage 0 Pass Gates

All must pass:

- proposal hash matches
  `BBDF67AE3EC4BD9D025707A8BB3A5008BAB5EB5C691D02D44516157802A87BF3`;
- no reward, success, done, simulator result, object pose, future observation,
  demonstration action at inference, or confirmatory identity is read;
- manifest has zero duplicate, missing, extra, or split-overlap keys;
- exceptions are zero;
- Base chunks are finite and postprocessor-valid;
- history-window row counts satisfy the fixed minimums;
- validation unmasked label count is at least `128`;
- validation positive count is at least `8`;
- validation negative count is at least `8`;
- validation positive fraction lies in `[0.02, 0.80]`;
- largest positive-task fraction is at most `0.75`;
- `z_i` has finite nonzero discovery IQR for every component after safe
  fallback handling is applied;
- history predictor beats the strongest of majority, task-only, and
  current-frame-only baselines by at least `0.02` normalized BCE/Huber score;
- history nearest-neighbor `e_hist` beats current-frame nearest-neighbor
  `e_cur` by at least `0.01` mean Huber on validation positive rows;
- Base residual activity exists on validation positive rows;
- MTIL proxy leaves measurable residual headroom for MHS;
- MHS full beats the strongest of MTIL proxy, no-history ablation, and
  standard LoRA by at least `0.005` normalized validation proxy Huber on
  history-useful rows;
- identity initialization and disk reload max absolute error are `<= 1e-7`;
- expected MHS parameters receive finite nonzero gradients;
- frozen SmolVLA Base parameters receive no gradients;
- weighted objective gradient-norm ratio is at most `20x` median;
- after a small fit, MHS full differs from Base, MTIL proxy, no-history
  ablation, and standard LoRA in a bounded way;
- intervention fraction lies in `[0.02, 0.80]`;
- action deltas respect translation, rotation, and gripper group caps;
- clean validation rows have p95 deltas at most `10%` of each group cap;
- action validity is `1.0`.

## Stage 0 Stop Classes

Stage 0 must end with exactly one:

- `MHS_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `MHS_STAGE_0_NO_USABLE_HEADROOM`;
- `MHS_STAGE_0_DESIGN_FAILURE`;
- `MHS_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- `MHS_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Classify failures as:

- source, action-shape, history-window, label, task-coverage, duplicate-key,
  split-overlap, legal-label, privileged-input, or task-only-label failure:
  `MHS_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- no Base residual activity, no history-over-current-frame benefit, no
  history predictability, or no residual headroom beyond the MTIL proxy:
  `MHS_STAGE_0_NO_USABLE_HEADROOM`;
- MHS equivalent to no-history ablation, standard LoRA explaining the effect,
  MTIL proxy dominating the matched claim axis, global gate activation, or
  nonacting selective mechanism:
  `MHS_STAGE_0_DESIGN_FAILURE`;
- hash, serialization, identity, reload, gradient, objective-scale,
  frozen-parameter, checkpoint, action-semantics, action-validity, persistence,
  global-delta, or exception defect:
  `MHS_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- all gates pass: `MHS_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

No Stage 0 stop is a closed-loop scientific kill and no Stage 0 stop may be
rescued by changing thresholds, history-window construction, proxy definition,
task/reset identities, label construction, or action-validity semantics after
seeing results.

## Bounded Validation Search

Allowed only after `MHS_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Maximum six configurations:

1. `mtil_history_state_proxy`;
2. `standard_lora`;
3. `mhs_no_history_state_ablation`;
4. `mhs_full_lambda_clean_0_5`;
5. `mhs_full_lambda_clean_1_0`;
6. `mhs_full_lambda_clean_2_0`.

One seed per configuration by default. A second seed is allowed only if the
first validation search is underpowered or numerically unstable, and must be
recorded before confirmatory testing.

Validation score must combine:

- validation proxy improvement over Base and MTIL proxy;
- clean retention;
- history mechanism activation locality;
- full-versus-no-history distinction;
- standard-LoRA distinction;
- action validity;
- compute overhead.

Do not select purely by offline action L2.

## Worker And Resume Rules

Before any expensive command, inspect PID, heartbeat, status, partial, result,
log, and exit-code artifacts. If an MHS worker is alive, monitor it only. If it
completed, adjudicate the result and do not rerun. If it died with valid
partial rows, resume only missing keys:

`(split, task_suite, task_id, demo_id, window_start, history_identity, policy,
config_label)`

Completed rows must not repeat. Duplicate-key and manifest checks must run
before accepting the result.

## Current Authorization

This preregistration authorizes prototype protocol drafting next. It does not
authorize implementation, validation search, training, rollout, or
confirmatory testing until the prototype protocol is frozen.
