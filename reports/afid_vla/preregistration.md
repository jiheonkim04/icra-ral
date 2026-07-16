# AFID-VLA Preregistration

Date: 2026-07-16 KST

Decision: `AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`

Method: `AFID-VLA`, Action-Factor Instruction Densification for
Base-preserving SmolVLA.

Proposal SHA-256:
`B5D1EE12FF2D0280511452DA7FE55295740FD9942A8BE293F444C8EB157062BC`

Prerequisite documents:

- proposal: `reports/afid_vla/researcher_proposal.md`
- Reviewer B attack: `reports/afid_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/afid_vla/researcher_rebuttal.md`
- mathematical audit: `reports/afid_vla/mathematical_mechanism_audit.md`

No AFID implementation, training, validation search, rollout, simulator access,
or confirmatory-test access has happened before this preregistration.

## Frozen Claim

AFID tests whether a frozen-SmolVLA, Base-preserving residual gate can use
deployment-observable predictions of compact action factors to permit bounded
residual edits only in factor-conditioned action cells, while preserving exact
Base behavior whenever factor confidence is low or the mask is inactive.

The claim is not:

- official FineVLA reproduction unless official compatible assets are
  installed and verified before confirmatory testing;
- a new VLA backbone;
- ordinary LoRA as the method;
- generic action imitation;
- direct use of factor labels or demonstration actions at inference;
- or a rescue of LCG or any previous closed method.

## Evidence Partitions

`DISCOVERY / TRAINING`

- legal LIBERO demonstrations only;
- demonstrations `0..7` for each fixed development task;
- used for Base chunk decoding, factor-label extraction, residual-scale
  fitting, factor-mask construction, small-fit gradient checks, proxy
  diagnostics, and implementation debugging;
- may not include confirmatory reset identities, labels, failures, or
  outcomes.

`VALIDATION`

- legal LIBERO demonstrations only;
- demonstrations `8..9` for each fixed development task;
- used for Stage 0 data/mechanism gates and, only after a Stage 0 pass,
  bounded validation search and final configuration selection;
- may not use confirmatory outcomes.

`CONFIRMATORY TEST`

- untouched until method, configuration, policy list, ablation, tasks, reset
  identities, metrics, thresholds, manifests, and checkpoints are frozen;
- no confirmatory task/reset identities, rewards, success flags, done flags,
  object poses, future observations, policy actions, or failed rollouts may be
  read during Stage 0 or validation search;
- confirmatory outcomes may not retune AFID.

## Fixed Development Sources

Use these four source task families:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

Discovery/training demonstrations: `0..7`.

Validation demonstrations: `8..9`.

Minimum Stage 0 windows:

- at least `512` discovery windows;
- at least `128` validation windows;
- every task must contribute validation rows;
- no task may contribute more than `40%` of the Stage 0 validation subset.

If these row counts cannot be produced without duplicate keys or confirmatory
identity reads, Stage 0 must stop as
`AFID_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

## Frozen Mechanism

Use exactly the mathematical audit variables and constants:

- `H = 50`;
- `D = 7`;
- Base chunk `B_t = pi_base(o_t, q_t, l_t)`;
- demonstration chunk `E_t`;
- residual `R_t = E_t - B_t`;
- discovery-only residual scale `S_d`;
- labels `Z_axis`, `Z_dir`, `Z_grip_type`, `Z_grip_bin`, `Z_rot`, `Z_term`;
- factor mask `M_factor`;
- factor predictor probabilities `P_axis`, `P_dir`, `P_grip_type`,
  `P_grip_bin`, `P_rot`, `P_term`;
- confidence `c_theta`;
- residual head `Delta_theta`;
- identity gate `G_theta`;
- output `A_AFID = B_t + G_theta * Delta_theta`;
- caps `0.02` translation, `0.05` rotation, `0.25` gripper;
- no deterministic-action KL.

Fixed extraction thresholds:

- `tau_axis_motion = 0.03`;
- `tau_dir = 0.01`;
- `tau_rot = 0.02`;
- `tau_grip_event = 0.20`;
- `tau_settle = 0.015`;
- `tau_residual_mask = 0.50`;
- `tau_conf = 0.60`;
- `tau_entropy = 0.75`.

These thresholds may not change after Stage 0 begins.

## First Serious Comparison

The first serious comparison remains exactly:

1. `smolvla_base`
2. `finevla_action_factor_proxy`
3. `afid_full`
4. `afid_no_factor_ablation`
5. `standard_lora`

If official FineVLA assets are not locally compatible, policy 2 is a
transparent local proxy using the same frozen SmolVLA Base, same development
splits, same factor labels, same optimizer/parameter budget, same action
postprocessor, and same inference budget, but no AFID residual gate. It must
be labeled as a proxy, not official FineVLA.

The no-factor ablation must keep trainable capacity, labels, optimizer budget,
clean-retention terms, and action caps matched while removing predicted
factors from the gate.

## Stage 0 Purpose

Stage 0 is a development-only audit. It is not a closed-loop scientific result
and not a paper claim.

Stage 0 determines whether:

- factor labels can be extracted without collapse;
- factor labels are observable from legal deployment inputs;
- factor-conditioned residual headroom exists;
- the FineVLA proxy leaves residual headroom;
- AFID can act distinctly from Base, FineVLA proxy, no-factor ablation, and
  standard LoRA while preserving action validity and clean behavior.

## Stage 0 Required Artifacts

Stage 0 must produce:

- `reports/afid_vla/stage_0_preflight.json`;
- `reports/afid_vla/stage_0_manifest.json`;
- `reports/afid_vla/stage_0_partial.json`;
- `reports/afid_vla/stage_0_result.json`;
- `reports/afid_vla/stage_0_result.md`;
- `reports/afid_vla/stage_0_adjudication.md`;
- `reports/afid_vla/stage_0_status.json`;
- `reports/afid_vla/stage_0_heartbeat.json`;
- `reports/afid_vla/stage_0_pid.txt`;
- `reports/afid_vla/stage_0_exit_code.txt`;
- `reports/afid_vla/stage_0_action_semantics.json`;
- `reports/afid_vla/stage_0_official_prior_asset_check.json`;
- `reports/afid_vla/stage_0_serializer_preflight.json`.

If launched detached, Stage 0 must also persist stdout and stderr logs.

## Stage 0 Required Metrics

Required metrics:

- planned and completed row counts;
- exception count;
- duplicate/missing/extra/split-overlap key counts;
- proposal hash match;
- no reward/success/done/object-pose/future-observation/confirmatory reads;
- Base chunk finite and shape `[50,7]`;
- factor-label counts by split, task, phase, and action group;
- factor-label entropy and largest-class fractions;
- usable factor count;
- factor-mask positive fraction by task, phase, timestep, and action group;
- factor prediction accuracy and macro-F1 versus majority baseline;
- factor prediction accuracy and macro-F1 versus task/phase baseline;
- factor-conditioned oracle Huber reduction over Base;
- FineVLA proxy score and remaining residual headroom;
- AFID full versus Base, FineVLA proxy, no-factor ablation, and standard LoRA;
- identity reload max absolute error;
- finite nonzero gradients for AFID trainable parameters;
- zero gradients for frozen Base parameters;
- weighted objective gradient-norm ratio;
- gate activation fraction;
- clean-retention error on inactive/low-confidence rows;
- translation, rotation, and gripper delta summaries;
- action-validity rate.

## Stage 0 Pass Gates

All must pass:

- proposal hash matches
  `B5D1EE12FF2D0280511452DA7FE55295740FD9942A8BE293F444C8EB157062BC`;
- no reward, success, done, simulator result, object pose, future observation,
  or confirmatory identity is read;
- manifest has zero duplicate, missing, extra, or split-overlap keys;
- exceptions are zero;
- Base chunks are finite and postprocessor-valid;
- at least one factor is usable under the audit health gates;
- factor mask global positive fraction lies in `[0.02, 0.80]`;
- every validation task has factor-mask positive fraction in `[0.01, 0.90]`;
- factor prediction beats the best trivial validation baseline by at least
  `0.05` macro-F1 or `0.05` accuracy for every factor used by the gate;
- factor-conditioned oracle reduction over Base is at least `2%`;
- FineVLA proxy leaves measurable residual headroom;
- identity reload max absolute error `<= 1e-6`;
- expected AFID parameters receive finite nonzero gradients;
- frozen Base parameters receive no gradients;
- weighted objective gradient-norm ratio is at most `20x` median;
- after a small fit, AFID full differs from Base, FineVLA proxy, no-factor
  ablation, and standard LoRA in a bounded way;
- gate activation fraction lies in `[0.02, 0.80]`;
- action deltas respect group caps;
- clean-retention error on inactive/low-confidence rows is within the frozen
  exact-Base tolerance;
- action validity is `1.0`.

## Stage 0 Stop Classes

Stage 0 must end with exactly one:

- `AFID_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `AFID_STAGE_0_NO_USABLE_HEADROOM`;
- `AFID_STAGE_0_DESIGN_FAILURE`;
- `AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE`;
- `AFID_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Classify failures as:

- source, action-shape, factor-label, factor-mask, task-coverage,
  duplicate-key, split-overlap, or legal-label failure:
  `AFID_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- no factor-conditioned residual headroom or no headroom beyond the FineVLA
  proxy: `AFID_STAGE_0_NO_USABLE_HEADROOM`;
- unobservable factors, AFID equivalent to FineVLA proxy, no-factor ablation
  explaining the effect, standard LoRA explaining the effect, or global/nonacting
  gate activation: `AFID_STAGE_0_DESIGN_FAILURE`;
- hash, serialization, identity, reload, gradient, objective-scale,
  frozen-parameter, action-semantics, action-validity, persistence,
  global-delta, or exception defect:
  `AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE`;
- all gates pass: `AFID_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

No Stage 0 stop is a closed-loop scientific kill and no Stage 0 stop may be
rescued by changing thresholds, proxy definition, task/reset identities, masks,
or action-validity semantics after seeing results.

## Bounded Validation Search

Allowed only after `AFID_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Maximum six configurations:

1. `finevla_action_factor_proxy`;
2. `afid_tau050_clean1_small`;
3. `afid_tau060_clean1_small`;
4. `afid_tau070_clean1_small`;
5. `afid_no_factor_ablation`;
6. `standard_lora`.

One seed per configuration by default. A second seed is allowed only if the
first validation search is underpowered or numerically unstable, and must be
recorded before confirmatory testing.

Validation score must combine:

- validation proxy improvement over Base and FineVLA proxy;
- clean retention;
- factor predictability;
- mechanism activation locality;
- action validity;
- compute overhead.

Do not select purely by offline action L2.

## Worker And Resume Rules

Before any expensive command, inspect PID, heartbeat, status, partial, result,
log, and exit-code artifacts. If an AFID worker is alive, monitor it only. If
it completed, adjudicate the result and do not rerun. If it died with valid
partial rows, resume only missing keys:

`(split, task_suite, task_id, demo_id, window_start, factor_key, policy)`

Completed rows must not repeat. Duplicate-key and manifest checks must run
before accepting the result.

## Current Authorization

This preregistration authorizes prototype protocol drafting next. It does not
authorize implementation, validation search, training, rollout, or
confirmatory testing until the prototype protocol is frozen.
