# LCG-VLA Preregistration

Date: 2026-07-16 KST

Decision: `LCG_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`

Method: `LCG-VLA`, Language-Contrastive Guidance for Base-preserving SmolVLA
actions.

Proposal SHA-256:
`F0D980AA0760F143D781C723DB632BC324C1E18F390D9C33C5DA94F3A897D11E`

Prerequisite documents:

- proposal: `reports/lcg_vla/researcher_proposal.md`
- Reviewer B attack: `reports/lcg_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/lcg_vla/researcher_rebuttal.md`
- mathematical audit: `reports/lcg_vla/mathematical_mechanism_audit.md`

No LCG implementation, training, validation search, rollout, simulator access,
or confirmatory-test access has happened before this preregistration.

## Frozen Claim

LCG tests whether a frozen-SmolVLA, Base-preserving, identity-initialized
action-cell gate can use deployment-observable original-versus-null language
contrast to permit bounded residual edits only in language-sensitive action
cells, while preserving exact Base behavior elsewhere.

The claim is not:

- official Counterfactual Action Guidance reproduction unless official assets
  are installed and verified;
- a new VLA backbone;
- ordinary LoRA as the method;
- counterfactual label augmentation;
- direct use of `B_t - N_t` as an expert residual target;
- or a rescue of S2C or any previous closed method.

## Evidence Partitions

`DISCOVERY / TRAINING`

- legal LIBERO demonstrations only;
- demonstrations `0..7` for each fixed development task;
- used for Base/null chunk decoding, discovery contrast-scale fitting,
  residual-label construction, small-fit gradient checks, proxy diagnostics,
  and implementation debugging;
- may not include confirmatory reset identities, labels, failures, or outcomes.

`VALIDATION`

- legal LIBERO demonstrations only;
- demonstrations `8..9` for each fixed development task;
- used for Stage 0 data/mechanism gates and, only after a Stage 0 pass, bounded
  validation search and final configuration selection;
- may not use confirmatory outcomes.

`CONFIRMATORY TEST`

- untouched until method, configuration, policy list, ablation, tasks, reset
  identities, metrics, thresholds, manifests, and checkpoints are frozen;
- no confirmatory task/reset identities, rewards, success flags, done flags,
  object poses, future observations, policy actions, or failed rollouts may be
  read during Stage 0 or validation search;
- confirmatory outcomes may not retune LCG.

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
`LCG_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

## Frozen Mechanism

Use exactly the mathematical audit variables and constants:

- `H = 50`;
- `D = 7`;
- `l_null = ""`;
- original Base chunk `B_t = pi_base(o_t, q_t, l_t)`;
- null Base chunk `N_t = pi_base(o_t, q_t, l_null)`;
- raw language contrast `U_t = B_t - N_t`;
- discovery-only contrast scale `s_lang_d`;
- normalized contrast `C_t = abs(U_t) / s_lang_d`;
- language mask `M_lang = 1[C_t >= 0.25]`;
- demonstration residual `R_t = E_t - B_t`;
- bounded residual head `Delta_theta`;
- identity gate `G_theta = eta * sigmoid(Z_theta)`, with `eta = 0`
  initialization;
- output `A_LCG = B_t + M_lang * G_theta * Delta_theta`;
- caps `0.02` translation, `0.05` rotation, `0.25` gripper;
- no deterministic-action KL.

`B_t - N_t` is a gate and conditioning signal only. It is not the target
residual.

## First Serious Comparison

The first serious comparison remains exactly:

1. `smolvla_base`
2. `counterfactual_action_guidance_proxy`
3. `lcg_full`
4. `lcg_no_language_contrast_ablation`
5. `standard_lora`

The local CAG proxy is:

`A_CAG(beta) = B_t + beta * clip_group(B_t - N_t, rho_trans, rho_rot, rho_grip)`.

Allowed validation-only coefficients: `beta in {0.25, 0.5, 1.0}`.

The no-language-contrast ablation must keep trainable capacity, optimizer
budget, labels, clean-retention terms, and action caps matched while removing
`N_t` and `C_t` from the gate/residual input.

## Stage 0 Purpose

Stage 0 is a development-only audit. It is not a closed-loop scientific result
and not a paper claim.

Stage 0 determines whether:

- the null branch is valid;
- Base/null language contrast is noncollapsed;
- contrast predicts useful Base-to-demonstration residual headroom;
- the CAG proxy leaves residual headroom;
- LCG can act distinctly from Base, CAG proxy, no-language-contrast ablation,
  and standard LoRA while preserving action validity and clean behavior.

## Stage 0 Required Artifacts

Stage 0 must produce:

- `reports/lcg_vla/stage_0_preflight.json`;
- `reports/lcg_vla/stage_0_manifest.json`;
- `reports/lcg_vla/stage_0_partial.json`;
- `reports/lcg_vla/stage_0_result.json`;
- `reports/lcg_vla/stage_0_result.md`;
- `reports/lcg_vla/stage_0_adjudication.md`;
- `reports/lcg_vla/stage_0_status.json`;
- `reports/lcg_vla/stage_0_heartbeat.json`;
- `reports/lcg_vla/stage_0_pid.txt`;
- `reports/lcg_vla/stage_0_exit_code.txt`;
- `reports/lcg_vla/stage_0_action_semantics.json`;
- `reports/lcg_vla/stage_0_official_prior_asset_check.json`;
- `reports/lcg_vla/stage_0_serializer_preflight.json`.

If launched detached, Stage 0 must also persist stdout and stderr logs.

## Stage 0 Required Metrics

Required metrics:

- planned and completed row counts;
- exception count;
- duplicate/missing/extra/split-overlap key counts;
- proposal hash match;
- no reward/success/done/object-pose/future-observation/confirmatory reads;
- Base chunk finite and shape `[50,7]`;
- null chunk finite and shape `[50,7]`;
- null-branch action validity;
- Base/null contrast positive fraction by task, phase, timestep, and action
  group;
- residual-label noncollapse by task, phase, timestep, and action group;
- contrast-residual Spearman correlation;
- contrast-conditioned residual probe versus task/phase residual baseline;
- best fixed CAG proxy score and remaining residual headroom;
- masked residual oracle diagnostic;
- identity reload max absolute error;
- finite nonzero gradients for LCG trainable parameters;
- zero gradients for frozen Base parameters;
- weighted objective gradient-norm ratio;
- LCG full versus Base, CAG proxy, no-language-contrast ablation, and standard
  LoRA;
- gate activation fraction;
- clean-retention error on inactive-mask rows;
- translation, rotation, and gripper delta summaries;
- action-validity rate.

## Stage 0 Pass Gates

All must pass:

- proposal hash matches
  `F0D980AA0760F143D781C723DB632BC324C1E18F390D9C33C5DA94F3A897D11E`;
- no reward, success, done, simulator result, object pose, future observation,
  or confirmatory identity is read;
- manifest has zero duplicate, missing, extra, or split-overlap keys;
- exceptions are zero;
- Base and null chunks are finite and postprocessor-valid;
- global language-mask positive fraction lies in `[0.05, 0.95]`;
- no development task has all-zero or all-one language mask;
- residual labels are noncollapsed by action group;
- contrast-residual Spearman is at least `0.05`;
- contrast-conditioned residual probe beats task/phase residual baseline by at
  least `1%`;
- best CAG proxy leaves measurable residual headroom for masked oracle;
- identity reload max absolute error `<= 1e-6`;
- expected LCG parameters receive finite nonzero gradients;
- frozen Base parameters receive no gradients;
- weighted objective gradient-norm ratio is at most `20x` median;
- after a small fit, LCG full differs from Base, CAG proxy, no-language
  ablation, and standard LoRA in a bounded way;
- gate activation fraction lies in `[0.02, 0.80]`;
- action deltas respect group caps;
- clean-retention error on inactive-mask rows is within the frozen tolerance;
- action validity is `1.0`.

## Stage 0 Stop Classes

Stage 0 must end with exactly one:

- `LCG_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `LCG_STAGE_0_NO_USABLE_HEADROOM`;
- `LCG_STAGE_0_DESIGN_FAILURE`;
- `LCG_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE`;
- `LCG_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Classify failures as:

- source, action-shape, null-branch, contrast-scale, language-mask,
  residual-label, task-coverage, or legal-label failure:
  `LCG_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- no CAG residual headroom, no masked oracle headroom, or CAG proxy dominance:
  `LCG_STAGE_0_NO_USABLE_HEADROOM`;
- contrast not predictive, LCG equivalent to CAG coefficient tuning, LCG
  equivalent to the no-language ablation, standard LoRA explaining the effect,
  or failure to infer the mechanism from legal deployment inputs:
  `LCG_STAGE_0_DESIGN_FAILURE`;
- hash, serialization, identity, reload, gradient, objective-scale,
  frozen-parameter, action-semantics, action-validity, persistence,
  global-delta, or exception defect:
  `LCG_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE`;
- all gates pass: `LCG_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

No Stage 0 stop is a closed-loop scientific kill and no Stage 0 stop may be
rescued by changing thresholds, null-instruction text, proxy definition,
task/reset identities, masks, or action-validity semantics after seeing
results.

## Bounded Validation Search

Allowed only after `LCG_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Maximum six configurations:

1. `counterfactual_action_guidance_proxy` with validation-selected
   `beta in {0.25, 0.5, 1.0}`;
2. `lcg_small_clean_0p5`;
3. `lcg_small_clean_1p0`;
4. `lcg_medium_clean_1p0`;
5. `lcg_no_language_contrast_ablation`;
6. `standard_lora`.

One seed per configuration by default. A second seed is allowed only if the
first validation search is underpowered or numerically unstable, and must be
recorded before confirmatory testing.

Validation score must combine:

- validation proxy improvement over Base and CAG proxy;
- clean retention;
- mechanism activation;
- action validity;
- compute overhead.

Do not select purely by offline action L2.

## Worker And Resume Rules

Before any expensive command, inspect PID, heartbeat, status, partial, result,
log, and exit-code artifacts. If an LCG worker is alive, monitor it only. If it
completed, adjudicate the result and do not rerun. If it died with valid
partial rows, resume only missing keys:

`(split, task_suite, task_id, demo_id, window_start, instruction_variant, policy)`

Completed rows must not repeat. Duplicate-key and manifest checks must run
before accepting the result.

## Current Authorization

This preregistration authorizes prototype protocol drafting next. It does not
authorize implementation, validation search, training, rollout, or
confirmatory testing until the prototype protocol is frozen.
