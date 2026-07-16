# BRID-VLA Preregistration

Date: 2026-07-16 KST

Decision: `BRID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`

Method: `BRID-VLA`, Base-Residual Implicit Diffusion for SmolVLA action
chunks.

Proposal SHA-256:
`2D4769CF126DF0580029486F7D64EF3C09D435571589F87C569F60A71CBC5CA2`

Prerequisite documents:

- proposal: `reports/brid_vla/researcher_proposal.md`
- Reviewer B attack: `reports/brid_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/brid_vla/researcher_rebuttal.md`
- mathematical audit: `reports/brid_vla/mathematical_mechanism_audit.md`

No BRID implementation, training, validation search, rollout, simulator
access, or confirmatory-test access has happened before this preregistration.

## Frozen Claim

BRID tests whether a frozen-SmolVLA, Base-conditioned residual diffusion score
field can learn bounded useful corrections around the Base action chunk while
preserving exact Base behavior by default.

The claim is not:

- official Diffusion Policy reproduction unless official compatible assets are
  installed and verified before confirmatory testing;
- a new VLA backbone;
- ordinary LoRA as the method;
- raw action diffusion as the contribution;
- generic residual imitation or smoothing;
- direct use of demonstration actions at inference;
- or a rescue of AFID or any previous closed method.

## Evidence Partitions

`DISCOVERY / TRAINING`

- legal LIBERO demonstrations only;
- demonstrations `0..7` for each fixed development task;
- used for Base chunk decoding, residual-target construction, diffusion noise
  diagnostics, small-fit gradient checks, raw diffusion proxy diagnostics, and
  implementation debugging;
- may not include confirmatory reset identities, labels, failures, or
  outcomes.

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
- confirmatory outcomes may not retune BRID.

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
`BRID_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

## Frozen Mechanism

Use exactly the mathematical audit variables and constants:

- `H = 50`;
- `D = 7`;
- Base chunk `B_t = pi_base(o_t, q_t, l_t)`;
- demonstration chunk `E_t`;
- residual `R_t = E_t - B_t`;
- noisy residual `r_k`;
- diffusion step `k`;
- noise target `epsilon`;
- predicted noise `epsilon_theta`;
- denoised residual proposal `D_theta`;
- clipped residual `Delta_t`;
- gate `g_theta`;
- output `A_t = B_t + g_theta * Delta_t`;
- default diagnostic diffusion step count `K = 8`;
- identity tolerance `1e-7`;
- no deterministic-action KL.

Fixed Stage 0 caps and gates:

- `rho_translate = 0.02`;
- `rho_rotate = 0.05`;
- `rho_gripper = 0.25`;
- residual-active threshold: validation residual L2 above task median;
- clean-retention rows: validation residual L2 below task lower quartile;
- score-prediction margin requirement: at least `0.02` Huber improvement over
  the strongest trivial baseline;
- residual-headroom requirement: at least `2%` validation Huber reduction
  oracle over Base on residual-active rows;
- intervention fraction range: `[0.02, 0.80]`.

These thresholds may not change after Stage 0 begins.

## First Serious Comparison

The first serious comparison remains exactly:

1. `smolvla_base`
2. `diffusion_policy_action_chunk_proxy`
3. `brid_full`
4. `brid_no_base_residual_ablation`
5. `standard_lora`

If official Diffusion Policy code cannot be run directly under the existing
SmolVLA/LIBERO scaffold, policy 2 is a transparent raw action-chunk diffusion
proxy using the same development rows, same legal inputs, same action
semantics, same split, same optimizer/parameter budget, and same inference
budget, but no Base-residual conditioning or exact Base passthrough. It must
be labeled as a proxy, not official Diffusion Policy.

The no-Base-residual ablation must keep denoising objective, trainable
capacity, optimizer budget, inference step count, action caps, and
clean-retention terms matched while removing Base-residual conditioning and
zero-residual identity integration.

## Stage 0 Purpose

Stage 0 is a development-only audit. It is not a closed-loop scientific result
and not a paper claim.

Stage 0 determines whether:

- residual targets can be constructed without collapse;
- score/noise targets are predictable from legal deployment inputs;
- residual headroom exists relative to frozen Base and raw diffusion proxy;
- BRID can act distinctly from Base, raw diffusion proxy, no-Base-residual
  ablation, and standard LoRA while preserving action validity and clean
  behavior.

## Stage 0 Required Artifacts

Stage 0 must produce:

- `reports/brid_vla/stage_0_preflight.json`;
- `reports/brid_vla/stage_0_manifest.json`;
- `reports/brid_vla/stage_0_partial.json`;
- `reports/brid_vla/stage_0_result.json`;
- `reports/brid_vla/stage_0_result.md`;
- `reports/brid_vla/stage_0_adjudication.md`;
- `reports/brid_vla/stage_0_status.json`;
- `reports/brid_vla/stage_0_heartbeat.json`;
- `reports/brid_vla/stage_0_pid.txt`;
- `reports/brid_vla/stage_0_exit_code.txt`;
- `reports/brid_vla/stage_0_action_semantics.json`;
- `reports/brid_vla/stage_0_official_prior_asset_check.json`;
- `reports/brid_vla/stage_0_serializer_preflight.json`.

If launched detached, Stage 0 must also persist stdout and stderr logs.

## Stage 0 Required Metrics

Required metrics:

- planned and completed row counts;
- exception count;
- duplicate/missing/extra/split-overlap key counts;
- proposal hash match;
- no reward/success/done/object-pose/future-observation/confirmatory reads;
- Base chunk finite and shape `[50,7]`;
- residual target counts by split, task, phase, time index, and action group;
- residual L2/Huber p50, p75, p95, and max;
- residual largest-task and largest-action-group fractions;
- residual-active and clean-retention row counts;
- deterministic noise replay match;
- score-prediction Huber versus zero-noise baseline;
- score-prediction Huber versus mean-noise baseline;
- score-prediction Huber versus task/phase baseline;
- residual oracle Huber reduction over Base;
- raw diffusion proxy score and remaining residual headroom;
- BRID full versus Base, raw diffusion proxy, no-Base-residual ablation, and
  standard LoRA;
- identity reload max absolute error;
- finite nonzero gradients for BRID trainable parameters;
- zero gradients for frozen Base parameters;
- weighted objective gradient-norm ratio;
- intervention fraction;
- clean-retention error on inactive rows;
- translation, rotation, and gripper delta summaries;
- action-validity rate.

## Stage 0 Pass Gates

All must pass:

- proposal hash matches
  `2D4769CF126DF0580029486F7D64EF3C09D435571589F87C569F60A71CBC5CA2`;
- no reward, success, done, simulator result, object pose, future observation,
  or confirmatory identity is read;
- manifest has zero duplicate, missing, extra, or split-overlap keys;
- exceptions are zero;
- Base chunks are finite and postprocessor-valid;
- residual targets are noncollapsed across tasks and action groups;
- validation score prediction beats the strongest trivial baseline by at least
  `0.02` mean Huber;
- residual oracle reduction over Base is at least `2%`;
- raw diffusion proxy leaves measurable residual headroom;
- identity reload max absolute error `<= 1e-7`;
- expected BRID parameters receive finite nonzero gradients;
- frozen Base parameters receive no gradients;
- weighted objective gradient-norm ratio is at most `20x` median;
- after a small fit, BRID full differs from Base, raw diffusion proxy,
  no-Base-residual ablation, and standard LoRA in a bounded way;
- intervention fraction lies in `[0.02, 0.80]`;
- action deltas respect group caps;
- clean-retention error on inactive rows is within the frozen exact-Base
  tolerance;
- action validity is `1.0`.

## Stage 0 Stop Classes

Stage 0 must end with exactly one:

- `BRID_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `BRID_STAGE_0_NO_RESIDUAL_HEADROOM`;
- `BRID_STAGE_0_DESIGN_FAILURE`;
- `BRID_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- `BRID_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Classify failures as:

- source, action-shape, residual-target, task-coverage, duplicate-key,
  split-overlap, noise-identity, or legal-label failure:
  `BRID_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- no residual headroom or no headroom beyond the raw diffusion proxy:
  `BRID_STAGE_0_NO_RESIDUAL_HEADROOM`;
- unobservable score target, BRID equivalent to raw diffusion proxy,
  no-Base-residual ablation explaining the effect, standard LoRA explaining
  the effect, or global/nonacting intervention:
  `BRID_STAGE_0_DESIGN_FAILURE`;
- hash, serialization, identity, reload, gradient, objective-scale,
  frozen-parameter, action-semantics, action-validity, persistence,
  global-delta, or exception defect:
  `BRID_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- all gates pass: `BRID_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

No Stage 0 stop is a closed-loop scientific kill and no Stage 0 stop may be
rescued by changing thresholds, proxy definition, task/reset identities, noise
identities, or action-validity semantics after seeing results.

## Bounded Validation Search

Allowed only after `BRID_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Maximum six configurations:

1. `diffusion_policy_action_chunk_proxy`;
2. `brid_k4_cap1_clean1`;
3. `brid_k8_cap1_clean1`;
4. `brid_k8_cap05_clean1`;
5. `brid_no_base_residual_ablation`;
6. `standard_lora`.

One seed per configuration by default. A second seed is allowed only if the
first validation search is underpowered or numerically unstable, and must be
recorded before confirmatory testing.

Validation score must combine:

- validation proxy improvement over Base and raw diffusion proxy;
- clean retention;
- score predictability;
- residual mechanism activation locality;
- action validity;
- compute overhead.

Do not select purely by offline action L2.

## Worker And Resume Rules

Before any expensive command, inspect PID, heartbeat, status, partial, result,
log, and exit-code artifacts. If a BRID worker is alive, monitor it only. If it
completed, adjudicate the result and do not rerun. If it died with valid
partial rows, resume only missing keys:

`(split, task_suite, task_id, demo_id, window_start, diffusion_step,
noise_identity, policy)`

Completed rows must not repeat. Duplicate-key and manifest checks must run
before accepting the result.

## Current Authorization

This preregistration authorizes prototype protocol drafting next. It does not
authorize implementation, validation search, training, rollout, or confirmatory
testing until the prototype protocol is frozen.
