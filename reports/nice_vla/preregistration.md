# NICE-VLA Preregistration

Date: 2026-07-15 KST

Proposal hash:
`898BA577B38966D877E3EEC724EB98751BD8C2685CCD0BBA620EB6B6B9598C0A`.

Decision: `NICE_PREREGISTRATION_FROZEN_STAGE_0A_PENDING`.

## Frozen Method

NICE extends the official-code-derived VLA-Corrector proxy with an
action-conditioned heteroscedastic covariance, normalized innovation, and an
episode-cluster split-conformal threshold. It preserves the frozen mean,
action queue, truncation, recovery, OGG, inference budget, and action
postprocessing of the prior arm.

The method does not use CAVM memory, FANG fields, SPARC conceptors, PCAV
candidates, FAMR adapters, simulator state, reward, done, success, reset
identity, future observations, or future actions at inference.

## Frozen Source

- VLA-Corrector paper: https://arxiv.org/abs/2607.01804;
- official repository: https://github.com/ZJU-OmniAI/vla-corrector;
- inspected commit: `9d23a0ba6fad562d3ed1a68fc52c8a12459abb41`;
- license: Apache-2.0;
- local arm label: `vla_corrector_official_proxy` until exact reproduction is
  independently established.

## Frozen Task Partitions

Discovery tasks:

1. `libero_10/task_1`;
2. `libero_10/task_3`;
3. `libero_goal/task_1`;
4. `libero_goal/task_3`;
5. `libero_object/task_1`;
6. `libero_spatial/task_1`.

Validation tasks:

1. `libero_10/task_5`;
2. `libero_goal/task_5`;
3. `libero_object/task_3`;
4. `libero_spatial/task_3`.

Confirmatory tasks:

1. `libero_10/task_7`;
2. `libero_goal/task_7`;
3. `libero_object/task_5`;
4. `libero_spatial/task_5`.

Task partitions are disjoint. A task string resolves to the zero-indexed HDF5
file in lexicographically sorted canonical path order within its suite. Stage
0A must persist the resolved path, file hash metadata, HDF5 task language, and
canonical mapping; a mismatch is a data failure, not permission to substitute
a task.

## Frozen Demonstration Partitions

- discovery fit/diagnostic demonstrations: `demo_0..demo_29` on discovery
  tasks;
- validation calibration demonstrations: `demo_30..demo_34` on validation
  tasks;
- validation utility demonstrations: `demo_35..demo_39` on validation tasks;
- confirmatory tasks are unread before all configuration and checkpoint hashes
  are frozen.

No frame, episode, task, or extracted pair crosses these roles. Pair key:

`(suite,task_id,hdf5_path,demo_id,frame_t,frame_t_plus_10)`.

Duplicate, missing, extra, or cross-partition pair keys are hard failures.

## Frozen Reset Identities

- discovery closed-loop: `20262001..20262012`;
- validation closed-loop: `20262021..20262032`;
- confirmatory Stage A/B base pool: `20262041..20262050`;
- unresolved expansion pool: `20262051..20262060`.

Closed-loop key:

`(policy_id,suite,task_id,reset_identity)`.

Completed keys are never repeated. A partial episode without a complete row is
rerun under its same key. Resume executes only missing keys.

## Stage 0A: Source, Interface, And Algebra Smoke

Authorized data only:

- `libero_10/task_1`, `demo_0`, `demo_1`;
- `libero_goal/task_1`, `demo_0`, `demo_1`;
- 32 deterministic equally spaced valid `t` frames per demonstration after
  requiring `t+10` in the same demonstration;
- at most 128 records and 128 pairs;
- zero validation or confirmatory records.

If a demonstration has fewer than 32 valid starts, retain all valid starts and
fail the fixed row-count gate; do not substitute another demo.

Stage 0A implements and verifies:

1. exact source commit/license/mechanism provenance;
2. HDF5 observation/action mapping through the official campaign preprocessor;
3. frozen SmolVLA visual latent extraction and stable measured `(L,D)`;
4. normalized action width 7 and within-episode `k=10` pairing;
5. the fixed condition vector and discovery-only gripper deadband;
6. tiny mean and both covariance-family checkpoint round trips;
7. finite nonzero intended gradients and zero frozen gradients;
8. direct-reference diagonal and rank-8 Woodbury algebra;
9. positive definiteness and scale bounds;
10. episode-cluster conformal order statistics including ties;
11. exact monitor-disabled Base queue/action passthrough;
12. manifest, duplicate, split, privileged-input, and test-read checks.

Tiny training is capped at 20 optimizer steps for the mean and 20 for one
diagonal covariance smoke. The rank-8 family requires forward/backward and
reload but no additional fit. Optimizer is AdamW, learning rate `1e-3`, zero
weight decay, batch size at most 8, seed `20262011`. These values are interface
smoke settings and are not validation configurations.

Stage 0A pass requires every mathematical-audit gate plus:

- exactly 128 planned and 128 completed pair rows;
- four demonstrations and two tasks represented equally;
- exception count zero;
- duplicate, missing-manifest, and extra-result counts zero;
- all image/action/latent values finite;
- stable latent shape across all rows;
- action bound validity `1.0` after official normalization/postprocessing;
- Base action identity max absolute error `0.0` with monitor disabled;
- checkpoint reload max absolute error `<=1e-6`;
- covariance clamped fraction below `0.05` at initialization and after smoke;
- zero validation records, confirmatory records, simulator rollouts, and task
  outcomes read.

One implementation repair is allowed only for code, schema, shape, path, or
serialization before scientific gates. It may not change source, task, demo,
frame sampler, row count, hook, model topology, loss, rank, constants, coverage,
optimizer, steps, or criteria. Every attempt remains durable.

Stage 0A classifications:

- malformed/collapsed source data: `DATA_FAILURE`;
- bad wiring, shape, gradients, algebra, serialization, or passthrough:
  `IMPLEMENTATION_FAILURE`;
- valid implementation with absent legal latent/action signal:
  `DESIGN_FAILURE`.

None is a closed-loop scientific kill. Stage 0B is forbidden after a failed
gate or consumed failed repair.

## Stage 0B: Development Headroom And Data Audit

Authorized only after a committed Stage 0A pass. Build the complete frozen
discovery and validation latent manifests, then fit one shared official-topology
mean and bounded covariance diagnostics.

Required data gates:

- nonzero residual variance on every task;
- noncollapsed gripper transition contrast;
- at least 16 natural pairs per episode after censoring;
- equal episode quotas and no task above 25% of each aggregate;
- zero duplication or partition overlap;
- all required targets inferable from deployment inputs;
- no confirmatory task or identity read.

Required mechanism gates:

- mean exceeds zero-change and task-mean residual baselines;
- finite noncollapsed covariance scales;
- episode-cluster coverage error `<=0.03`;
- diagnostic mismatch AUROC `>0.60` and at least `0.03` above prior cosine
  error on the task-balanced aggregate;
- nonconstant scores on every validation task;
- exact monitor-disabled Base behavior;
- same shared mean and recovery budget across prior, Ours, and ablation.

Diagnostic mismatch families are same-episode temporal offset,
same-task/cross-episode future, and action-regime swap. They are development
smoke only.

Before validation search, paired validation-only closed-loop headroom must show
meaningful Base failure and residual failure after the prior. The fixed-short
simple killer and a privileged diagnostic oracle bound must be reported. No
headroom, prior saturation, or fixed-short parity stops before confirmatory
testing under the frozen rule.

## Bounded Validation Search

Exactly six configurations:

1. `diag_c090`;
2. `diag_c095`;
3. `diag_c0975`;
4. `rank8_c090`;
5. `rank8_c095`;
6. `rank8_c0975`.

Every configuration runs both seeds `20262011` and `20262012`; the mean utility
selects the configuration. Seeds are not separate candidates and neither can
be dropped. There are no other architecture, coefficient, history, horizon,
threshold, persistence, cooldown, OGG, task, reset, or learning-rate variants.

Utility and tie-breaks are exactly those in the mathematical audit. All
artifacts and negative results are retained. The selected model, threshold,
policy list, tasks, resets, metrics, gates, and hashes freeze before any
confirmatory task read.

## First Serious Five-Policy Comparison

Policy order is exactly:

1. `smolvla_base_fixed_horizon`;
2. `vla_corrector_official_proxy`;
3. `nice_full`;
4. `nice_mean_only_global_error_ablation`;
5. `fixed_short_horizon_replan`.

Prior, Ours, and ablation share mean checkpoint, action normalization, queue,
recovery, OGG, postprocessing, policy-call budget, and paired cases. Fixed
short uses the preregistered short horizon and no monitor/OGG.

## Confirmatory Stage A

Exactly ten paired cases per policy:

- `libero_10/task_7`, resets `20262041..20262043` (3);
- `libero_goal/task_7`, resets `20262041..20262043` (3);
- `libero_object/task_5`, resets `20262041..20262042` (2);
- `libero_spatial/task_5`, resets `20262041..20262042` (2).

Permanent Stage A stops are limited to mechanism invalidity, no headroom,
catastrophic degradation, clear prior/ablation/simple-baseline dominance, or
exact practical equivalence under the later frozen numerical rule. Small or
ambiguous differences advance automatically. No test outcome may change any
method or threshold.

## Confirmatory Stage B And Expansion

Stage B contains forty paired cases per key policy, including Stage A: each of
the four tasks at all resets `20262041..20262050`.

One unresolved expansion to eighty paired cases per key policy is allowed: add
the same four tasks at resets `20262051..20262060`. There is no third expansion,
task substitution, reset addition, seed selection, or threshold change.

Report task-balanced success, paired wins/losses/ties, paired effect and
failure-rate reduction, deterministic episode/task-cluster bootstrap 95%
interval with seed `20262020` and 10,000 draws, per-task effects, trigger and
recovery mechanism, clean retention, action validity, and uncontaminated
compute only.

## Paper-Candidate Gate

NICE advances only if Ours beats Base, the prior proxy, the key ablation, and
fixed short on the matched claim axis; retains clean behavior; preserves action
validity; and has mechanism evidence consistent with calibrated innovation.

Then, and only then, verify Quantized OpenVLA-OFT INT4 with and without NICE,
add one claim-specific second condition, compare directly relevant recent
baselines when feasible, and report uncontaminated compute and latency.

## Runtime And Resource Integrity

Detached runs persist PID, heartbeat, status, atomic partial JSON, manifest,
result JSON/Markdown, stdout/stderr, and exit code. Before launch or resume,
audit the newest files, worker liveness, JSON parse, planned/completed count,
exceptions, duplicates, and manifest keys. A live worker is monitored only; a
completed worker is adjudicated without rerun; a dead worker resumes missing
keys only.

Intervals in `reports/resource_contention_intervals.json` quarantine latency,
throughput, wall-clock efficiency, and utilization when overlap is positive or
unknown. Synchronous task-success rows remain eligible only with no timeout or
exception and unchanged action, task, and reset semantics.

## Next Authorized Step

Implement and run Stage 0A only. Do not begin Stage 0B, validation search,
closed-loop rollout, or confirmatory access until Stage 0A is validly passed,
adjudicated, committed, and pushed.
