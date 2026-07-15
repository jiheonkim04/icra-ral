# NICE-VLA Stage 0B1 Offline Development Protocol

Date: 2026-07-15 KST

Proposal hash:
`898BA577B38966D877E3EEC724EB98751BD8C2685CCD0BBA620EB6B6B9598C0A`.

Decision: `NICE_STAGE_0B1_PROTOCOL_FROZEN_IMPLEMENTATION_PENDING`.

## Purpose

Stage 0B1 asks whether the legal frozen visual/action signal is observable
enough for the source-derived mean and NICE covariance to pass the
preregistered offline development gates. It does not select among six
configurations, run a simulator, read task outcomes, or access confirmatory
tasks.

Passing Stage 0B1 authorizes only the fixed validation-only closed-loop
headroom screen in Stage 0B2. Failure is classified before rollout and is not a
confirmatory scientific kill.

## Frozen Extraction Manifest

Discovery training tasks are the six proposal tasks. Use demos `0..11` on each
task and 16 deterministic valid starts per demo. For episode length `T`, starts
are `floor(linspace(0,T-11,16))`. Expected discovery pairs:

`6 tasks * 12 demos * 16 = 1152`.

Validation calibration tasks are the four proposal validation tasks. Use demos
`30..34` and 16 starts per demo. Expected calibration pairs:

`4 * 5 * 16 = 320`.

Validation evaluation uses demos `35..39` and the same sampler. Expected
evaluation pairs: `320`.

Total planned pairs: `1792`. Task contribution is `1/6` in discovery and
exactly `1/4` in each validation split. No task exceeds 25%. Pair keys and
source mapping follow the preregistration. Every row gets a durable compressed
feature cache and atomic partial row.

At most `1792 * 4 = 7168` image views are encoded. Feature caches use float16
for frozen `z_t` and `Delta z`, float32 for actions. The run may consume at
most 4 wall-clock hours and 15.5 GiB peak CUDA allocation. No download is
allowed.

## Frozen Mean Fit

Use VLA-Corrector's official `SiglipResidualMLP` from the inspected commit:

- token width: measured `960`;
- action width: `7`;
- action embedding: `256`;
- hidden widths: `[2048,2048,2048,2048]`;
- dropout: `0`;
- history: one frame;
- target horizon: `k=10`;
- objective: flattened cosine loss;
- optimizer: AdamW;
- learning rate: `1e-3`;
- weight decay: `1e-4`;
- batch size: `8`;
- optimizer steps: exactly `400`;
- gradient clipping: global norm `1.0`;
- seed: `20262011`.

Sample discovery rows with a deterministic shuffled epoch permutation. No
early stopping, validation-selected epoch, architecture, learning-rate, or
loss variant is allowed. Persist step 0, 100, 200, 300, and 400 losses and the
final checkpoint. The final checkpoint is shared by all later arms.

Mean evaluation is equal-task/equal-episode weighted. Full mean must have
strictly lower validation cosine loss than both:

1. zero-change prediction `Delta z=0`, scored as cosine loss `1.0`;
2. a discovery-only equal-task task-mean delta baseline, using the matching
   suite family when the exact validation task is unseen.

The task-mean mapping is by suite (`libero_10`, `libero_goal`,
`libero_object`, `libero_spatial`) and never by validation outcome.

## Frozen Reference Covariance Smoke

Stage 0B1 uses one nonselecting reference covariance solely for observability:

- family: diagonal;
- conditioning: frozen 18D condition;
- requested coverage: `0.95`;
- covariance architecture and variance bounds: mathematical audit;
- mean checkpoint frozen;
- optimizer: AdamW;
- learning rate: `1e-3`;
- weight decay: `0`;
- batch size: `8`;
- optimizer steps: exactly `300`;
- gradient clipping: global norm `1.0`;
- seed: `20262011`.

This is not one of the six validation configurations and cannot be carried
forward as the selected scientific checkpoint. Its purpose is to decide
whether conditional variance can act at all. The rank-8 PCA basis is built and
audited for rank/orthogonality but not trained or scored as a competing family.

## Calibration

Compute natural-pair scores for every calibration row. Group by episode, take
the nearest-rank 90th percentile, and apply the frozen finite-sample
task-balanced conformal estimator at coverage `0.95`.

Apply the resulting threshold once to the disjoint evaluation episodes. Equal-
task empirical episode coverage must be within `0.03` of `0.95`. No alternate
quantile, frame-level calibration, per-task threshold, or threshold variant is
allowed.

## Diagnostic Mismatches

For every eligible validation evaluation natural row construct exactly three
diagnostic rows without changing its current `z_t`:

1. temporal offset: use same-episode `Delta z_(t+20)` when `t+20<T`, otherwise
   exclude that diagnostic key;
2. cross episode: use the same task and normalized frame quantile in the next
   validation evaluation demo, wrapping demo 39 to demo 35;
3. action regime: use the natural target but replace current and previous
   actions with those from the next row in that task whose gripper-transition
   indicator differs; if none exists, exclude and report.

Natural rows are label 0 and diagnostics label 1. Every family is scored and
reported separately. The frozen aggregate gives equal weight to four tasks and
then equal weight to the three diagnostic families. AUROC uses average ranks
for ties.

Prior score is `1-cosine(mu,Delta z)`. NICE score is normalized innovation from
the reference diagonal covariance. Required aggregate gates:

- NICE AUROC `>0.60`;
- NICE AUROC minus prior AUROC `>=0.03`.

These diagnostics cannot become paper evidence or override later closed-loop
headroom.

## Additional Gates

- all 1792 planned pair keys complete;
- exception, duplicate, missing, extra, and overlap counts zero;
- stable latent shape `[128,960]` and action width 7;
- all latent/action values finite;
- every task and episode quota exact;
- mean and covariance intended gradients finite and nonzero;
- SmolVLA and frozen mean covariance gradients exactly zero;
- covariance scale clamped fraction `<0.05`;
- covariance score finite and standard deviation `>1e-6` on every validation
  task;
- rank-8 basis has eight positive directions and orthonormal error `<=1e-5`;
- final mean/covariance reload max error `<=1e-6`;
- Stage 0A Base passthrough remains `0.0` by artifact hash and one fresh
  monitor-disabled decode;
- validation records read exactly `640` pairs;
- confirmatory records, task outcomes, rewards, dones, reset identities, and
  simulator rollouts read zero.

## Decisions

`NICE_STAGE_0B1_PASS_STAGE_0B2_HEADROOM_ALLOWED` requires every gate.

`NICE_STAGE_0B1_DATA_FAILURE` covers malformed, collapsed, missing, imbalanced,
or overlapping legal data.

`NICE_STAGE_0B1_IMPLEMENTATION_FAILURE` covers invalid source fidelity, math,
gradient, checkpoint, manifest, or passthrough implementation. A mechanical
repair may fix code/schema/path/shape/serialization only and must preserve the
failed attempt; it cannot change this protocol.

`NICE_STAGE_0B1_DESIGN_FAILURE_NONOBSERVABLE` applies when implementation and
data are valid but the mean, covariance coverage, nonconstant-score, or frozen
AUROC gates fail. It closes NICE without a rescue, alternate likelihood,
distance, rank, mismatch, threshold, task, sample, or training schedule.

## Runtime Integrity

Persist PID, heartbeat, status, atomic partial JSON, extraction manifest,
feature caches, checkpoints, stdout/stderr, result, validation, and exit code.
Before launch, audit worker liveness and every durable file. Monitor a live
worker only. Resume a dead worker from valid partial on missing pair keys only.

All timing and utilization are development diagnostics. Any overlap or unknown
overlap with a registered resource-contention interval quarantines them. No
Stage 0B1 efficiency value is paper evidence.
