# COVI-VLA Preregistration

Date: 2026-07-15 KST

Method: `COVI-VLA`

Proposal hash: `338430D2C6CF1D82410C036D79102ED3F38B2367BB35B9AE2811161698A3E621`

Decision: `COVI_PREREGISTRATION_FROZEN_STAGE_0_PENDING`

Reviewer status: `APPROVE_WITH_FIXED_EMPIRICAL_RISKS`

This file freezes the executable development and evaluation contract. It does
not reopen candidate selection, proposal review, rebuttal, or the mathematical
audit.

## Frozen Claim

COVI is a frozen-SmolVLA, identity-preserving complementary-feature adapter
for scene-induced occlusion. It is not official VIM, a complete VIM
reproduction, a calibration-free camera-centric action method, or a generic
random-cutout defense.

Stage 0 may use the synthetic irregular-occluder transform below only as a
development mechanism proxy. A Stage 0 pass does not validate physical
scene-induced occlusion. Physical occlusion must be evaluated before the
confirmatory paper claim is made.

## Evidence Partitions

The existing official manifest is frozen:

- `DISCOVERY/FIT`: one of the two existing train episodes per task, selected by
  smaller official episode index; `40` episodes and `600` sampled records.
- `DISCOVERY/ONE_CHECK`: the other existing train episode per task; `40`
  episodes and `600` sampled records. It remains sealed unless Stage 0 returns
  `COVI_STAGE_0_UNDERPOWERED_ONE_CHECK_ALLOWED`.
- `VALIDATION`: the existing validation split; `40` episodes and `400` sampled
  records.
- `CONFIRMATORY_TEST`: the existing test split; `80` episodes and `1200`
  sampled records. Stage 0 and the one-check path must not decode, inspect,
  train on, or score these records.

All overlap counts must be zero at sample, frame, episode, and split identity
levels. No confirmatory identity may affect occluder severity, training,
checkpoint choice, normalization, threshold selection, or adjudication.

## Executable SmolVLA Contract

The following values were measured by loading the official local checkpoint
before freezing this preregistration:

- checkpoint: `C:\assets\checkpoints\smolvla_libero`
- dataset: `C:\assets\datasets\lerobot_libero`
- raw image per stream: `[B, 3, 256, 256]`
- prepared image per stream: `[B, 3, 512, 512]`
- available dataset streams: `2`
- configured but absent third stream: omitted by `empty_cameras=0`
- visual embedding per available stream: `[B, 64, 960]`
- two-stream visual prefix: `[B, 128, 960]`
- action chunk: `[B, 50, 7]` after output slicing
- stable intervention point:
  `SmolVLAPolicy.model.embed_prefix`, immediately after
  `vlm_with_expert.embed_image` and before prefix concatenation

If this hook or any listed shape is not reproduced at runtime, Stage 0 returns
`IMPLEMENTATION_OR_DATA_FAILURE`; COVI may not be rescued as an action-residual
method.

## Stage 0 Development Configuration

Stage 0 is one fixed mechanism smoke, not validation search:

- seed: `20260715`
- fit records: `600`
- validation records: `400`
- batch size: `32`
- optimizer: AdamW
- learning rate: `0.0003`
- weight decay: `0.0001`
- epochs: `40`
- early stopping: forbidden
- hidden dimension: `256`
- target dimension `D_c`: `960`
- gate maximum `g_max`: `0.10`
- feature clip `tau_E`: `0.25` normalized feature units
- injection stream: second available camera only
- changed visual tokens when active: `64 / 128`

For each stream, the frozen image encoder returns `E_i in R^[B,64,960]`.
The legal source summary is the token mean of each occluded stream. The exact
predictor input is:

`x = concat(mean(E_occ_1), mean(E_occ_2), state_8, base_action_7,
task_one_hot_40, mask_context_6) in R^[B,1981]`.

`mask_context_6` contains per-stream covered fraction and per-stream normalized
mask centroid `(fraction_1, fraction_2, cx_1, cy_1, cx_2, cy_2)`. It is
computed from the known deterministic development transform. At deployment it
must be replaced by a predicted context from RGB; transform metadata is not a
legal confirmatory input.

The development target is:

`c = normalize(mean(E_clean_camera2)) in R^[B,960]`.

Clean camera 2 is a development label and diagnostic oracle only. It may not
enter COVI inference on an occluded confirmatory record.

The Stage 0 predictor is fixed as:

`u = Linear(1981,256) -> GELU -> Linear(256,960)`.

The residual projection is `Linear(960,960)` with zero-initialized final
weights and bias. The gate is `0.10 * sigmoid(Linear(1981,1))`, with a bias
corresponding to an initial gate below `1e-4`. Zero residual initialization
must make the initial policy exactly Base within `1e-6` action p95.

The residual is broadcast to the `64` second-camera tokens and clipped to
`[-0.25, 0.25]` before addition. The frozen SmolVLA parameters receive no
optimizer update.

## Development Occlusion Proxy

The Stage 0 transform is `irregular_scene_obstruction_proxy_v1`:

- use a deterministic SHA256-derived seed from proposal hash plus sample id;
- build one connected mask from three overlapping ellipses;
- target covered fraction `0.18`, with accepted realized range `[0.14, 0.22]`;
- place the centroid inside normalized image coordinates `[0.30,0.70]^2`;
- use different deterministic masks for the two camera streams;
- fill the mask with a resized border-connected crop from the same frame;
- do not use segmentation, object pose, success, reward, or reset identity.

The simple killer uses an equal-area axis-aligned random rectangle and the same
training budget. Because both are synthetic, neither can establish the final
physical-occlusion claim.

## Stage 0 Comparators

Stage 0 must report, without expanding the later five-policy comparison:

1. `train_mean_target`
2. `direct_two_camera_ridge`
3. `vim_view_imagination_proxy_knn5`
4. `random_cutout_equal_area_mlp`
5. `covi_no_imagined_view_ablation`
6. `covi_stage0_full`
7. `clean_complementary_view_oracle_diagnostic`

The VIM label remains `faithful_transparent_local_proxy_not_official_vim`.
The clean complementary view and direct clean-view pass-through are diagnostic
oracles, never deployable policies.

## Frozen Objectives

The Stage 0 objective weights are:

- `lambda_view = 1.0`
- `lambda_clean = 1.0`
- `lambda_delta = 0.10`
- `lambda_gate = 0.01`
- `lambda_action = 0.25`

`L_view` is Huber loss between normalized predicted and target complementary
features. `L_clean`, `L_delta`, `L_gate`, and `L_action` follow the mathematical
audit. Stage 0 may estimate `L_action` and its gradient on a fixed small batch
rather than on every training batch. All loss magnitudes and nonzero gradient
norms must be recorded. If the largest nonzero objective gradient is more than
`100` times the smallest, Stage 0 returns an implementation/optimization
failure or repairs only the documented normalization bug before rerunning.

No KL divergence is permitted between deterministic actions, flow vectors,
features, or action chunks.

## Stage 0 Metrics

Representation metrics are computed on validation records and clustered by
the `40` validation episodes:

- normalized RMSE to clean complementary target;
- cosine similarity;
- full-minus-strongest-trivial normalized RMSE margin;
- full-minus-VIM-proxy margin;
- full-minus-random-cutout margin;
- episode-cluster bootstrap `95%` interval with `5000` resamples;
- normalization sensitivity using raw, L2-normalized, and train-z-scored
  targets.

Action smoke uses exactly one validation record per task, selected as the frame
nearest normalized phase `0.5`, for `40` task-balanced records. It uses fixed
shared diffusion noise across Base, occluded Base, oracle, ablation, and COVI.
Report:

- Base action and COVI action for every smoke record;
- action delta L2 and p95;
- translation, rotation, and gripper deltas;
- output-bound validity;
- clean action retention;
- residual norm, gate value, changed dimensions/tokens, and occlusion context;
- offline target-action error for clean Base, occluded Base, clean-view oracle,
  and COVI as a diagnostic only.

## Stage 0 Gates

Required implementation/data gates:

- exact checkpoint and hook identity reproduced;
- `600` fit and `400` validation records decoded;
- no reserved-test record decoded;
- zero duplicates and zero partition overlap;
- target variance noncollapsed in at least `100` of `960` dimensions;
- mask coverage in `[0.14,0.22]` for at least `0.99` of records;
- finite nonzero predictor, residual, and gate gradients;
- checkpoint save/reload maximum output difference at most `1e-6`;
- initial action delta p95 at most `1e-6`;
- output validity `1.0`;
- no forbidden inference source.

The practical representation advantage threshold is `0.02` normalized RMSE.

Return `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH` only when:

- diagnostic headroom exists;
- COVI point margin over the strongest non-oracle comparator is at least
  `0.02`;
- the episode-bootstrap lower bound is above `0.0`;
- COVI is not dominated by the VIM proxy or random-cutout baseline;
- trained action changes are bounded and clean retention passes;
- the mechanism acts and differs from the no-imagined-view ablation.

Return `COVI_STAGE_0_UNDERPOWERED_ONE_CHECK_ALLOWED` when implementation, data,
headroom, identity, and safety gates pass but the practical margin or interval
is unresolved. A small point estimate, tie, normalization sensitivity, or wide
interval cannot permanently kill COVI.

The one allowed check fits the same frozen Stage 0 configuration on the
previously sealed `DISCOVERY/ONE_CHECK` episode set and evaluates the unchanged
model contract on the same validation split. No coefficient, architecture,
threshold, occlusion severity, or metric may change. Results must be combined
with episode-cluster bootstrap and reported once.

Return `ROBUST_EMPIRICAL_DESIGN_FAILURE` only when data and implementation are
valid, headroom exists, the method acts safely, at least `40` independent
validation episodes are represented, normalization sensitivity is resolved,
and the bootstrap upper bound excludes the `0.02` useful-advantage threshold
against both the strongest trivial baseline and the VIM proxy.

Other allowed outcomes:

- `FATAL_PREIMPLEMENTATION`
- `IMPLEMENTATION_OR_DATA_FAILURE`
- `CONDITION_TOO_SEVERE_OR_NO_HEADROOM`

Only `FATAL_PREIMPLEMENTATION` and `ROBUST_EMPIRICAL_DESIGN_FAILURE` are
permanent pre-rollout kills. Implementation/data failure is not a scientific
method kill.

## Bounded Validation Search After Stage 0

Run only after Stage 0 passes. Exactly six configurations remain frozen:

1. `covi_feat_s005`
2. `covi_feat_s010`
3. `covi_feat_s020`
4. `covi_bottleneck_s005`
5. `covi_bottleneck_s010`
6. `covi_bottleneck_s020`

No seventh configuration, second architecture family, extra coefficient, new
seed, confirmatory identity, or post-result threshold is allowed. Save all
negative configurations and freeze one selected checkpoint before Stage A.

## First Paper-Oriented Comparison

Exactly five policies:

1. `frozen_smolvla_occluded`
2. `vim_view_imagination_proxy`
3. `covi_full`
4. `covi_no_imagined_view_ablation`
5. `random_cutout_clean_retention_baseline`

The matched manifest, physical occlusion condition, checkpoint identities,
metrics, and thresholds must be frozen before confirmatory outcomes are read.

