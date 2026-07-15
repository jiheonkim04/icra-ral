# COVI-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Method: `COVI-VLA`

Proposal hash: `338430D2C6CF1D82410C036D79102ED3F38B2367BB35B9AE2811161698A3E621`

Researcher rebuttal: `reports/covi_vla/researcher_rebuttal.md`

Decision: `COVI_MATHEMATICAL_AUDIT_PREREGISTERED`

## Claim Boundary

COVI is mathematically valid only as a narrowed method:

- a frozen-SmolVLA, identity-preserving complementary-feature adapter for
  scene-induced occlusion;
- not official VIM;
- not a full viewpoint-imagination reproduction;
- not calibration-free camera-centric action generation;
- not a generic random-cutout or patch defense.

The closest prior remains LIBERO-Occ / Viewpoint Imagination. The local
`vim_view_imagination_proxy` must remain a transparent faithful proxy until
official equivalence is independently established.

## Variables And Shapes

Per batch or rollout step:

- `B`: batch size.
- `C_img = 2`: official SmolVLA RGB stream count unless implementation audit
  finds a different processor-visible count.
- `I_t`: current RGB input streams after the predeclared occlusion condition.
  Shape `[B, C_img, 3, H, W]` after processor normalization. Exact `H` and `W`
  must be written by Stage 0.
- `q_t`: current proprioceptive state. Shape `[B, 8]`.
- `l`: language or task instruction embedding. Shape implementation dependent.
- `a_base_t`: frozen Base emitted 7D action. Shape `[B, 7]`.
- `A_base_t`: frozen Base action chunk if available. Shape `[B, H_a, 7]`.
  Exact `H_a` must be written by Stage 0.
- `E_t = E_base(I_t, q_t, l)`: tapped frozen SmolVLA visual or
  pre-action feature representation. Shape `[B, N, D_e]` or the closest stable
  implementation hook. Exact `N` and `D_e` must be written by Stage 0.
- `m_t`: predicted occlusion context from legal deployment inputs. Shape
  `[B, D_m]`.
- `c_t`: development-only complementary-view feature target. Shape `[B, D_c]`.
- `u_t`: predicted complementary-view feature. Shape `[B, D_c]`.
- `delta_E_t`: bounded feature residual or adapter conditioning derived from
  `u_t`. Shape `[B, N, D_e]` for feature-token intervention, or `[B, D_z]` for
  the smallest faithful auxiliary conditioning hook.
- `g_t`: adapter gate in `[0, g_max]`. Shape `[B, 1]` or `[B, N, 1]`.
- `E'_t`: adapted feature representation used by the frozen action generator.
  Shape matches `E_t` when a feature hook exists.
- `a_covi_t`: COVI emitted 7D action. Shape `[B, 7]`.

If no stable visual-feature, post-encoder, or pre-action conditioning hook can
be identified, Stage 0 must classify the method as
`IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`. COVI may not be quietly converted into
an action residual, action-history wrapper, or generic correction module.

## Legal Source Gate

Legal inference input:

`x_t = concat(phi_occ(I_t), q_t, emb(l), stopgrad(a_base_t), m_t)`.

Allowed inference values:

- current official RGB streams available to SmolVLA under the predeclared
  occlusion condition;
- current proprioception/state;
- language or task instruction;
- current frozen Base action or action chunk;
- internally predicted occlusion context and complementary representation
  computed from those values.

Forbidden inference values:

- clean unoccluded versions of occluded evaluation images;
- future frames or future actions;
- success, reward, or failure labels;
- reset identity, manifest key, held-out outcome, or task/reset-specific
  tuning signal;
- simulator object pose, segmentation masks, object visibility labels, or
  privileged camera calibration unavailable to the policy;
- confirmatory-test identities or outcomes.

## Complementary Target Construction

Development-only target:

`c_t = P(phi_clean(I_clean_comp_t))`.

Here `I_clean_comp_t` is a clean or complementary camera image available only on
discovery and validation identities before the occlusion transform, or an
official LIBERO-Occ/VIM target if the local assets and protocol are
independently verified. `P` is a frozen projection or pooling operator with
shape `[B, D_c]`.

The clean/complementary view may be used for:

- training target construction on discovery/validation identities;
- a diagnostic oracle upper bound;
- label-health and target-variance audits.

It may not be used as a confirmatory inference input.

Stage 0 must report four separated quantities:

1. Base under the occlusion condition.
2. Clean complementary-view oracle as a diagnostic upper bound only.
3. Direct two-camera pass-through or reweighting diagnostic.
4. Predicted complementary representation from legal occluded inputs.

COVI may proceed only if the predicted representation has noncollapsed signal
and adds value beyond direct pass-through or trivial multiview reweighting by
the preregistered margin.

## Adapter Formula

Complementary feature prediction:

`u_t = f_theta(x_t)`.

Residual or adapter conditioning:

`delta_E_t = R_theta(u_t, E_t)`.

Gate:

`g_t = g_max * sigmoid(h_theta(x_t) + beta_0)`.

`beta_0` must initialize the gate to Base passthrough, with action delta p95 at
most `1e-6` before training. `g_max` is the residual-scale budget selected only
from the frozen validation configs: `0.05`, `0.10`, or `0.20`.

Preferred feature intervention:

`E'_t = E_t + g_t * clip(delta_E_t, -tau_E, tau_E)`.

Action:

`a_covi_t = pi_base(E'_t, q_t, l)`.

The frozen SmolVLA action generator remains frozen. If the implementation uses
an auxiliary conditioning hook instead of direct visual tokens, Stage 0 must
write the exact policy component affected and prove that the hook is still a
feature or representation adapter, not an action residual.

## Objective Terms

Development objective, used only after Stage 0 passes:

`L = lambda_view L_view + lambda_clean L_clean + lambda_delta L_delta + lambda_gate L_gate + lambda_action L_action`.

Complementary-view feature prediction:

`L_view = mean Huber(norm(u_t), norm(c_t))`.

`norm` is a fixed feature normalization chosen before training and reported in
Stage 0. If raw feature scales are used, their mean and standard deviation must
be reported.

Clean retention:

`L_clean = mean ||a_covi_t - a_base_t||_2^2`

on clean or low-occlusion discovery/validation records.

Bounded representation update:

`L_delta = mean max(0, ||g_t * delta_E_t||_F - tau_delta)^2`.

Gate localization:

`L_gate = mean |g_t|`.

Action preservation:

`L_action = mean ||a_covi_t - a_base_t||_2^2`

on records where the occlusion context is negative or below the activation
threshold. If local autograd through the action generator is unavailable,
`L_action` becomes a post-hoc smoke metric rather than a training loss.

No KL divergence may be computed between deterministic 7D actions, SmolVLA flow
vectors, feature residual vectors, or action chunks. They are not normalized
probability distributions. Permitted distances are Huber, L2, normalized
feature error, Mahalanobis distance with fixed covariance, MMD over validated
feature distributions, action-delta norms, and paired closed-loop success after
frozen evaluation.

## Gradient Paths

Allowed gradients:

- through complementary predictor `f_theta`;
- through residual or adapter map `R_theta`;
- through gate `h_theta`;
- through any small projection `P_theta` added by the adapter and declared
  before training.

Frozen paths:

- SmolVLA Base weights;
- Base action `a_base_t` when used as an input feature;
- clean/complementary target features;
- confirmatory-test outcomes.

Forbidden gradients:

- no reward, success, or failure optimization on confirmatory identities;
- no simulator-state, object-pose, segmentation, or visibility-label inference
  path;
- no target leakage from clean confirmatory images.

Before validation training, Stage 0 or the training smoke must estimate on a
small batch:

- each loss term mean and standard deviation;
- gradient norm by parameter group;
- maximum finite/nonfinite count;
- gradient norm ratios across objective terms.

If any objective term dominates another by more than `100:1` without a
predeclared normalization justification, stop for
`IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` or repair before full validation
search.

## Mechanism Smoke Requirements

Before any closed-loop confirmatory rollout:

1. Checkpoint save and disk reload must pass.
2. Adapter and gate parameters must receive finite nonzero gradients.
3. Training and validation losses must behave sensibly.
4. Full COVI must differ from Base and from `covi_no_imagined_view_ablation`.
5. The difference must be bounded and localized to occlusion-relevant states.
6. Action validity must be preserved.
7. Clean validation behavior must be retained.
8. No privileged inference input or confirmatory identity may be used.

For representative occluded and clean states, report:

- Base action;
- COVI action;
- action delta L2 plus translation, rotation, and gripper deltas;
- feature residual norm;
- gate value;
- changed feature dimensions or tokens;
- occlusion context and activation state.

Feature reconstruction alone is insufficient. The smoke must show a bounded
action-distribution consequence under occlusion and clean retention away from
occlusion.

## Simpler Alternatives And Required Ablations

Required closest-prior proxy:

- `vim_view_imagination_proxy`: transparent local proxy until official
  LIBERO-Occ/VIM equivalence is established.

Required key ablation:

- `covi_no_imagined_view_ablation`: removes complementary-view prediction while
  preserving the same frozen Base, processor, source gate, and training budget.

Required simple reviewer-killer baseline:

- `random_cutout_clean_retention_baseline`: ordinary cutout/erasing robustness
  with clean-retention controls. It remains the only mandatory simple killer in
  the first serious comparison and may not be dropped, renamed, or weakened.

Required diagnostics:

- direct two-camera pass-through or reweighting diagnostic;
- clean complementary-view oracle diagnostic, not an inference policy;
- synthetic-mask-only downgrade check for the physical scene-induced occlusion
  claim.

COVI is stopped if random cutout, no-imagined-view, direct camera pass-through,
or the transparent VIM proxy explains the gain.

## Stage 0 Mathematical Audit Requirements

Before validation search, final training, manifest freeze, or rollout:

1. Split and source gate:
   - persisted discovery, validation, and confirmatory identities;
   - zero identity overlap;
   - legal inference feature manifest;
   - no clean evaluation image, segmentation, object pose, visibility label,
     reset identity, success label, or confirmatory outcome at inference.

2. Occlusion headroom:
   - Base degrades meaningfully under the predeclared scene-induced occlusion
     or faithful physical proxy;
   - the VIM proxy leaves residual headroom;
   - diagnostic clean-view oracle proves that the intervention target can
     matter.

3. Label and contrast health:
   - noncollapsed occlusion contexts;
   - target feature variance above a trivial floor;
   - positive and negative occlusion examples across multiple tasks and
     phases;
   - duplicate frame/sample keys equal zero;
   - no train/validation/test overlap.

4. Mechanism observability:
   - predicted complementary features beat image-statistic, direct-cutout,
     random-cutout, and direct-pass-through trivial baselines by the
     preregistered margin;
   - the signal is predictable from deployment-time inputs.

5. Identity-preserving integration:
   - initial action delta p95 at most `1e-6`;
   - Base action validity `1.0`;
   - post-training translation, rotation, and gripper deltas separately
     bounded;
   - activation localized to occlusion-relevant states.

6. Implementation smoke:
   - stable feature or auxiliary conditioning hook documented;
   - expected parameters receive finite nonzero gradients;
   - loss magnitudes and gradient norms reported;
   - checkpoint save/reload succeeds.

Allowed Stage 0 decisions:

- `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`
- `DATA_OR_SUPERVISION_FAILURE`
- `CONDITION_TOO_SEVERE_OR_NO_HEADROOM`
- `DESIGN_FAILURE`
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

Stage 0 stops are development outcomes, not closed-loop scientific kills.

## Validation Search Budget

Run only if Stage 0 passes.

Maximum six configs:

1. `covi_feat_s005`
2. `covi_feat_s010`
3. `covi_feat_s020`
4. `covi_bottleneck_s005`
5. `covi_bottleneck_s010`
6. `covi_bottleneck_s020`

No other architecture, residual scale, coefficient, occlusion severity,
training seed, source variant, or baseline may be added before confirmatory
testing.

Selection score remains:

`S = 0.25 * occluded_validation_proxy + 0.20 * feature_predictability_margin + 0.20 * clean_retention + 0.15 * bounded_action_validity + 0.10 * localized_occlusion_activation + 0.10 * efficiency`.

The score may not use confirmatory identities and may not be offline action L2
alone.

## First Comparison

Exactly five policies:

1. `frozen_smolvla_occluded`
2. `vim_view_imagination_proxy`
3. `covi_full`
4. `covi_no_imagined_view_ablation`
5. `random_cutout_clean_retention_baseline`

Use a matched paired manifest and keep the VIM proxy label transparent unless
official equivalence is established.

## Audit Decision

COVI's mathematical form is valid only under the narrowed claim, source gates,
identity-preserving feature-adapter requirement, physical scene-induced
occlusion requirement, direct-fusion diagnostic, and live random-cutout simple
killer.

Proceed to preregistration and prototype protocol. Do not implement, train,
run validation search, freeze a rollout manifest, or evaluate closed-loop before
those documents freeze Stage 0 and the first comparison.
