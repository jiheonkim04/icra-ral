# SCVC-VLA Researcher Proposal

Date: 2026-07-12 KST

Method: `SCVC-VLA`, Sensor-Canonicalized VLA Control.

## Claim

Frozen VLAs can lose closed-loop success under controlled sensor-domain shifts even when their action generator is otherwise adequate. `SCVC-VLA` tests whether a lightweight calibration-derived canonicalizer on observations can recover shifted closed-loop performance without changing the policy, adding an action head, using a teacher, or ranking candidates.

## Method

For each camera stream, collect clean calibration observations on predeclared train identities and estimate target statistics `(mu_c, sigma_c)`. At deployment, the input image is shifted by a predeclared synthetic sensor transform and canonicalized before the frozen policy:

`x'_c = gamma_c x_c + beta_c`

`c_phi(x'_c) = clip((x'_c - m_c(x'_c)) / (s_c(x'_c)+eps) * sigma_c + mu_c)`.

The full method maintains a running per-camera estimate for temporal stability. The key ablation canonicalizes each frame independently. The simple killer baseline applies the known inverse affine for the synthetic shift.

The final action is always produced by frozen SmolVLA:

`a_t = pi_S(c_phi(o'_t), q_t, l)`.

## What Changes Relative To Prior Kills

- Not action generation, transition prefixes, flow-noise priors, teacher distillation, or trace memory.
- Data source is calibration observations, not teacher actions or success labels.
- Claim axis is controlled deployment sensor-shift robustness, not clean LIBERO improvement.
- Inference intervention is observation preprocessing only.

## Required Baselines

1. `clean_frozen_smolvla`: unshifted diagnostic ceiling.
2. `shifted_frozen_smolvla`: unmodified backbone under sensor shift.
3. `known_inverse_affine`: simple reviewer-killer baseline.
4. `scvc_no_temporal`: per-frame canonicalization ablation.
5. `scvc_full`: calibration-derived temporal canonicalization.

## Prototype

Calibration identities:

- `20260711..20260715`

Stage A held-out identities:

- `20260716..20260720`

Tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

Stage A contains `50` episodes: 2 tasks x 5 identities x 5 variants.

## Kill Conditions

Kill if:

- full SCVC is `0 / 10` while a baseline has at least `4 / 10`;
- full SCVC is at least `0.30` below strongest baseline;
- known inverse affine or no-temporal canonicalization matches full;
- shifted frozen is not degraded and the claimed condition has no headroom;
- the method requires success/reward/privileged simulator signals at inference.
