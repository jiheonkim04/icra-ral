# SCVC-VLA Reviewer B Attack

Date: 2026-07-12 KST

Reviewed fixed proposal hash: `BE52CB82140F56E84A0FDBC4D3F51ACD4E704551AC10CC72CE624801DABDE20C`

## Closest Papers

- TTT-VLA: https://arxiv.org/html/2606.03127v1
- Domain Arithmetic: https://arxiv.org/html/2607.00666v1
- VISTA: https://arxiv.org/html/2606.04708v1
- PAD/test-time training for robotic policies
- GCAP-VLA archived in this repository

## Novelty Attack

SCVC is close to ordinary domain adaptation and image normalization. The proposal becomes paper-interesting only if:

- the sensor shift is physically meaningful;
- full SCVC beats shifted frozen and per-frame canonicalization;
- known inverse affine does not fully explain the result;
- clean behavior is retained;
- the final claim is framed as calibration-derived frozen-VLA sensor canonicalization, not generic test-time training.

If the known inverse-affine baseline matches full SCVC, the method is killed. If per-frame mean/std matching matches full SCVC, the temporal calibration component is not useful.

## Simplest Equivalent Method

The simplest equivalent method is:

`x = clip((x' - beta) / gamma)`.

The next simplest method is per-frame mean/std normalization. Both must be included.

## Leakage Risks

- Do not estimate calibration stats on held-out Stage A identities.
- Do not tune color shift severity after seeing Stage A.
- Do not claim clean-task improvement; this is a shifted-condition claim.
- Do not use reward, success, object poses, or simulator state to canonicalize.

## Pre-Implementation Decision

Decision: `IMPLEMENTATION_ALLOWED_WITH_STRONG_KILLER_BASELINES`

Reason: SCVC is not a near-exact duplicate across all axes, but the novelty is fragile. The decisive experiment is cheap and should kill it if it is just normalization.
