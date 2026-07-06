# ExecSpec-Repair Failure Tree

Root failure: ExecSpec-Repair is not supported as a broad RA-L-stable method claim.

## Branch A: Evidence Was Real But Narrow

Positive evidence:

- exact-init expert replay can reproduce LIBERO demonstration success,
- wrong executable specs can degrade reward/success,
- calibrated repair can recover many degraded exact-init cases.

Failure:

- the evidence remains exact-init replay/control evidence, not default-reset or benchmark-scale rollout evidence.
- default-reset sanity did not establish robust execution.

Consequence: the route needed unusually strong novelty over baselines to justify continuation. It did not have it.

## Branch B: Simple Baseline Dominance

Observed in STATE 3.5:

- full repair recovered `17 / 19` degraded cases.
- diagonal affine calibration recovered `17 / 19` degraded cases.
- full repair action recovery was `1.0`.
- diagonal affine action recovery was `1.0`.
- full-minus-diagonal success gain was `0.0`.

Consequence: the apparent repair result is explained by per-dimension affine calibration.

## Branch C: Routing Did Not Rescue The Claim

The mismatch-aware selector chose plausible simple repairs per mismatch type and also recovered `17 / 19`.

Failure:

- selector gain over diagonal affine was `0.0`.
- oracle best-per-case recovery also matched full repair and diagonal affine.

Consequence: routing is diagnostically interpretable, but not a publishable mechanism under this evidence.

## Branch D: Novelty Collapsed To Calibration

The intended contribution required a nontrivial executable-spec repair mechanism. The measured contribution is compatible with:

- per-dimension scale/offset correction,
- gripper-only convention handling for gripper cases,
- global affine correction for global/range cases.

Consequence: a paper would be vulnerable to the simplest reviewer objection: "why is this not just diagonal affine calibration?"

## Branch E: Continue Conditions Failed

The route should only be revived if a new benchmark is declared before results are inspected and includes:

- cases where diagonal affine is insufficient,
- control/replay metrics within 48 hours,
- simple baseline comparison within 72 hours,
- direct robot/simulator evidence rather than offline-only proxy,
- kill criteria that include clipping-only, safety-only, mean-action, and diagonal-affine dominance.

Until then, ExecSpec-Repair is archived.

