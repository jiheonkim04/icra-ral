# Final Project State

Date: 2026-07-06

## Status

Killed / archived for RA-L-stable submission in the current low-compute form.

## Final Main Commit At Kill Decision

The kill/archive package starts from:

`f4b1f778c53344dcc8ba39307e7141c12d8d6b81`

## What Happened

The project explored a low-compute TCA-Map route:

- SmolVLA-first,
- frozen/fixed-prior target conditioning,
- offline proxy comparisons,
- LoRA attribution diagnostics,
- LIBERO/RoboSuite bridge validation,
- matched-init expert replay,
- non-leaking online 7D diagnostic heads.

The offline evidence was promising, but the online action-quality gate failed.

## Final Technical Diagnosis

The method does not currently fail because the target-prior audit leaked labels. The fixed prior-source audit passed.

The method does not currently fail because LIBERO 7D action replay is impossible. The bridge and expert replay were validated.

The method fails because the current low-compute online 7D action head is too weak. The best redesigned head beats ActionMap but loses to a mean-action baseline, so the method does not clear rollout readiness.

## Final Policy State

- No paper-grade claims.
- No further target-prior conditioning redesign on this route.
- No further rollout from the current weak 7D head.
- No TCA-Select revival as a core contribution.
- No representation-collapse claim.
- Preserve logs and reports as negative evidence.

## Recommended Direction

The best next direction is either:

1. start a new rollout-first project with enough training budget for a real online action head, or
2. reframe the work as a diagnostic/tooling project about offline proxy evidence, action-source provenance, and rollout-readiness gates.

