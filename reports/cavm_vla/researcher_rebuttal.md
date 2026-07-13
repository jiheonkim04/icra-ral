# Researcher A Rebuttal: CAVM-VLA

Date: 2026-07-13 KST

Reviewer verdict: `CONDITIONAL_PROCEED`.

## Accepted Objections

The proposal accepts the main reviewer risk: success-only memory and nearest-success replay are the decisive baselines. CAVM cannot claim novelty if the failed-trace contrast term does not improve held-out closed-loop success or mechanism metrics.

The proposal also accepts that terminal failure labels are coarse. Failed episodes are not assumed to contain bad actions at every step. CAVM therefore uses a density-and-margin gate rather than a supervised per-step value label, and Stage 0 must show local success/failure action separation before any closed-loop method evaluation.

## Design Tightening

1. The retrieval key and all standardization parameters are fit only on acquisition/calibration identities.
2. Stage 0 computes separation only on acquisition/calibration records.
3. Held-out Stage 2A/2B identities are never inserted into memory.
4. No object pose, reward, simulator predicate, or task-progress signal is used at inference.
5. The no-failure ablation and success-only proxy are mandatory and use the same retrieval key, bandwidth, density gate, and clipping constants.
6. The nearest-success baseline uses the same density gate so it is a fair simple killer rather than a weak replay baseline.

## Clarified Contribution

CAVM is not a memory paper in the broad sense. It is a contrastive action-prior test:

`success memory alone` versus `success memory minus failure memory`.

The scientific decision is therefore binary:

- if the failure-memory term helps, the method earns a narrow outcome-contrastive memory claim;
- if not, CAVM is a valid kill and should not be retuned or rescued.

## Proceed Decision

Proceed to preregistration and implementation under Reviewer B's required conditions.
