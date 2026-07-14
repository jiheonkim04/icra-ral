# Researcher A Rebuttal: FANG-VLA

Date: 2026-07-14 KST

Proposal hash: `6837DBA2A1307F7C9938FA9F5463ED483907AF3C168F1C0514F6E281804E859B`

Reviewer verdict: `CONDITIONAL_PROCEED_TO_AUDIT`.

## Accepted Constraints

Researcher A accepts the core reviewer risks:

- AFIL preempts generic failure-negative learning.
- A2C2 preempts generic chunk residual correction.
- Pre-VLA, VeriSpace, and DREAM-Chunk preempt generic verifier/ranker/candidate-selection framing.
- CAVM's non-parametric memory signal is not enough to rescue memory by retuning.

FANG therefore proceeds only as an AFIL prior extension with identity-preserving integration. The method is dead if the failure residual and gate do not matter.

## Rebuttal To Main Risks

### Residual Target Collapse

The reviewer is correct that logged frozen actions can make naive `a_logged - a_base` residual targets collapse. The implementation must therefore use class-conditional action-field prediction and derive residuals only at inference relative to the current base action, not assume oracle corrective deltas exist in the trace.

Concretely, the audit must measure:

- success/failure action-field separation in action space;
- trainable target variance after feature standardization;
- whether success and failure heads produce different action fields and bounded derived residuals on validation states.

If this fails, the outcome is `DATA_FAILURE`, not a closed-loop negative result.

### AFIL Proxy Dominance

The AFIL proxy is mandatory. If the unconstrained proxy beats FANG and clean retention is acceptable, then identity preservation may not be the right extension and this formulation should not be rescued by threshold tuning.

### Success-Only Equivalence

The no-failure ablation is mandatory. FANG's contribution requires the failure residual to improve beyond success-only residual guidance.

### Global Disruption

FANG will not proceed to confirmatory rollout unless action deltas are bounded before rollout. The audit must report translation, rotation, gripper, residual norm, gate rate, and action-bound validity.

## Revised Experimental Discipline

The preregistration must freeze:

- discovery/validation/test identities;
- six validation configurations;
- validation score;
- final policy variants;
- Stage A and Stage B manifests;
- GO and kill criteria;
- no confirmatory retuning.

No additional hidden configurations are allowed. Failed configurations remain reported.

## Decision

Proceed to:

1. `mathematical_mechanism_audit.md`;
2. `preregistration.md`;
3. `prototype_protocol.md`;
4. development-only audit implementation.

Do not launch closed-loop rollout before the audit classifies labels, target variance, gradient behavior, checkpoint reload, and policy disruption risk as acceptable.
