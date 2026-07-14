# RAC-VLA Researcher A Rebuttal

Date: 2026-07-14 KST

Reviewer decision: `ALLOW_WITH_REQUIRED_AUDITS`

Researcher proposal hash: `71ABA93E37FC725C1A2E5EAE6E1461BC77AACDAFF9B0711C37F17D5C0AB0902F`

## Scope Clarification

RAC-VLA will be framed as a prior extension of Reflective VLA, not as a new discovery that action consequences matter. The technical claim is narrower:

`A compact, identity-preserving consequence-history calibration layer can transfer the Reflective action-consequence principle to a frozen SmolVLA policy under controlled action-channel deployment shift.`

## Response To Trivial Equivalence Threat

The online diagonal inverse-gain baseline will be mandatory in Stage A/B. RAC is not a paper candidate unless it beats this baseline.

The no-consequence history ablation will be mandatory. It receives the same state, action, task, phase, and horizon features but no realized state-delta consequences.

The action-only baseline will be mandatory in Stage 0. If synthetic perturbation labels are predictable from transformed action features alone at the same quality as action-consequence features, RAC stops as `DESIGN_FAILURE`.

## Response To Synthetic Label Threat

Stage 0 will use predeclared perturbation transforms that cannot be accepted as closed-loop evidence. They are development-only labels for checking whether the mechanism can identify action-effect mismatch. The actual paper-relevant evidence requires closed-loop shifted-condition success against Base, the Reflective proxy, no-consequence ablation, and online inverse-gain.

## Response To EvoState Failure

EvoState tried to improve clean next-state prediction with action-conditioned dynamics and failed the actionless baseline threshold. RAC does not reuse that claim. RAC asks whether consequence history identifies an externally imposed calibration context. The Stage 0 comparator therefore includes action-only and no-consequence classifiers, not only actionless next-state prediction.

## Response To Headroom Threat

The Stage 0 audit will include a bounded oracle check:

- clean Base behavior;
- shifted Base behavior;
- diagnostic inverse-shift oracle behavior or offline proxy when closed-loop oracle is too costly.

The oracle cannot become an inference method. It is only used to confirm that the shift creates recoverable headroom before training or rollout.

## Response To Prior Proxy Threat

The Reflective proxy will use the same consequence history to choose a predeclared inverse template. It will not use RAC's learned residual head or validation-selected residual coefficient. This makes it a faithful local proxy for action-consequence history without architecture retraining.

## Revised Preconditions

RAC must stop before rollout if:

- full action-consequence prediction does not beat action-only and no-consequence features by the preregistered validation margin;
- the gate is collapsed;
- p95 clean action delta exceeds `0.20`;
- the online inverse-gain baseline wins in development validation;
- no recoverable headroom exists under the predeclared action-channel shift.

RAC may proceed to Stage A only if all required audits pass.
