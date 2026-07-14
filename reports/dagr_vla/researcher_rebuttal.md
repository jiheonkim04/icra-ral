# DAGR-VLA Researcher A Rebuttal

Date: 2026-07-14 KST

Responds to: `reports/dagr_vla/reviewer_attack.md`

Decision: `DAGR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

## Accepted Constraints

Researcher A accepts Reviewer B's narrowing of the novelty claim:

- DAGR-VLA will not claim the broad novelty of dynamic arm/gripper routing; DAM-VLA owns that prior.
- DAGR-VLA's local claim is the minimal frozen-SmOLVLA, identity-preserving route-gated residual adaptation of that prior.
- `dam_static_component_proxy` will be labeled as a faithful transparent local proxy, not an official DAM-VLA reproduction.
- MTF-VLA remains closed and will not be retuned or rescued.

## Protocol Commitments

DAGR-VLA will proceed only if the Stage 0 development audit passes all of the following:

- route labels are noncollapsed for translation, rotation, and gripper groups;
- validation route prediction beats a trivial majority baseline;
- group-specific route activations are not identical and not always-on;
- full DAGR action differs from both static component proxy and shared-residual ablation on validation;
- residuals are initialized to base passthrough;
- action deltas are bounded by group;
- train/validation/test overlap is zero;
- no confirmatory identities are used for route thresholds, residual alpha, policy selection, or metric thresholds.

## Mathematical Commitments

The mathematical audit will define:

- tensor shapes for observations, base action chunks, expert actions, residual targets, route logits, route labels, group masks, and clipped residuals;
- exact route-label construction;
- group-normalized Huber residual objective;
- route-label loss and positive/negative counts;
- action-delta regularizer;
- gradient paths;
- coefficient scales and small-batch loss/gradient magnitude estimates;
- why Huber/L2 is used rather than KL over deterministic 7D action vectors.

## Baseline Commitments

The first serious comparison remains exactly five policies:

1. `frozen_smolvla`
2. `dam_static_component_proxy`
3. `dagr_full`
4. `dagr_no_dynamic_route_ablation`
5. `gripper_transition_heuristic`

No additional internal control will precede this comparison unless a Stage 0 implementation ambiguity would otherwise invalidate one of these five policies.

## Rebuttal Conclusion

The Reviewer B attack does not kill the method before implementation. It correctly narrows DAGR-VLA into a bounded, prior-anchored, falsifiable local extension. Proceed to the mathematical mechanism audit, preregistration, prototype protocol, and then Stage 0 development audit before any expensive training or rollout.
