# MARC-VLA Researcher A Rebuttal

Date: 2026-07-15 KST

Responds to: `reports/marc_vla/reviewer_attack.md`

Decision: `MARC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

## Accepted Constraints

Researcher A accepts Reviewer B's narrowing of the novelty claim:

- OpenVLA-OFT owns the broad continuous L1/action-chunking fine-tuning prior.
- MARC's local claim is the minimal frozen-SmOLVLA, identity-preserving median-anchor correction of that prior.
- `openvla_oft_l1_proxy` is a faithful transparent local proxy, not an official OpenVLA-OFT reproduction.
- `static_l1_mixture_baseline` is mandatory and will kill the method if it explains the gain.
- DAGR and MTF remain closed and will not be retuned or rescued.

## Protocol Commitments

MARC-VLA will proceed only if Stage 0 passes all of the following:

- disagreement labels are noncollapsed and have adequate positive/negative counts;
- validation gate prediction beats a trivial majority baseline;
- the L1 proxy is valid and separately reported;
- MARC full differs from the L1 proxy, no-gate ablation, and static mixture on validation;
- emitted action equals Base at initialization;
- action deltas are bounded and not globally active;
- train/validation/reserved-test overlap is zero;
- no confirmatory identities are used for thresholds, alpha, architecture, policy selection, or metrics.

## Mathematical Commitments

The mathematical audit will define:

- tensor shapes for observations, base actions, expert actions, anchor predictions, disagreement labels, gates, clipped corrections, and emitted actions;
- exact train-only disagreement-label construction;
- L1/Huber anchor objective;
- gate BCE objective;
- delta and clean-retention regularizers;
- coefficient scales and small-batch loss/gradient magnitude estimates;
- why L1/Huber is used instead of KL over deterministic 7D action vectors.

## Baseline Commitments

The first serious comparison remains exactly five policies:

1. `frozen_smolvla`
2. `openvla_oft_l1_proxy`
3. `marc_full`
4. `marc_no_disagreement_gate_ablation`
5. `static_l1_mixture_baseline`

No additional internal control will precede this comparison unless Stage 0 exposes a concrete implementation ambiguity that would otherwise invalidate one of these five policies.

## Rebuttal Conclusion

The Reviewer B attack does not kill the method before implementation. It correctly narrows MARC-VLA into a bounded, prior-anchored, falsifiable local extension. Proceed to the mathematical mechanism audit, preregistration, prototype protocol, and Stage 0 development audit before any expensive training or rollout.
