# RCV-VLA Researcher Rebuttal

Date: 2026-07-13 KST

Responds to: `reviewer_attack.md`

Decision: `REBUTTAL_ACCEPTS_LIMITED_CLAIM`

## Position

I accept Reviewer B's narrowing. RCV-VLA will not claim novelty as generic replanning, adaptive chunk size, or closed-loop verification broadly. The claim is limited to a self-supervised, frozen-policy disagreement verifier that predicts when a queued action chunk suffix should be abandoned.

## Changes Accepted

- The direct closest-prior proxy `sv_deviation_proxy` enters the first matched closed-loop comparison.
- `stateless_first_action` is the only mandatory simple reviewer-killer baseline.
- `rcv_no_context_ablation` tests whether the current-state/action-history context matters.
- `tau_train` and `theta_train` are selected only from acquisition identities, never from the evaluation manifest.
- `rcv_full` may not compute queued-vs-fresh disagreement at every step during inference.

## Remaining Hypothesis

The proposal remains worth testing because SV-VLA's central cost problem is that closed-loop reference actions require frequent policy calls. RCV asks whether the frozen policy can supervise a tiny verifier before deployment, so that inference can replan only on predicted stale suffixes.

The method fails scientifically if the verifier's savings come only from ignoring necessary replans, or if success comes only from frequent replanning that the stateless baseline already explains.

## One Allowed Repair

One measurement repair is allowed only if the implementation records an invalid action queue, missing verifier features, missing threshold artifact, or mismatched evaluation manifest before any scientific decision is made. A weak result is not a repair condition.
