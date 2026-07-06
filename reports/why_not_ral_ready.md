# Why This Is Not RA-L Ready

## Short Answer

The project is not RA-L-ready because it does not establish closed-loop robotics-control support for the proposed low-compute method.

## Blocking Reasons

1. Offline proxy success does not transfer to rollout support.
2. Fixed-prior TCA improves offline metrics but lacks a strong online action generator.
3. TCA-Select repeatedly adds no meaningful gain and should not be a main contribution.
4. The representation-collapse claim is unsupported by the current audits.
5. The 7D bridge is validated, but the learned online 7D head is too weak.
6. Expert replay succeeds, but method rollouts do not.
7. The best redesigned online head loses to a mean-action baseline.
8. The final rollout gate is red.

## What Would Be Needed For RA-L

A RA-L-stable version would need at least:

- a non-leaking online action source that clears action-quality gates,
- closed-loop rollout improvement over ActionMap or native baselines,
- target-prior robustness that does not depend on unavailable labels,
- meaningful ablations showing where TCA helps,
- a compute table and no privileged inference,
- evidence that improvements survive more than tiny offline proxy splits.

The current project does not meet those conditions.

## Honest Submission Verdict

Do not submit this as a main RA-L robotics-control paper in the current form.

The strongest honest statement is:

Fixed-prior TCA produces useful offline proxy improvements and exposes a target-conditioned action-decoding hypothesis, but the low-compute route does not yet produce a rollout-ready action policy.

