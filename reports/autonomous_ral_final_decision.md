# Autonomous RA-L Final Decision

Current decision: `AUTONOMOUS_CAMPAIGN_CONTINUES`

This is not a terminal state.

The active campaign target is `PAPER_READY_EXPERIMENTAL_PACKAGE`.

Cycle 1, `DICD-VLA`, is closed with valid prototype decision `SIMPLE_BASELINE_EXPLAINS_METHOD`.

The 50-episode Stage A closed-loop rollout completed with zero exceptions. Full DICD reached `1 / 10`, the direct chunk-index delay baseline reached `2 / 10`, the delay-only baseline reached `2 / 10`, and the no-history ablation reached `1 / 10`. This kills the method under the preregistered rules.

Cycle 2, `FEDO-VLA`, has passed proposal, adversarial review, synthetic mechanism smoke, and real SmolVLA trace training. It has not yet produced the preregistered Stage A closed-loop result.

No paper-ready claim is made. The next required action is FEDO-VLA Stage A.
