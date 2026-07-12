# Autonomous RA-L Final Decision

Current decision: `AUTONOMOUS_CAMPAIGN_CONTINUES`

This is not a terminal state.

The active campaign target is `PAPER_READY_EXPERIMENTAL_PACKAGE`.

Cycle 1, `DICD-VLA`, is closed with valid prototype decision `SIMPLE_BASELINE_EXPLAINS_METHOD`.

The 50-episode Stage A closed-loop rollout completed with zero exceptions. Full DICD reached `1 / 10`, the direct chunk-index delay baseline reached `2 / 10`, the delay-only baseline reached `2 / 10`, and the no-history ablation reached `1 / 10`. This kills the method under the preregistered rules.

Cycle 2, `FEDO-VLA`, is now closed with valid prototype decision `CLEAN_RETENTION_FAILURE`.

The 70-episode Stage A closed-loop rollout completed with zero exceptions. Full FEDO under faults reached `1 / 10`, while static inverse gain, the APEX-style feedback proxy, and the no-feedback ablation each reached `2 / 10`. Clean frozen SmolVLA reached `4 / 10`; clean FEDO reached `0 / 10`, a `0.40` absolute clean-retention drop. This kills the method under the preregistered rules.

Cycle 3, `GCAP-VLA`, has passed proposal, adversarial review, preregistration, focused unit tests, and synthetic image-mechanism smoke. It has not yet produced the preregistered Stage A closed-loop result.

No paper-ready claim is made. The next required action is GCAP-VLA Stage A, the final permitted distinct method cycle.
