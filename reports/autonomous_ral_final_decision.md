# Autonomous RA-L Final Decision

Current decision: `NO_METHOD_AFTER_3_VALID_CYCLES`

This is a terminal state under the governance correction.

The active campaign target is `PAPER_READY_EXPERIMENTAL_PACKAGE`.

Cycle 1, `DICD-VLA`, is closed with valid prototype decision `SIMPLE_BASELINE_EXPLAINS_METHOD`.

The 50-episode Stage A closed-loop rollout completed with zero exceptions. Full DICD reached `1 / 10`, the direct chunk-index delay baseline reached `2 / 10`, the delay-only baseline reached `2 / 10`, and the no-history ablation reached `1 / 10`. This kills the method under the preregistered rules.

Cycle 2, `FEDO-VLA`, is now closed with valid prototype decision `CLEAN_RETENTION_FAILURE`.

The 70-episode Stage A closed-loop rollout completed with zero exceptions. Full FEDO under faults reached `1 / 10`, while static inverse gain, the APEX-style feedback proxy, and the no-feedback ablation each reached `2 / 10`. Clean frozen SmolVLA reached `4 / 10`; clean FEDO reached `0 / 10`, a `0.40` absolute clean-retention drop. This kills the method under the preregistered rules.

Cycle 3, `GCAP-VLA`, is closed with valid prototype decision `NO_OCCLUSION_ROBUSTNESS_GAIN`.

The 70-episode Stage A closed-loop rollout completed with zero exceptions. Full GCAP under occlusion reached `3 / 10`; occluded frozen SmolVLA reached `4 / 10`; Sobel edge boost reached `5 / 10`; the no-temporal ablation reached `4 / 10`; full-frame hold-last reached `0 / 10`. Clean GCAP reached `5 / 10` versus clean frozen `1 / 10`, but the targeted occlusion robustness claim failed.

No paper-ready claim is made. The final governed decision is `NO_METHOD_AFTER_3_VALID_CYCLES`.
