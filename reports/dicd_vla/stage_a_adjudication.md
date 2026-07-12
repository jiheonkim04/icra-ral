# DICD-VLA Stage A Adjudication

Date: `2026-07-12 KST`

Final decision: `SIMPLE_BASELINE_EXPLAINS_METHOD`

This is a valid scientific result, not a measurement failure.

## Evidence

- completed episodes: `50 / 50`
- rollout exceptions: `0`
- elapsed checkpointed Stage A runtime: `5637.278 s`
- result JSON: `reports/dicd_vla/stage_a_result.json`
- partial checkpoint JSON: `reports/dicd_vla/stage_a_partial_result.json`

## Closed-Loop Success

| Variant | Successes | Task-balanced success rate |
| --- | ---: | ---: |
| frozen SmolVLA clean | `5 / 10` | `0.50` |
| frozen SmolVLA delay | `2 / 10` | `0.20` |
| direct chunk-index delay | `2 / 10` | `0.20` |
| DICD no-history ablation | `1 / 10` | `0.10` |
| DICD full | `1 / 10` | `0.10` |

## Reviewer B Ruling

The mechanism was active: full DICD changed actions with mean action-delta norm `0.291109`. However, the extra delay-indexed history-conditioned adapter did not improve closed-loop task success. The direct chunk-index delay baseline exceeded full DICD, and the no-history ablation matched it.

Under the preregistered Stage A rule, DICD-VLA is killed. No repeat, longer training rescue, threshold tuning, or cosmetic relabeling is allowed.

Next action: start Cycle 2 with a genuinely distinct method family.
