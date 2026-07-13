# Epoch 4 Cycle 1 RCV-VLA Adjudication

Date: 2026-07-13 KST

Decision: `STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`

## Execution Boundary

- method: `RCV-VLA`
- branch: `codex/autonomous-until-paper-governance-v2`
- proposal hash: `86044E841D178DB5AA485B7D12B01FF8E4274CBDFDCDAC7D427477BF0646F26F`
- Stage 0 diagnostic: `20 / 20` episodes, zero exceptions
- Stage 1 acquisition/training: `7276` step records, full and no-context verifiers saved
- Stage 2A: `50 / 50` episodes, zero exceptions, positive enough to require Stage 2B
- Stage 2B: `200 / 200` episodes, zero exceptions, `40` paired episodes per key policy

## Stage 2B Evidence

Task-balanced closed-loop success:

| Variant | Successes | Task-balanced success | Heavy policy calls / step | Replan rate |
| --- | ---: | ---: | ---: | ---: |
| `queued_frozen_smolvla` | `14 / 40` | `0.35` | `0.021734` | `0.000000` |
| `sv_deviation_proxy` | `16 / 40` | `0.40` | `1.008553` | `0.114241` |
| `rcv_full` | `20 / 40` | `0.50` | `0.563500` | `0.557293` |
| `rcv_no_context_ablation` | `24 / 40` | `0.60` | `0.429078` | `0.421724` |
| `stateless_first_action` | `24 / 40` | `0.60` | `1.000000` | `1.000000` |

Paired comparisons versus `rcv_full`:

| Comparator | Full-minus-comparator delta | Wins | Losses | Ties | Bootstrap CI |
| --- | ---: | ---: | ---: | ---: | --- |
| `queued_frozen_smolvla` | `0.15` | `10` | `4` | `26` | `[-0.025, 0.325]` |
| `sv_deviation_proxy` | `0.10` | `7` | `3` | `30` | `[-0.050, 0.250]` |
| `rcv_no_context_ablation` | `-0.10` | `2` | `6` | `32` | `[-0.250, 0.025]` |
| `stateless_first_action` | `-0.10` | `2` | `6` | `32` | `[-0.225, 0.025]` |

## Scientific Ruling

The RCV mechanism acted: `rcv_full` replanned on `0.557293` of steps and used fewer heavy policy calls than the stateless and SV-proxy variants.

However, the full method failed the preregistered Stage 2B GO gate. It did not beat the key no-context ablation, did not beat the stateless first-action baseline, and used more heavy policy calls per step than the no-context ablation. The observed improvement over queued SmolVLA and the SV-deviation proxy is therefore explained by simpler replanning or no-context thresholding rather than the claimed current-state queued-vs-fresh validity mechanism.

This is a valid current-formulation kill. Do not rescue RCV-VLA by threshold retuning, alternate ablation framing, or another expansion.

## Consequence

Archive RCV-VLA and continue to Epoch 4 Cycle 2. The next candidate must change at least two core dimensions relative to RCV's frozen-policy disagreement verifier, receding-chunk replanning intervention, and efficiency-versus-stateless claim.
