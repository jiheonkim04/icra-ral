# LIBERO-Spatial Task 5 Candidate Generation

Decision: `EXACTLY_TWO_CANDIDATES_GENERATED_ONE_SELECTED`

This is a report-only candidate-generation step for `libero_spatial/task_5`, reset identity `20260727`. It does not train, implement, checkpoint, or evaluate Ours.

## Target

- Instruction: `pick up the black bowl on the ramekin and place it on the plate`
- Initial state index: `16`
- Initial state SHA-256: `7230223d3b36c289be0dc4cfbfe916bfe65e2b20c4755b123504b97f9db19e76`

## Gate evidence

| Gate | Evidence | Result |
| --- | --- | --- |
| First prior | `reports/post_secondprior_libero_spatial_20260727_prior_scan_result.json` | X-VLA completed 10/10; task 5 failed cleanly; task result SHA `9a6da411db84298748e5a35d23aa5784339f6bc14cdbe24f6842e6a5e6ce40be` |
| Base | `reports/post_secondprior_libero_spatial_20260727_base_gate_result.json` | SmolVLA Base failed cleanly; result SHA `353e3d66bd98696f2a5d64e86f3eb72295b61b18091aba56fdda09da0b3e0941` |
| Headroom | `reports/post_secondprior_libero_spatial_20260727_headroom_result.json` | Task-level expert replay succeeded; same-reset HDF5 headroom unavailable; result SHA `42c0b9e287904a7781cf077397c64578a3a5fb7ab651f30f85f810f18eb44fb9` |
| Second prior | `reports/post_secondprior_libero_spatial_20260727_second_prior_result.json` | Quantized OpenVLA-OFT INT4 failed validly with `libero_spatial_no_noops`; result SHA `ac550a1cf3c779495f645c6a9f9cf10d336d99723ddefdc872b803e19a69b0f1` |
| Data audit | `reports/post_secondprior_libero_spatial_20260727_data_audit_result.json` | PASS; 50 demos, fixed 40/10 split, no residual init overlap, source/transit/target chunks present |

The data audit allows candidate generation only. Training is still blocked until a frozen specification is written and validated.

## Exactly two candidates

| Candidate | Status | Core idea | Main reason |
| --- | --- | --- | --- |
| `R2P-XVLA` / Ramekin-to-Plate Phase-Balanced X-VLA Adapter | `SELECTED_FOR_FROZEN_NO_TRAINING_SPEC` | Use training-only HDF5 geometry to label `source_on_ramekin`, `transit`, and `target_on_plate`, then give transit and placement phases explicit weight in a small future adapter/OFT objective. | Directly matches the observed bowl transfer residual and the audited phase supervision without requiring privileged inference inputs. |
| `CTR-XVLA` / Clearance-Triggered Temporal Requery X-VLA | `NOT_SELECTED` | Shorten or requery action chunks around inferred lift/place windows. | Plausible simple-control threat, but current evidence does not show chunk staleness is the primary failure; it risks becoming a heuristic wrapper. |

## Selected candidate: `R2P-XVLA`

`R2P-XVLA` is selected as the narrowest next candidate because it is tied to the only fresh residual and the audited supervision:

- train phase chunks: source/transit/target = `2627 / 650 / 1048`;
- validation phase chunks: source/transit/target = `711 / 164 / 246`;
- inference inputs remain only RGB, wrist RGB, proprioception, and instruction;
- phase labels and object state slices are training-only and must not be used at inference.

Future LoRA/OFT would be implementation infrastructure only, not the claimed contribution. No LoRA/QLoRA training happened here.

## Comparator-role requirements for the future spec

| Comparator | Scientific question | Blocking condition |
| --- | --- | --- |
| X-VLA-Libero closest prior | Does the mechanism improve the matched residual over the policy it extends? | No target-condition advantage or no coherent Pareto advantage. |
| Quantized OpenVLA-OFT INT4 context prior | Is the residual serious under another official prior family? | Already failed validly; future spec must not treat its failure as Ours evidence. |
| Uniform LoRA/OFT simple control | Would ordinary task adaptation explain the gain? | Uniform adaptation matches/exceeds R2P under matched data, compute, and checkpoint selection. |
| No-phase-balancing ablation | Is the phase-balanced component responsible? | Removing phase weighting preserves the same claim-specific effect. |

## Non-selected candidate: `CTR-XVLA`

`CTR-XVLA` is not selected because the current evidence points more directly to transition/placement policy content than to generic action-chunk staleness. It may later serve as a simple control if the selected candidate reaches a frozen implementation stage.

## No-training statement

- Training happened: `false`
- Optimizer step happened: `false`
- Checkpoint written: `false`
- Closed-loop Ours evaluation happened: `false`
- LoRA/QLoRA training happened: `false`
- Candidate generation happened: `true`

## Next action

Create a frozen no-training specification for selected candidate `R2P-XVLA`. Do not train, run optimizer steps, write checkpoints, or launch closed-loop Ours rollouts until that spec is validated.
