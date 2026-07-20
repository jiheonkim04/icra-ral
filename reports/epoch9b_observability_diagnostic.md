# Epoch 9B Development-Only Observability Diagnostic

Date: 2026-07-21T01:06:09+09:00

Decision: `NO_RELIABLE_MASS_SIGNAL_IN_EXISTING_FIXED_PROBE_TRAJECTORIES`

This is a controller-investment diagnostic only. It uses Epoch 9 development demos 30..39, groups all trajectories from one reset identity into one fold, and never accesses validation 40..44 or confirmation 45..49.

| control | accuracy | reset-group bootstrap 95% CI |
|---|---:|---:|
| balanced_no_probe_chance | 0.500 | [0.500, 0.500] |
| first_frame_rgb | 0.512 | [0.500, 0.537] |
| endpoint_only_controller_error | 0.512 | [0.500, 0.537] |
| frozen_epoch8_aggregate_lda_shared_channels | 0.500 | [0.500, 0.500] |
| raw_sequence_temporal_encoder | 0.412 | [0.300, 0.500] |
| shuffled_temporal_order | 0.450 | [0.350, 0.500] |
| final_displacement_only_eval_control | 0.463 | [0.400, 0.500] |
| candidate_position_and_order_only_eval_control | 0.500 | [0.500, 0.500] |
| position_displacement_order_eval_control | 0.475 | [0.438, 0.500] |
| response_summary_residualized_against_position_displacement_order | 0.475 | [0.425, 0.500] |

## Interpretation

Existing legal fixed-probe trajectories do not provide a reliable grouped mass signal. A new dynamic probe would need to establish observability before model investment.

The frozen Epoch 8 LDA is not refit. These traces lack its wrist-video channels, so those two features are fixed to the original training mean and the five available controller/agent-view features use the preserved normalization and direction. This is a transparent degraded frozen control, not an official re-evaluation of the Epoch 8 probe.

The conditional control fits a nuisance-to-response map using candidate positions, final simulator displacements, and probe order on each training fold, then classifies only the held-out residual response. Simulator displacement and position are evaluation-only diagnostics and are absent from the raw temporal encoder's inputs.
