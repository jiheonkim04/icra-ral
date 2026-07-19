# Epoch 6 Schedule-Invariance Stage 0 Result

Decision: `ACTION_LEVEL_SCHEDULE_DEPENDENCE_GO`

The fresh calibrated run
`epoch6_stage0_calibrated_repair1_20260720_0405_kst` completed all four
outcome-suppressed sequences. A and its cold restart matched all 20 raw action
chunk hashes. Reversing request order changed all 20 hashes. Median normalized
RMS was `0.0010716091` for the order contrast and `0.0009629101` for the
independent-root reference, a ratio of `1.1128859` against the frozen minimum
of `0.10`.

All 80 query rows used identical input and provenance. There were zero
exceptions, zero WSL swap use, zero simulator actions, and no reward, success,
or done read. This authorizes only the frozen closed-loop problem gate. It does
not authorize Ours, training, method design, validation, confirmation,
replication, or paper generation.

The resource-only amendment qualified the run with a valid 60-second idle
control, 71.51% peak host RAM, one finite CUDA forward, no sustained paging,
no OOM/offload, and a restored clean state. A 5 MiB pagefile `CurrentUsage`
change was recorded as allocation-only telemetry without paging I/O.

Machine-readable evidence and immutable run-artifact hashes are in
`stage0_result.json`.
