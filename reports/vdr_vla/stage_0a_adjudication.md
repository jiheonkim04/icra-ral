# VDR-VLA Stage 0A Adjudication

Decision: `VDR_STAGE_0A_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.

This is a development-only audit, not a closed-loop scientific kill.
Stage 0B allowed: `False`.
Valid scientific result: `False`.

The frozen Stage 0A gates were applied without changing horizons, PCA dimension, residual construction, baselines, or thresholds.

Final worker PID `411` completed `1536 / 1536` planned model rows with runner
exception count `0`. The final manifest and partial key sets match exactly:
duplicate manifest keys `0`, duplicate partial keys `0`, missing keys `0`,
extra keys `0`, and split-overlap keys `0`.

The blocking gates were the preregistered development gates, not any
confirmatory outcome: action-validity input was false, static predictor
relative improvement was `-2.727311064830038`, action-residual relative /
absolute improvement was `1.2785489495615547e-05 / 5.777584853650097e-06`,
and the FutureVLA-proxy relative / absolute gap was
`-0.08671267131320196 / -0.17766005523582384`.

Attempt 1 is preserved as a pre-manifest preflight/self-worker launch wrapper
blocker with PID `379`, completed rows `0`, and raw launch exit-code text
`1n`. The accepted final run has raw launch exit-code text `0n`; this is a
launcher provenance formatting defect, while the runner status, heartbeat,
validation, and result JSON all record completed execution. The stderr log
also records a non-row heartbeat-thread temporary-file race after completion;
it did not create duplicate rows, missing rows, simulator rows, or a rerun
basis.

VDR Stage 0B, rerun, threshold repair, clipping, rescue, and reinterpretation
are forbidden. Continue to Epoch 4 Cycle 25 candidate generation.
