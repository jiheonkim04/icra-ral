# KITE-VLA Stage 0A Adjudication

Date: 2026-07-16 KST

Final decision: `KITE_STAGE_0A_IMPLEMENTATION_FAILURE`.

This is an `IMPLEMENTATION_FAILURE`, not a valid scientific result or a
scientific kill. Stage 0B, rerun, and KITE rescue are forbidden.

Attempt 1 PID `368` persisted `115 / 128` model rows, then exited `1` on a
Windows/WSL atomic partial-file replacement denial. Its partial parsed with
zero duplicate or extra keys. Resume PID `385` preserved those rows and the
exception, executed only the 13 missing manifest keys, completed `128 / 128`,
and exited `0`. Final duplicate, missing, extra, and split-overlap counts are
all zero. All 64 unique feature-cache hashes pass.

The method diagnostics acted: rank-6 discovery operators improved validation
MSE by `0.9059144587642893` and `0.9208542724043707`; Base headroom passed
with relative deficit `4.216099058335802` and absolute gap
`0.1084517682912324`; KITE gradient was finite and nonzero with ratio
`3.41963609249749`; Base hash, initialized/reloaded identity, and checkpoint
reload passed exactly.

The frozen action-validity gate independently failed. All `128 / 128`
reconstructed `u=0.5` action rows exceeded raw `[-1,1]` bounds somewhere,
across all four tasks, with maximum absolute value `1.1056011915206909`.
There were zero nonfinite rows and zero processor mismatches. Do not clip,
retune, reinterpret, or rerun this formulation.

No optimizer step, simulator load, reward/success/done read, confirmatory
identity access, or closed-loop experiment occurred. Timing, throughput,
wall-clock efficiency, latency, and resource-use evidence are ineligible.

The generated commit scope exceeds 5,000 lines because the preregistered full
manifest contains 11,984 label rows (`396,066` JSON lines, `15,386,993`
bytes) and the resumable partial contains 128 evidence rows (`37,008` JSON
lines, `1,001,424` bytes). These artifacts are required for full split,
duplicate, cache, and missing-key verification; they are generated evidence,
not source-code churn.

Continue automatically to Epoch 4 Cycle 24 exact-three candidate generation
without KITE repair or rescue.
