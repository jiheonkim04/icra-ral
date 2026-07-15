# PCAV-VLA Stage 0A Adjudication

Date: 2026-07-15 KST

Decision: `PCAV_STAGE_0A_NO_USABLE_HEADROOM`

Failure class: `NO_HEADROOM`

This is a valid frozen Stage 0A diagnostic result. It is not a closed-loop
scientific kill.

## Execution Continuity

The first worker, PID `386`, completed all 24 initial rows and atomically wrote
a valid `24 / 96` expansion partial before a runner `NameError`. The error was
caused by an unbound closure variable after checkpoint reload and is preserved
in `stage_0a_implementation_blocker_attempt_1.json`. It did not invalidate any
completed row.

A second launch was refused in preflight when the stale attempt-1 PID was
reused by a new WSL process. It created no worker row and is preserved in
`stage_0a_preflight_attempt_2.json`.

After the implementation-only repair in commit `9391ea2`, PID `371` resumed
the valid partial. It retained the first 24 row objects exactly and added only
the 72 missing manifest keys. No completed row was regenerated.

Final durable state:

- status and heartbeat: `completed`;
- exit code: `0`;
- partial/result JSON: parsed;
- completed/planned rows: `96 / 96`;
- exception count: `0`;
- duplicate keys: `0`;
- missing/extra manifest keys: `0 / 0`.

## Integrity Validation

The row manifest recomputes to
`D1D1D4B1717EBA3F6D428859AC10E3AB57775D2A94BA833FD5ACAE211F8210DE`.
The candidate manifest recomputes to
`054A5B64E109A4B0ABE29BC6EB3459670FE64051E53CC27F00DFCD9E6D654051`.
Both match the final result. Partial, candidate, and expanded-manifest key sets
match exactly.

Source health, discovery/validation/confirmatory partition separation, raw
image/state/action mapping, task identity separation, exact task/phase quotas,
and manifest audits passed. All 150 sources ended successfully. Base identity
and checkpoint reload max absolute errors were `0.0`; the Base checkpoint hash
did not change. Confirmatory observations decoded and actions computed were
`0 / 0`.

Candidate generation acted safely enough to evaluate headroom:

- all Base candidates valid: yes;
- rows with a valid alternative: `90 / 96 = 0.9375`;
- rows with two unique valid chunks: `90 / 96 = 0.9375`;
- median nonzero pairwise RMS L2: `0.02089794711365724`.

## Frozen Headroom Decision

The candidate oracle found a strictly better valid alternative on `67 / 96`
rows, but most improvements were too small. Only `7 / 96 = 0.0729167` rows
met the preregistered 5% material-improvement threshold, below the required
25%. Median relative reduction over improvable rows was
`0.0166833`, below the required 5%.

The automatic 96-row expansion is therefore resolved as
`PCAV_STAGE_0A_NO_USABLE_HEADROOM`. Stage 0B, head training, validation search,
rollout, and confirmatory evaluation are forbidden for this formulation.
Loading an adapted generator or changing candidate noise would be a new method
cycle, not a PCAV rescue.

## Resource Evidence

Both PCAV attempts began after the two recorded Windows Efficiency Mode
intervals ended. Stage 0A contains no closed-loop task-success rows. Timing,
throughput, wall-clock, CUDA-memory, and resource measurements remain
diagnostic and are not needed for this decision.

## Campaign Action

Close PCAV unchanged and begin Epoch 4 Cycle 19 candidate generation. Generate
exactly three new prior-anchored candidates and select exactly one. Do not
rescue FAMR or PCAV.
