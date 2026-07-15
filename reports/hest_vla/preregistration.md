# HEST-VLA Preregistration

Date: 2026-07-15 KST

Proposal hash:
`E56B4717BDF949E1A4371457058DFC662E0D79C70D9E2FBEF35A5415FD0F0527`.

Decision: `HEST_PREREGISTERED_STAGE_0A_ALLOWED`

## Frozen Method

- input: official postprocessed SmolVLA chunk `[50,7]`;
- arm dimensions: `0..5`;
- gripper dimension: `6`;
- cumulative second-difference objective;
- `lambda = 4.0`;
- endpoint constraints at indices `0` and `49`;
- exact gripper copy;
- whole-chunk Base fallback;
- no clipping;
- alpha search set `{0.25,0.50,1.00}` only after Stage 0B;
- smaller-alpha tie break.

No method repair, coefficient extension, alternate spline family, task swap, or
threshold change is allowed within HEST after execution begins.

## Frozen Source Partitions

Tasks, in this order:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

Discovery demonstrations: IDs `0..7` per task.

Validation demonstrations: IDs `8..9` per task.

For each demonstration, eligible starts satisfy `start + 50 <= episode_length`.
Sort eligible starts numerically and select four quantile-indexed starts at
fractions `{0.0, 1/3, 2/3, 1.0}`, with nearest integer and stable first-use
deduplication. If fewer than four unique starts exist, the source fails; no
replacement demo or task is allowed.

Expected fixed window count:

- discovery: `4 tasks * 8 demos * 4 windows = 128`;
- validation: `4 tasks * 2 demos * 4 windows = 32`;
- total: `160`.

Window key:

`(partition, suite, task_identity, source_path, demo_id, start, stop)`.

Duplicate, missing, extra, and partition-overlap counts must all be zero.

Confirmatory resets and task outcomes are forbidden in Stage 0.

## Stage 0A Gates

Stage 0A is a CPU-only source, algebra, action-validity, and headroom audit.

All hard gates must pass:

1. proposal hash recomputes exactly;
2. all `160` manifest keys and rows persist;
3. duplicate/missing/extra/overlap counts are zero;
4. every source and transformed action is finite and shape-valid;
5. every arm dimension has discovery range greater than `1e-8`;
6. at least `8` validation chunks contain any adjacent gripper-command change
   greater than `1e-8`;
7. HEST float64 endpoint error is at most `1e-8`;
8. HEST first-action error is at most `1e-8`;
9. HEST gripper maximum absolute difference is exactly `0.0`;
10. HEST action support validity is `1.0` with no clipping;
11. HEST acts above `1e-8` arm maximum difference on at least `80%` of
    validation chunks at `alpha = 1`;
12. median validation cumulative-arm second-difference energy reduction is at
    least `10%` at `alpha = 1`;
13. HEST is not exactly equivalent within `1e-10` to SplineProxy, NoEndpoint,
    or MovingAverage on all validation chunks;
14. deterministic disk round-trip output difference is at most `1e-12`;
15. exception count is zero.

Decision if all pass: `HEST_STAGE_0A_PASS_STAGE_0B_ALLOWED`.

Failure mapping:

- malformed or inadequate source/transition coverage: `DATA_FAILURE`;
- solver, invariant, support, persistence, or determinism defect:
  `IMPLEMENTATION_FAILURE`;
- nonacting or insufficient smoothness opportunity: `NO_HEADROOM`;
- exact comparator equivalence with valid implementation: `DESIGN_FAILURE`.

None is a scientific kill.

## Stage 0B Direct Replay Gate

Stage 0B is permitted only after Stage 0A passes unchanged. It replays all `32`
validation windows from their exact official simulator states for:

- original expert diagnostic reference;
- SplineProxy;
- HEST at alpha `1.0` for mechanism audit only;
- NoEndpoint;
- MovingAverage.

No alpha selection occurs from Stage 0B. It reports robot-state trajectory
error, object-state trajectory error, terminal deviation, gripper-event
strata, action validity, and exceptions.

Stage 0B passes only if:

- exact original expert replay is valid for every row;
- all transformed runs are synchronous and timeout-free;
- HEST action validity remains `1.0`;
- HEST median terminal controller-state error is no worse than both
  SplineProxy and MovingAverage;
- HEST median object-state error is no worse than both comparators;
- HEST retains at least the Stage 0A frozen `10%` arm-energy reduction;
- no comparator is exactly equivalent to HEST;
- exceptions, duplicate keys, missing keys, and extras are zero.

Failure is data, implementation, no-headroom, or design failure. Do not proceed
to closed-loop validation after a failed gate.

## Validation Search

Only after Stage 0B passes, run the exact three alpha values on validation reset
identities `20262101..20262110`. Use the proposal's weighted score and smaller
alpha tie rule. Save every result and freeze the selected alpha and code hash.

## Confirmatory Manifest

Stage A reset identities: `20262111..20262120`.

Stage B adds: `20262121..20262150`.

Policies:

1. Base;
2. SplineProxy;
3. HEST;
4. NoEndpoint;
5. MovingAverage.

No policy, task, reset, threshold, alpha, queue rule, or outcome interpretation
may change after confirmatory testing begins.

## Resource And Runtime Governance

Before every long WSL launch:

- inspect newest PID, heartbeat/status, partial/result, logs, and exit code;
- check Linux worker liveness;
- parse partial JSON;
- compare completed and planned keys;
- count exceptions, duplicates, missing keys, and extras;
- never duplicate a live or completed worker;
- resume only missing keys after a dead worker with valid partial data.

Record any Windows gaming or Efficiency Mode interval. Exclude overlapping or
overlap-unknown timing, throughput, wall-clock, and resource-utilization
evidence. Closed-loop success rows survive only after the frozen synchronous,
exception, identity, action-semantics, duplicate, and manifest checks.
