# NICE-VLA Stage 0B1 Adjudication

Date: 2026-07-15 KST

Proposal hash:
`898BA577B38966D877E3EEC724EB98751BD8C2685CCD0BBA620EB6B6B9598C0A`.

Raw worker decision: `NICE_STAGE_0B1_IMPLEMENTATION_FAILURE`.

Adjudicated decision:
`NICE_STAGE_0B1_DATA_FAILURE_COLLAPSED_ACTION_REGIME_CONTRAST`.

## Runtime And Manifest Audit

Linux PID `403` and its Windows WSL host are dead. Wrapper exit code is `1`.
The atomic partial parses and contains all `1792 / 1792` planned keys. Manifest
and partial hashes match; duplicate, missing, extra, and repeated completed key
counts are all zero. No resume or rerun is authorized because there are no
missing keys.

## Classification

The runner raised while requiring an action-regime diagnostic for
`libero_object/task_3`. Independent recomputation from the frozen memmaps gives
discovery deadband `2.0` and validation evaluation counts:

- `libero_10/task_5`: `[78,2]`;
- `libero_goal/task_5`: `[79,1]`;
- `libero_object/task_3`: `[80,0]`;
- `libero_spatial/task_3`: `[80,0]`.

The preregistration requires noncollapsed gripper transition contrast. The
Stage 0B1 diagnostic also requires action-regime rows. Two tasks have no
positive regime at all, so the legal supervision is collapsed. The correct
classification is `DATA_FAILURE`, not a closed-loop scientific result.

The runner should have applied this data gate before mean/covariance training;
that ordering defect wasted development compute but does not create a valid
mechanism result or justify a rerun. Checkpoints and rank basis are retained
only as failed-cycle artifacts. No unpersisted mean, coverage, or AUROC value is
reconstructed or interpreted.

## Evidence Boundary

Validation pair records read: `640`. Confirmatory records, task outcomes,
rewards, dones, reset identities, and simulator rollouts read: zero. No
validation search or confirmatory tuning occurred. Timing and utilization are
not paper evidence.

## Decision

Do not change the deadband, sampler, validation tasks, action-regime mismatch,
or data quotas. Do not rerun Stage 0B1 or launch Stage 0B2. NICE is closed
without scientific kill or rescue. Continue automatically to Cycle 21 and
generate exactly three new candidates.
