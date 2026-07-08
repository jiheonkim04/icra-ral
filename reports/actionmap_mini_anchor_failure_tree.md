# ActionMap Mini-Anchor Failure Tree

Date: 2026-07-08

## Root Outcome

`KILL_ACTIONMAP_ANCHOR`

The local learned heatmap/candidate head failed the hard gate on real LIBERO/HDF5 held-out action metrics.

## Failure Tree

1. Was there a real local LIBERO/HDF5 metric?
   - Yes.
   - Evidence: `8` demos, `1008 / 432` train/eval records, deterministic per-demo time holdout.
   - Result: continue the gate.

2. Was the candidate space itself hopeless?
   - No.
   - Evidence: oracle nearest-candidate action L2 `0.065653208`, far below mean-action L2 `0.466767673`.
   - Interpretation: candidate-space headroom exists, but the oracle is invalid as method evidence.

3. Did the learned ActionMap-style head beat mean action?
   - No.
   - Evidence: ActionMap-style action L2 `0.529931357` versus mean-action action L2 `0.466767673`.
   - Consequence: kill criterion triggered.

4. Did the learned ActionMap-style head beat cheap learned baselines?
   - No.
   - Evidence: simple MLP action L2 `0.501926707` matched or beat ActionMap-style action L2 `0.529931357`.
   - Consequence: kill criterion triggered.

5. Did the candidate selector maintain useful diversity?
   - No.
   - Evidence: candidate top1 `0.018518519`; unique translation/rotation/gripper bins `5 / 1 / 2`.
   - Consequence: candidate collapse triggered.

6. Can Target-Grounded ActionMap proceed from this result?
   - No.
   - Reason: the base local heatmap substrate failed before target grounding. Target grounding would be a new method on top of a failed local anchor.

## Failure Diagnosis

The strongest local positive signal was oracle headroom, not learned decoder quality. The decisive failure was the gap between an oracle candidate assignment and the collapsed learned selector.

## Stop Rule

No further local proxy heatmap variants should be tried. Revival requires official ActionMap reproduction or a stronger non-collapsed heatmap implementation that passes the same simple-baseline gate before any target-grounded method work.
