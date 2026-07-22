# Epoch 10B manifest/adjudicator erratum forensic audit

Recorded before any erratum scientific execution. This audit is outcome-blind with respect to all prospective checkpoints: no checkpoint action, checkpoint score, adapter inference result, closed-loop success label, validation result, or confirmation result was opened.

## Authority and preservation

The immutable original terminal adjudication remains `EPOCH10B_ICAE_ASSAY_INVALID_ROUTE_CLOSED` at `reports/epoch10b_assay_adjudication.json` (SHA-256 `b0538e83f14fe6764260fb570b9a6f7877d283fcc6825f181a8c1541feba231e`). The original preregistration, raw log, mechanics report, collision records, prompts, and terminal text were not modified.

The active erratum branch is `codex/epoch10b-manifest-adjudicator-erratum` at parent `bffbb29d7e638f66d4da5a2c45c84a727da7abec`. The matching remote Epoch 10 source branch also points to that commit. WSL was stopped, no scientific worker was active, host RAM was 46.7671%, system-drive free space was 316,781,596,672 bytes, and the RTX 5080 reported 2,185/16,303 MiB used at 9% utilization.

Both protected rollout ledgers reproduce their recorded hashes. The adapter panel retains 168 files / 28,417,918 bytes and every one of the 12 registered `adapter_model.safetensors` hashes matches `reports/epoch10_checkpoint_panel_manifest.json`. The append-only starting-state audit initially obtained `b96cfb...` by sorting relative paths case-sensitively. The native Windows full-path sort used by the original ledger is case-insensitive and reproduces the recorded aggregate `12b46e4003284fe995e8769fe2605a08a218467dde8667149a07941e2caf4bd5` exactly. Thus this was an encoding/order-method discrepancy, not protected evidence drift. No protected file was changed.

CHKDSK remains `UNAVAILABLE_INCONCLUSIVE`; it was not revisited and is not used as evidence of either damage or cleanliness.

## Raw-log integrity and decomposition

`runs/epoch10b_mechanics_certification/branches.jsonl` is 15,618,820 bytes, ends in LF, and has SHA-256 `a2f2992d03fae52177408f057ba311b4f522b3955d91ba78dcd74a165e55ced7`.

- 1,287/1,287 lines parse as JSON; all rows are valid and recursively finite.
- All 1,287 branch keys are unique.
- The 326-row prefix hashes to `3250ccf43919fbde7af1787a899525549d6afcc7ae5da37a74cbe93b05cd4f66`.
- The 330-row prefix hashes to `f7e642d4f72d67b03a18901e4752c323667448617c11f571333f07af13a8f950`, exactly matching the preserved probation record.
- Primary: 1,143 rows = 127 distinct state IDs x 9 controls. Every primary panel has nine controls.
- Reverse: 144 rows = 16 distinct state IDs x 9 controls. Every reverse panel has nine controls.
- Each of the nine controls occurs exactly 143 times.

The original frozen preregistration has 128 list entries but only 127 distinct state IDs. Its nominal schedule has 1,296 slots but only 1,287 distinct serialized keys. The raw log contains every one of those 1,287 distinct keys, with neither a missing nor an extra key. This is a cardinality/serialization failure, not an incomplete runner loop.

## Independent equality checks

All 16 reverse state IDs have primary counterparts. Within every state, the observed reverse control sequence is the exact reversal of its observed primary sequence. Across all 144 matched primary/reverse control rows, the following are exactly identical:

- registered-state hash;
- pre-action state hash;
- pre-action observation hash;
- executed first-action hash;
- first-step state; and
- H=4, H=8, and H=16 terminal states.

The maximum L2 difference is `0.0` for the first-step state and each terminal horizon.

The frozen even-phase nominal-twin subset contains 64 pairs, 16 per suite. All pre-action state/observation and first-action hashes match, and all first-step and H=4/8/16 state L2 differences are `0.0`. Thus all 64/64 pairs are within `1e-8`; suite counts are 16/16 in each of LIBERO Spatial, Object, Goal, and 10.

## Manifest deficiency

There are 64 suite x task x demo x phase cells. Exactly 63 have two distinct states. The sole deficiency is:

`libero_spatial|task_0|demo_8|transport_goal`

It has only `libero_spatial|task_0|demo_8|frame_61|transport_goal`. In the original preregistration, phase slots 6 (fraction 0.80, reverse duplicate true) and 7 (fraction 0.90, reverse duplicate false) both clamp to frame 61 because trajectory length 79 with horizon guard 17 has last admissible frame 61.

A preserved pre-probation, outcome-blind collision correction independently fixes the exact omitted record before its execution:

- preregistration: `runs/epoch10b_probation_recovery_archive_20260722/collision_corrected_preregistration_pre_probation.json`;
- file SHA-256: `2e5f2fe58153ed7dad17a88a11c41377275b7ebd56e7f71d989b30d073b8b6b6`;
- canonical payload SHA-256: `439e25f4ad0502aa39faa73ac51001fb728f5669527231b0ea66c84f4b5fe5da`;
- exact state: frame 60, phase index 6, registered seed `1857632994`;
- state hash: `83a7679534180c3933cfc6b46b9a7bc452b5812129fbe75c4d5910d4c30d9971`;
- expert-action hash: `681bed04690c7eea5a9c01e73d6f0978f2b0dc194b6f93743e76eaae7be5a9f7`;
- HDF5 hash: `ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead`.

The archived implementation computes all guarded phase frames and scans backward, shifting only an earlier collision to one less than its successor. For length 79 this yields fixed terminal phase frames `[60, 61]` without inspecting any score. The collision report explicitly records `outcome_input: NONE` and that the 326 mechanics rows were not used to choose the replacement. This was authorized by the original Epoch 10B pre-certification implementation-repair rule and frozen before the target panel executed.

## Why the report counted 15 instead of 16

The failure is deterministic and reproducible:

1. `_state_manifest` serialized both Spatial/demo_8 transport slots to the same frame-61 `state_id`.
2. `adjudicate_certification` built `state_by_id` with a dict comprehension. The later phase-index-7 entry overwrote the phase-index-6 entry.
3. The later entry has `reverse_order_duplicate=false`; therefore the adjudicator never joined the nine already-present reverse rows for that state.
4. `reverse_order_rows` consequently contained 15 states at every horizon even though the raw log contains 16 complete reverse panels.

Focused alternative checks exclude nearby explanations:

- Index zero: the defect does not depend on integer index zero; it is a string-key overwrite. A regression case must nevertheless prove a reverse-marked first manifest entry is retained.
- Falsey lookup: all branch-row dictionaries are truthy; `all(primary.values())` does not discard a valid zero score. A reverse L2 of zero is checked with `is not None`, so zero is retained.
- Inclusive/exclusive bounds: the guard calculation itself makes frame 61 inclusive; two fixed fractions clamp there. The missing uniqueness invariant allowed the collision.
- Deduplication: the schedule has 1,296 slots but 1,287 unique keys. The raw loader did not lose a row; the keys were already colliding before execution.
- Registered-set join: preserving the OR of duplicate manifest reverse flags makes the unchanged raw log report 16 reverse panels and maximum L2 `0.0` at every horizon.

No threshold, action, endpoint, score, state array, or raw row changes in that in-memory join audit.

## Original responsiveness and endpoint gates

The table reports the unchanged original grouped whole-demo bootstrap results. `BRC` is bounded expert recovery cost and `NGE` is native goal-error AUC.

| H | Endpoint | Harmful-minus-nominal mean [95% CI] | Responsive states (Spatial/Object/Goal/10) | Spearman | ICC | Eligible |
|---:|---|---|---|---:|---:|---:|
| 4 | BRC | 0.0226404 [0.0157762, 0.0311071] | 113 (29/24/30/30) | 1.0 | 1.0 | yes |
| 4 | NGE | 0.00650504 [-0.00305563, 0.0142350] | 43 (12/9/8/14) | 1.0 | 1.0 | no |
| 8 | BRC | 0.0155450 [0.00983623, 0.0222961] | 110 (29/22/29/30) | 1.0 | 1.0 | yes |
| 8 | NGE | 0.0111915 [0.0000861025, 0.0206506] | 45 (10/9/10/16) | 1.0 | 1.0 | yes |
| 16 | BRC | 0.00797010 [-0.00207529, 0.0179037] | 107 (26/23/30/28) | 1.0 | 1.0 | no |
| 16 | NGE | 0.0113029 [0.00329331, 0.0182372] | 53 (15/10/11/17) | 1.0 | 1.0 | yes |

With only the registered-set join corrected in memory, the reverse count becomes 16, the reverse maximum remains 0, and the twin and endpoint gates pass at H=4, H=8, and H=16. The unchanged selector would choose H=4 and BRC. That counterfactual is not certification because the original candidate set still has only 127 distinct primary states, below the explicit original target of at least 128 candidate mechanics states.

## Frozen audit conclusion

Path A alone is insufficient: the adjudicator defect is real, but 127 distinct primary states leaves an explicit frozen sample-completeness requirement unmet.

Path B is choice-free and identifiable. The preserved, hash-bound, pre-probation correction record fixes exactly one frame-60 state, its source hashes, seed, and nine primary branch keys using a deterministic score-independent collision rule. The erratum expected ledger has 1,296 unique keys and differs from the raw log by exactly those nine primary keys, with no extras. The raw log already has the required 16 reverse panels, so no reverse row is authorized.

The frozen decision is recorded separately in `reports/epoch10b_erratum_frozen_decision.json`. No scientific row may execute until that decision file is hashed.
