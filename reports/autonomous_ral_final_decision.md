# Autonomous RA-L Decision

Date: 2026-07-12 KST

Current decision: `EPOCH_3_CYCLE_2_SCVC_KILLED_PIVOT_REQUIRED`

This is not a terminal state under the active governance.

Active governance: `reports/current_research_governance.md`

The prior fixed-cycle terminal stop is procedurally invalid under the current Goal. Epoch 1 is corrected as a completed related-method set that requires an Epoch 2 pivot.

Corrected adjudication:

- Cycle 1 `DICD-VLA`: `UNDERPOWERED_STAGE_A_NON_GO_ARCHIVED`
- Cycle 2 `FEDO-VLA`: `VALID_CURRENT_FORMULATION_KILL`
- Cycle 3 `GCAP-VLA`: `UNDERPOWERED_TARGET_AXIS_NON_GO_ARCHIVED`

Epoch 2 Cycle 1 `PTC-VLA` is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`: full PTC reached `0 / 10` versus frozen SmolVLA `3 / 10`, with zero exceptions and active transition mechanism.

Epoch 2 Cycle 2 `SACF-VLA` is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`: full SACF reached `0 / 10` versus frozen SmolVLA `7 / 10`, with zero exceptions and active semantic mechanism.

Epoch 2 Cycle 3 `OCFN-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: expanded Stage B completed `80` paired episodes per key policy with zero exceptions and active mechanism. OCFN full reached `26 / 80`, zero-noise SmolVLA reached `27 / 80`, and the paired upper confidence bound for full minus zero-noise was `0.0625`.

Epoch 3 Cycle 1 `CBFD-VLA` is archived as `STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE`: full CBFD reached `0 / 10` while frozen SmolVLA reached `7 / 10`, with zero exceptions and active mechanism.

Epoch 3 Cycle 2 `SCVC-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: full SCVC reached `11 / 40`, shifted frozen SmolVLA reached `20 / 40`, and the paired bootstrap CI versus shifted frozen was `[-0.425, -0.025]`.

Next action: generate exactly three Epoch 3 Cycle 3 candidates, select exactly one, and continue to implementation unless Reviewer B proves exact duplication, trivial equivalence, or hard infeasibility.
