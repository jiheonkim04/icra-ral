# Autonomous RA-L Decision

Date: 2026-07-13 KST

Current decision: `EPOCH_4_CYCLE_2_CAVM_NON_GO_CONTINUE_CYCLE_3`

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

Epoch 3 Cycle 3 `PSE-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: expanded Stage B completed `400 / 400` rows with zero exceptions, full PSE reached `50 / 80`, bright-single reached `51 / 80`, and the paired CI versus bright-single was `[-0.1000, 0.0750]`.

Epoch 4 Cycle 1 `RCV-VLA` is archived as `STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: Stage 2B completed `200 / 200` episodes with zero exceptions, full RCV reached `20 / 40`, no-context ablation reached `24 / 40`, and stateless first-action reached `24 / 40`.

Epoch 4 Cycle 2 `CAVM-VLA` is archived as `STAGE_2B_EXPANDED_NON_GO_NO_THIRD_EXPANSION`: the expanded result completed `290 / 290` rows with zero exceptions, full CAVM reached `24 / 58`, nearest-success replay reached `23 / 58`, frozen SmolVLA reached `22 / 58`, success-only memory proxy reached `20 / 58`, and no-contrast ablation reached `21 / 58`.

Next action: begin Epoch 4 Cycle 3 candidate generation under the post-CAVM performance-oriented research-design gate. Do not rescue CAVM by threshold retuning, additional expansion, memory reconstruction changes, or added hyperparameter variants.
