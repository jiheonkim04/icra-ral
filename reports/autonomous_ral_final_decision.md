# Autonomous RA-L Decision

Date: 2026-07-14 KST

Current decision: `EPOCH_4_CYCLE_5_RAC_VALIDATION_SELECTED_STAGE_A_PENDING`

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

Epoch 4 Cycle 3 selected and preregistered `FANG-VLA`. Proposal hash: `6837DBA2A1307F7C9938FA9F5463ED483907AF3C168F1C0514F6E281804E859B`.

The development audit passed and the calibrated validation search selected `fang_c01`. The uncalibrated gate failure is preserved as a negative validation result. Stage A completed `50 / 50` episodes with all five policies tied at `3 / 10`.

Stage B completed `200 / 200` episodes with zero exceptions. Full FANG reached `11 / 40`, while frozen SmolVLA reached `16 / 40`, AFIL local proxy reached `15 / 40`, nearest-success replay reached `14 / 40`, and the no-failure ablation also reached `11 / 40`. Full-minus-base paired delta was `-0.125` with CI `[-0.250, 0.000]`; full was exactly tied with the key ablation.

Final FANG decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`. Do not rescue this formulation.

Epoch 4 Cycle 4 selected and preregistered `EvoState-VLA`. Proposal hash: `A44ED68CC8E1F296DB8B0B3E16FF84D7D5BBE684EAF63EAE29E7CC91DCFD93C9`.

Stage 0 stopped before rollout as `AUDIT_STOP_DESIGN_FAILURE`: the full transition model improved only `0.024689` over an actionless model, below the preregistered `0.05` threshold.

Epoch 4 Cycle 5 selected and preregistered `RAC-VLA`, a Reflective VLA-anchored frozen-policy action-consequence calibration method. Proposal hash: `71ABA93E37FC725C1A2E5EAE6E1461BC77AACDAFF9B0711C37F17D5C0AB0902F`.

RAC Stage 0 passed without rollout: full action-consequence validation accuracy `0.585745` beat action-only `0.368496` and no-consequence `0.374483`, with margin `0.211262`; clean action delta p95 was `0.0`. The six-config validation search selected `rac_h4_a0.05` with score `0.508926`.

Next action: run RAC-VLA Stage A on the frozen five-policy shifted-condition manifest. Do not retune RAC using Stage A outcomes.
