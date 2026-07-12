# Epoch 2 Cycle 1 PTC-VLA Adjudication

Date: 2026-07-12 KST

Method: `PTC-VLA`

Proposal hash: `15A3027E02DFE46EF2B56461A245307E9588F13431A1C92952DDD76683964CC7`

## Result

Decision: `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`

This is a valid current-formulation kill under `reports/current_research_governance.md`.

Reason:

- Stage A completed `50 / 50` paired episodes with zero exceptions.
- `ptc_full` reached `0 / 10`, task-balanced success `0.00`.
- strongest baseline was `frozen_smolvla` at `3 / 10`, task-balanced success `0.30`.
- the full method was exactly `0.30` absolute task-balanced success below the strongest baseline.
- mechanism was active: mean transition-context norm `0.065772`, mean full-vs-ablation action delta `0.756346`.

This satisfies the Stage A permanent-kill rule: full method at least 30 absolute percentage points below the strongest baseline or key ablation.

## Archived Evidence

- candidate generation: `reports/epoch_2_candidate_generation.md`
- proposal: `reports/ptc_vla/researcher_proposal.md`
- reviewer attack: `reports/ptc_vla/reviewer_attack.md`
- preregistration: `reports/ptc_vla/preregistration.md`
- synthetic initial failed run: `reports/ptc_vla/synthetic_result_initial_fail.json`
- synthetic repaired pass: `reports/ptc_vla/synthetic_result.json`
- real trace training: `reports/ptc_vla/real_trace_train_result.json`
- Stage A result: `reports/ptc_vla/stage_a_result.json`
- Stage A partial checkpoint: `reports/ptc_vla/stage_a_partial_result.json`

## No Rescue

Do not rescue PTC-VLA through hidden-size tuning, phase-bin tuning, alternate mean-action smoothing, a renamed state-only MLP, or another direct policy-input-state transition head on the same tasks.

## Next Action

Pivot to Epoch 2 Cycle 2 with a method that changes at least two core dimensions relative to PTC-VLA and the Epoch 1 routes.
