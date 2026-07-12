# Epoch 2 Cycle 2 SACF-VLA Adjudication

Date: 2026-07-12 KST

Method: `SACF-VLA`

Proposal hash: `1C43D99A42AD97C29C1BDBDED1AB1326214C8FF0F514F79309266738C5FD1A20`

## Result

Decision: `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`

This is a valid current-formulation kill under `reports/current_research_governance.md`.

Reason:

- Stage A completed `50 / 50` paired episodes with zero exceptions.
- `sacf_full` reached `0 / 10`, task-balanced success `0.00`.
- strongest baseline was `frozen_smolvla` at `7 / 10`, task-balanced success `0.70`.
- the full method was `0.70` absolute task-balanced success below the strongest baseline.
- the full method also had `0 / 10` while a paired baseline had at least `4 / 10`.
- mechanism was active: mean semantic component norm `1.709826`, mean full-vs-plain action delta `0.429388`.

This satisfies two Stage A permanent-kill rules:

1. full method at least 30 absolute percentage points below the strongest baseline or key ablation;
2. full method `0 / 10` while a paired baseline has at least `4 / 10`.

## Archived Evidence

- candidate generation: `reports/epoch_2_cycle_2_candidate_generation.md`
- proposal: `reports/sacf_vla/researcher_proposal.md`
- proposal hash: `reports/sacf_vla/proposal_hash.txt`
- reviewer attack: `reports/sacf_vla/reviewer_attack.md`
- researcher rebuttal: `reports/sacf_vla/researcher_rebuttal.md`
- preregistration: `reports/sacf_vla/preregistration.md`
- protocol: `reports/sacf_vla/prototype_protocol.md`
- initial synthetic failed smoke: `reports/sacf_vla/synthetic_result_initial_fail.json`
- synthetic measurement repair: `reports/sacf_vla/synthetic_measurement_repair.md`
- repaired synthetic result: `reports/sacf_vla/synthetic_result.json`
- real-demo training: `reports/sacf_vla/real_demo_train_result.json`
- Stage A partial checkpoint: `reports/sacf_vla/stage_a_partial_result.json`
- Stage A result: `reports/sacf_vla/stage_a_result.json`

## No Rescue

Do not rescue SACF-VLA through prefix-length tuning, hidden-size tuning, different CAG guidance scale, phase-bin tuning, alternate same-demo BC prefixes, or a renamed same-scene semantic prefix on the same two tasks.

The failed assumption is specific: local demonstration-trained semantic prefixes disrupt closed-loop behavior even when the learned semantic component is active. Frozen SmolVLA was much stronger, especially on the object task.

## Next Action

Pivot to Epoch 2 Cycle 3 with a method that changes at least two core dimensions relative to both `PTC-VLA` and `SACF-VLA`.
