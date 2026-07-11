# Official Closed-Loop Method Gap Decision

Date: 2026-07-11 KST

Final decision: `OFFLINE_ONLINE_MISMATCH_CONFIRMED`

No method is implemented in this run. Candidate directions are emitted only if a repeated, success-critical, mechanism-linked failure survives frozen-base, LoRA-seed, task, and reset explanations.

## Evidence Summary

- completed episodes: `400/400`
- successful episodes: `282/400`
- infrastructure failures: `0`
- failure annotations: `118`
- category counts: `{'ambiguous_or_unclassified': 118}`
- paired differences: `{'reset_level': {'rank4_lora_seed_11': {'tie': 82, 'loss': 9, 'win': 9}, 'rank4_lora_seed_22': {'tie': 78, 'loss': 14, 'win': 8}, 'rank4_lora_seed_33': {'tie': 84, 'loss': 12, 'win': 4}}, 'task_level': {'rank4_lora_seed_11': {'tie': 13, 'win': 3, 'loss': 4}, 'rank4_lora_seed_22': {'loss': 6, 'tie': 11, 'win': 3}, 'rank4_lora_seed_33': {'loss': 7, 'tie': 12, 'win': 1}}}`
- offline-online: `{'pearson_l2_vs_success': -0.569086, 'spearman_l2_vs_success': -0.632456, 'offline_l2': {'frozen_base': 0.085579125, 'rank4_lora_seed_11': 0.086743582, 'rank4_lora_seed_22': 0.086474081, 'rank4_lora_seed_33': 0.086918872}, 'success_rate': {'frozen_base': 0.74, 'rank4_lora_seed_11': 0.74, 'rank4_lora_seed_22': 0.68, 'rank4_lora_seed_33': 0.66}}`

## Method-Worthiness Gate

| Gate | Status |
| --- | --- |
| Repeated across tasks or reset seeds | pass |
| Materially affects success | pass |
| Not explained by one bad LoRA seed | pass |
| Not solved by simply using frozen_base | pass for repeated all-policy failures |
| Not solved by choosing a seed after outcomes | pass |
| Measurable in official closed-loop eval | pass |
| Connected to identifiable mechanism or phase | fail |
| Large enough for intervention and ablation | not established without phase evidence |

The strongest closed-loop signal is task/reset structured failure, not a mechanism-linked failure. `libero_10/task_4` is the weakest task slice at `5/20` successes, and several task/reset pairs fail for all four policies. However, all automatic phase labels remain `ambiguous_or_unclassified` because no failure videos or semantic phase traces were captured. A new method would be premature from this evidence alone.

## Final Interpretation

This run confirms that offline action L2 is not enough for LoRA seed or method selection, while also showing that a bounded visual failure-review pass is needed before any novelty gate. It does not support FCAR revival, routing/retention/prior/correction/chunking method design, or best-seed selection.

## Exact Next Step

Use the bounded review queue to inspect failed episodes with official videos enabled for the strongest repeated task/reset failures, then run a novelty/method-design gate only if the visual phase evidence supports it.
