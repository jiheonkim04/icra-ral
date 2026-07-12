# Final Distinct Method Protocol

Date: 2026-07-12 KST
Method: `Intervention-Set Action-Chunk Fine-Tuning (ISAC-VLA)`

Protocol status: `NOT_EXECUTED_REVIEWER_KILL`

## Review Gate Outcome

The final distinct method was killed before implementation under allowed grounds:

- `NEAR_EXACT_PRIOR_ART_DUPLICATION`
- `HARD_UNAVAILABLE_RESOURCE`

Therefore no implementation protocol is activated.

## Frozen Protocol If It Had Survived

If the reviewer had not killed the method, the minimum valid protocol would have required:

- a paired intervention dataset with negative policy action chunks and positive corrective chunks;
- no privileged evaluation input;
- an OpenVLA-OFT-style continuous action-chunk fine-tuning path;
- checkpoint identity tests proving the fine-tuned policy is loaded during evaluation;
- action-change tests proving the policy distribution changes actions directly;
- an ablation without negative chunks;
- a non-intervention baseline using identical compute;
- a held-out LIBERO or robot evaluation split;
- manifest consistency tests for datasets, checkpoints, and metrics.

None of these steps was run because the proposal failed the pre-implementation reviewer gate.
