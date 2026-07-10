# Official SmolVLA LoRA Drift Decision

Date: 2026-07-10 KST

Final decision: `PROTOCOL_DRIFT_FOUND`

Decision rationale:

- the current persisted checkpoints are complete and checksum verified
- fixed-seed repeated disk evaluation did not show material evaluation nondeterminism
- saved regenerated artifact metrics did not exactly match fixed-seed disk re-evaluation metrics, so evaluation RNG state was unpinned
- split, labels, frozen/base predictions, metric protocol, and static-alpha grid align
- nevertheless, the old and regenerated LoRA prediction protocols differ
- old adapter weights and complete old RNG/sample-order identity were not preserved

Therefore the regenerated persisted checkpoints are not accepted as canonical in this audit. They can only become canonical after the PEFT protocol drift and evaluation RNG-state policy are fixed or after an explicit re-baselining decision that preserves old metrics as historical.

Exact next step:

Fix or explicitly adjudicate the PEFT in-memory versus persisted-reload protocol difference and evaluation RNG-state policy before canonicalizing or rolling out.
