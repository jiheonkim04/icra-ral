# Official SmolVLA LoRA Drift Audit

Date: 2026-07-10 KST

Final decision: `PROTOCOL_DRIFT_FOUND`
Historical reproduction tolerance preserved: `0.002`

## Execution Boundary

- experiments happened: `True`
- training happened: `False`
- optional single-seed probe ran: `False`
- GPU used for evaluation: `True`
- downloads happened: `False`
- rollout happened: `False`
- simulator dependency install happened: `False`
- OpenVLA-OFT happened: `False`
- FCAR revived: `False`

## Main Findings

- split/frame/label/frozen-base alignment is proven across old and regenerated artifacts
- current persisted checkpoint bundles are complete and checksum verified
- repeated disk evaluation is deterministic under the audit's fixed evaluation seed
- the saved regenerated artifact metrics do not exactly match this fixed-seed disk re-evaluation, so evaluation RNG state is part of the protocol identity
- historical adapter weights and complete training-state identity were not saved
- a real protocol difference exists between historical in-memory evaluation and regenerated persisted PEFT reload evaluation

## Root Cause

The metric drift is best explained as protocol drift in the LoRA prediction path: `5d48b1e` evaluated the trained in-memory policy and did not persist/reload the adapter; `15649d6` assigns the PEFT wrapper return, saves the adapter, reloads it through `PeftModel.from_pretrained`, then evaluates that disk identity. The fixed-seed disk re-evaluation is internally repeatable, but it does not exactly reproduce the saved regenerated artifact metrics, which shows that evaluation RNG state was also part of the unpinned protocol identity. Because the historical adapter weights were never saved, the old learned policy identity cannot be recovered exactly.

## Exact Next Step

Fix or explicitly adjudicate the PEFT in-memory versus persisted-reload protocol difference and evaluation RNG-state policy before canonicalizing or rolling out.
