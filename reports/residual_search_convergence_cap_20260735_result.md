# Residual-Search Convergence Cap

Decision: `NATURAL_RESET_SEARCH_FINAL_BUDGET_PREREGISTERED`

The residual-search convergence steer caps the natural-reset first-prior mining axis. The `libero_object` identity `20260735` worker was already in flight when the steer was read; I let it finish under the frozen protocol and recorded its 10/10 X-VLA saturation separately.

Final natural-reset budget:

- Exactly one remaining first-prior full-suite scan: `libero_spatial`, reset identity `20260735`, tasks `0..9`.
- Episode cap after this preregistration: `10` official-prior task episodes.
- No additional natural-reset identity sweeps after that scan.
- No task5 candidate regeneration, SGL-XVLA reopening/renaming, automatic OCR-XVLA reopening, Ours rollout, training, optimizer step, or checkpoint write.

End-of-budget rule:

- Return `REPEATED_RESIDUAL_FOUND` only if a claim-relevant residual survives Base, first prior, second prior, repeated independent identities, and recoverable headroom, excluding already exhausted task5/SGL/OCR paths.
- Otherwise return `NATURAL_RESET_SEARCH_SATURATED`, close the natural-reset search axis, and move to either a preregistered claim-specific condition or an official-prior ecosystem.

OCR clarification:

- OCR-XVLA is not a scientific method kill from the existing evidence.
- Its current status is an observability-data question.
- If the natural-reset axis saturates, the next OCR decision must be either `OCR_TRIGGER_TRACE_ACQUISITION_PREREGISTERED` or `OBSERVABILITY_DATA_BLOCKED_ARCHIVED`.
- Any trace acquisition must use only legal inference-time fields: per-step RGB, proprioception, issued action chunks, timestamps/chunk indices, frozen prior identity, and task/reset identity.
- Reward, success/done oracle, simulator object state, privileged contact/pose, and future observations remain forbidden as inference features.
