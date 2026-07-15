# COVI-VLA Stage 0 Invalid Run V1 Adjudication

Date: `2026-07-15 KST`

Proposal hash: `338430D2C6CF1D82410C036D79102ED3F38B2367BB35B9AE2811161698A3E621`

Decision: `IMPLEMENTATION_AUDIT_OMISSION_RERUN_REQUIRED`

The first frozen Stage 0 command completed without interruption and produced
`stage_0_result_invalid_v1.json`. It decoded the frozen `600` discovery-fit
and `400` validation records, decoded zero confirmatory-test records, and did
not run closed-loop evaluation or validation search.

The raw result is not an admissible scientific COVI adjudication. The runner
recorded parameter-group gradient norms but did not calculate or enforce the
preregistered weighted objective-specific gradient-norm ratio. Its recorded
parameter-group ratio of `44359.92745908541` cannot substitute for the missing
objective audit. The apparent `CONDITION_TOO_SEVERE_OR_NO_HEADROOM` outcome is
therefore preserved only as invalid diagnostic evidence and is not a method
kill.

The bounded repair adds the missing objective-specific calculation and the
frozen `100:1` gate. It does not change the method, data partitions, model,
checkpoint, objective coefficients, task identities, thresholds, comparator
list, occlusion transform, or sample counts. The same frozen command must be
rerun. Cached features may be reused because their identity is tied to the
unchanged proposal hash, samples, and Stage 0 configuration.
