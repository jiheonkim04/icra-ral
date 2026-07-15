# NICE-VLA Stage 0B1 Pre-Extraction Protocol Clarification

Date: 2026-07-15 KST

Proposal hash:
`898BA577B38966D877E3EEC724EB98751BD8C2685CCD0BBA620EB6B6B9598C0A`.

Decision: `NICE_STAGE_0B1_TEMPORAL_VIEW_ACCOUNTING_CLARIFIED_BEFORE_DATA_ACCESS`.

The frozen Stage 0B1 protocol requires a same-episode `t+20` temporal-offset
diagnostic but its view-accounting sentence budgets only the current and
`t+10` two-camera images. The diagnostic cannot be constructed exactly from
those four views.

Before any Stage 0B1 extraction, training, validation read, or diagnostic
result, the accounting ceiling is corrected as follows:

- all 1792 natural pairs retain four image views: `7168` views;
- each of at most 320 validation-evaluation rows may encode the two cameras at
  `t+20` when that frame remains in the same episode: at most `640` views;
- corrected total ceiling: `7808` image views.

The `t+20` tokens are stored only to construct the frozen diagnostic target
`z_(t+20)-z_t`. They are prohibited mean/covariance training inputs,
calibration inputs, natural-pair scores, policy inputs, and inference inputs.

No task, demo, frame sampler, natural pair, model, objective, optimizer,
training step, mismatch family, gate, threshold, decision, or confirmatory
boundary changes. This is one mechanical pre-extraction protocol-accounting
repair, not a scientific variant or rescue.
