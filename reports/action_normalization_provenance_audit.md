# Action Normalization Provenance Audit

This report-only step audits whether the local SmolVLA action normalization statistics and action dimensionality are compatible with the local LIBERO offline diagnostics.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\118_audit_action_normalization_provenance.ps1
```

Scope:

- reads `config.json`,
- reads `policy_preprocessor.json`,
- reads `policy_postprocessor.json`,
- reads processor safetensors with `safetensors.safe_open`,
- reads existing offline diagnostic runtime reports,
- compares checkpoint action-stat scale and prefixes against local LIBERO action previews,
- performs no downloads,
- performs no installs,
- imports no heavy VLA models,
- loads no model,
- performs no inference,
- runs no training,
- runs no simulator or rollout,
- uses no GPU job,
- executes no OpenVLA-OFT,
- accesses no tokens,
- makes no paper-grade claim.

Expected interpretation:

- If action stats are robot/provenance-specific and not LIBERO-scale, learned-policy rollout scaling remains blocked.
- If the 6D policy action convention remains unresolved against the 7D LIBERO expert-action convention, rollout scaling remains blocked.
- If clipping persists, the next safe step is a planning-only action-stat mapping or checkpoint/task-provenance correction plan.

Current local result:

- audit passed,
- decision: `no_go_rollout_scaling`,
- action stat prefixes: `so100`, `so100-blue`, and `so100-red`,
- action mean max abs: `125.720543`,
- action std max: `59.359951`,
- local LIBERO expert action preview max abs: `1.0`,
- policy action shape: `[6]`,
- local LIBERO/action-adapter convention: 7D,
- clipped decoded action values total: `3`,
- rollout scaling ready: false,
- benchmark claim ready: false,
- paper claim ready: false.

Interpretation: the local learned-policy rollout blocker is now a strong action-stat/checkpoint-provenance mismatch risk, not merely a VLM-loading issue. Do not scale learned-policy rollouts until an action-stat mapping or checkpoint/task-provenance correction plan is created and validated.
