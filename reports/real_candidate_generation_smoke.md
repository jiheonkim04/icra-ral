# Real Candidate-Generation Smoke

This report describes the bounded real candidate-generation smoke scaffold.

The smoke is an engineering check only. It is not standard success, rollout
success, learned-policy benchmark evidence, or paper-grade evidence.

It is blocked by default and may run only when all three task-local gates are
set for that task:

```text
ALLOW_REAL_CANDIDATE_GENERATION_SMOKE=1
ALLOW_HEAVY_IMPORT=1
ALLOW_SINGLE_SAMPLE_INFERENCE=1
```

Allowed scope:

- local SmolVLA checkpoint only,
- one synthetic input sample,
- one `select_action` call,
- CPU by default,
- at most four candidates,
- heatmap grid no larger than 8,
- no external verifier,
- no privileged simulator state,
- no training,
- no rollout,
- no simulator environment,
- no downloads,
- no OpenVLA-OFT,
- no paper claim.

The script is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\133_bounded_real_candidate_generation_smoke.ps1
```

Without the task-local gates, the script should refuse execution and write the
runtime reports under ignored paths:

```text
reports/real_candidate_generation_smoke_report.json
reports/real_candidate_generation_smoke_report.md
```
