# Real Candidate-Generation Smoke Plan

This planning-only risk gate determines whether a future bounded real candidate-generation smoke may be implemented.

It does not train, download, import heavy VLA models, load SmolVLA, infer with SmolVLA, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims.

The plan is implemented by:

- `scripts\132_plan_real_candidate_generation_smoke.ps1`
- `tca_map.smolvla.real_candidate_generation_smoke_plan`

Future smoke, if green, must be separately gated with:

- `ALLOW_REAL_CANDIDATE_GENERATION_SMOKE=1`
- `ALLOW_HEAVY_IMPORT=1`
- `ALLOW_SINGLE_SAMPLE_INFERENCE=1`

Hard boundaries:

- one sample only,
- max 4 candidates,
- heatmap grid max 8,
- max 10 minutes,
- max 14GB VRAM,
- no training,
- no rollout,
- no simulator execution,
- no OpenVLA-OFT,
- no external verifier,
- no privileged simulator state,
- no paper claim.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\132_plan_real_candidate_generation_smoke.ps1
```

Current local result:

- `real_candidate_generation_smoke_plan_passed=true`
- decision: `proceed_bounded_real_candidate_generation_smoke_implementation`
- `ready_for_real_candidate_generation_smoke_implementation=true`
- `ready_for_real_candidate_generation_smoke_execution=false`
- blockers: none
- required future gates: `ALLOW_REAL_CANDIDATE_GENERATION_SMOKE=1`, `ALLOW_HEAVY_IMPORT=1`, `ALLOW_SINGLE_SAMPLE_INFERENCE=1`
- no model load, model inference, training, rollout, GPU job, simulator execution, OpenVLA-OFT, token access, or paper claim was performed
