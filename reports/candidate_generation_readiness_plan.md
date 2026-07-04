# Learned-Policy Candidate-Generation Readiness Plan

This planning-only step defines what must be true before attempting to generate real candidate action heatmaps from the local SmolVLA/TCA-Map stack.

It does not train, download, import heavy VLA models, load SmolVLA, infer with SmolVLA, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims.

The plan is implemented by:

- `scripts\130_plan_candidate_generation_readiness.ps1`
- `tca_map.smolvla.candidate_generation_readiness_plan`

Purpose:

- bridge offline TCA-Select ambiguity evidence to a future bounded candidate-generation contract,
- separate synthetic/offline candidate stress evidence from real learned-policy candidate generation,
- require a contract checker before any model inference,
- keep rollout and paper claims blocked.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\130_plan_candidate_generation_readiness.ps1
```

Expected interpretation: if the plan passes, the next safe task is a synthetic-tensor candidate-generation contract checker. Real model inference remains blocked until a separate risk-gated smoke is implemented.

Current local result:

- `candidate_generation_readiness_plan_passed=true`
- `ready_for_candidate_generation_contract_checker=true`
- `ready_for_real_candidate_generation_smoke_plan=true`
- `ready_for_real_candidate_generation_smoke_execution=false`
- prior SmolVLA load-only smoke: passed
- prior single-sample interface smoke: passed
- prior feature-cache eval smoke: valid
- model load/inference, training, rollout, GPU jobs, simulator execution, OpenVLA-OFT, and paper claims: false
