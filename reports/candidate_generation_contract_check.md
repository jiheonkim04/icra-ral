# Candidate-Generation Contract Check

This synthetic-tensor checker validates the candidate action heatmap contract before any real SmolVLA/TCA-Map inference is attempted.

It does not train, download, import heavy VLA models, load SmolVLA, infer with SmolVLA, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims.

The checker is implemented by:

- `scripts\131_check_candidate_generation_contract.ps1`
- `tca_map.smolvla.candidate_generation_contract_check`

It validates:

- candidate action list shape,
- low-resolution heatmap limits,
- masked heatmap alignment,
- target heatmap fields,
- metadata absence of privileged simulator state,
- TCA-Select input/output compatibility.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\131_check_candidate_generation_contract.ps1
```

Expected interpretation: if the checker passes, the next safe task is a planning-only risk gate for a separately bounded real candidate-generation smoke. This checker itself is not model inference, rollout success, or paper-grade evidence.

Current local result:

- `candidate_generation_contract_check_passed=true`
- candidate count: `4`
- heatmap grid: `8`
- selected candidate index: `0`
- selected target index: `0`
- latency: about `0.22 ms`
- max GPU memory: `0.0 MB`
- model load/inference, training, rollout, GPU jobs, simulator execution, OpenVLA-OFT, external verifier, privileged inference, and paper claims: false
