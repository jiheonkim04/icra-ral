# TCA-Select Ambiguity Stress-Test Plan

This planning-only step defines an offline proxy test that can isolate inference-time TCA-Select gain from TCA-Map target-conditioning and LoRA adaptation gain.

It does not train, download, import heavy VLA models, load models, infer with SmolVLA, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims.

The plan is implemented by:

- `scripts\128_plan_tca_select_ambiguity_stress_test.ps1`
- `tca_map.smolvla.tca_select_ambiguity_stress_plan`

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\128_plan_tca_select_ambiguity_stress_test.ps1
```

Expected interpretation: if the plan passes, the next safe task is a CPU-only offline ambiguity stress-test runner over existing local counterfactual artifacts.
