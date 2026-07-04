# Offline TCA-Select Ambiguity Stress Test

This CPU-only runner creates ambiguous target/action candidate heatmaps from existing local LIBERO counterfactual HDF5 snippets.

It is offline proxy evidence only. It is not standard success, not rollout success, and not paper-grade evidence.

The runner is implemented by:

- `scripts\129_run_tca_select_ambiguity_stress_test.ps1`
- `tca_map.smolvla.tca_select_ambiguity_stress`

Policy:

- uses existing local counterfactual split artifacts,
- generates synthetic ambiguous candidates without model loading,
- compares Distributional TCA-Select against a top-heatmap baseline,
- reports wrong-target proxy, action L1, target consistency, condition sensitivity, candidate diversity, latency, and GPU memory,
- does not train, download, import heavy VLA models, load SmolVLA, infer with SmolVLA, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\129_run_tca_select_ambiguity_stress_test.ps1
```

Current local result:

- `tca_select_ambiguity_stress_passed=true`
- record count: `16`
- selected wrong-target proxy rate: `0.0`
- top-heatmap wrong-target proxy rate: `1.0`
- wrong-target proxy delta vs top heatmap: `-1.0`
- selected action L1: `0.0`
- top-heatmap action L1: `0.164299`
- action L1 delta vs top heatmap: `-0.164299`
- mean latency: `0.428231 ms`
- max GPU memory: `0.0 MB`

Interpretation: this provides selection-specific offline proxy evidence that Distributional TCA-Select can reject target-inconsistent high-logit candidates in the synthetic ambiguity stress setting. It is still not standard success, not rollout success, and not paper-grade evidence.
