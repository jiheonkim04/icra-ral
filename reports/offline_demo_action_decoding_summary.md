# Offline Demonstration Action Decoding Summary

This report summarizes the one-sample offline SmolVLA action-decoding diagnostic and converts it into a conservative rollout-scaling decision.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\107_summarize_offline_demo_action_decoding.ps1
```

The summary reads only `reports\offline_demo_action_decoding_report.json`. It does not load models, run inference, create simulator environments, rollout, train, download, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

Interpretation:

- `offline_alignment_signal=weak`: do not scale rollout; inspect VLM loading policy, checkpoint provenance, and action normalization.
- `offline_alignment_signal=moderate` or `strong`: still not rollout evidence; plan a tiny repeated offline decoding check before any rollout decision.
