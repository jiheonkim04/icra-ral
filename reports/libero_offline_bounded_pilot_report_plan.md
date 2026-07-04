# LIBERO Offline Bounded Pilot Report Plan

This report consolidates the local LIBERO offline proxy ladder:

- offline interface smoke,
- HDF5-backed counterfactual split,
- ActionMap vs TCA-Map head-only proxy comparison,
- required ActionMap + LoRA vs TCA-Map + LoRA proxy comparison.

It reads existing runtime reports only. It does not download, train, use GPU, import heavy VLA models, load models, run inference, execute simulators, run rollouts, access tokens, execute OpenVLA-OFT, or make paper-grade claims.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\54_generate_libero_offline_bounded_pilot_report.ps1
```

Runtime outputs:

- `reports\libero_offline_bounded_pilot_report.json`
- `reports\libero_offline_bounded_pilot_report.md`

