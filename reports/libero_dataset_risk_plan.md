# LIBERO Dataset Risk Plan

This plan adds a planning-only gate before any LIBERO or LIBERO-CF-style dataset acquisition, metadata setup, tiny subset creation, training, or rollout work.

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\42_plan_libero_dataset_risk.ps1
```

The planner writes ignored runtime reports:

```text
reports\libero_dataset_risk_report.json
reports\libero_dataset_risk_report.md
```

It does not download datasets, run GPU jobs, train, rollout, import simulators or heavy VLA models, access tokens, execute OpenVLA-OFT, or make paper-grade claims.

Proceed only if either:

- a local tiny LIBERO/LIBERO-CF-style subset already exists under `LIBERO_DATA_ROOT`, or
- a future acquisition task has an official/documented source, known expected size, enough disk margin, no token/login/payment/license click-through, and an approved local asset root.

Paper-grade standard success still requires simulator rollout evidence later. Offline subset metrics remain offline proxy evidence only.
