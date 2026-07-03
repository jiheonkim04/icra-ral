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

The planner now defaults to the official/documented LIBERO dataset source:

```text
https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
```

The full official dataset is expected to be about 100 GB, which exceeds the current 80 GB autonomous task budget. Therefore the default decision should remain `stop` for full dataset acquisition unless a later risk policy changes the budget or a smaller documented subset is selected.

Proceed only if either:

- a local tiny LIBERO/LIBERO-CF-style subset already exists under `LIBERO_DATA_ROOT`, or
- a future acquisition task has an official/documented small subset source, known expected size within budget, enough disk margin, no token/login/payment/license click-through, and an approved local asset root.

Use the source-resolution planner before setup:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\45_resolve_libero_robosuite_sources.ps1
```

Paper-grade standard success still requires simulator rollout evidence later. Offline subset metrics remain offline proxy evidence only.
