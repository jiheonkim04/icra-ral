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

The full official dataset is expected to be about 100 GB. The current policy has a LIBERO-only acquisition exception: this source may use a 180 GB task budget only if at least 250 GB disk remains after acquisition and no token/login/payment/license click-through gate appears. The planner remains dry-run only and does not acquire data.

Proceed only if either:

- a local tiny LIBERO/LIBERO-CF-style subset already exists under `LIBERO_DATA_ROOT`, or
- the official LIBERO acquisition gate in `scripts\49_acquire_libero_data.ps1` reports a green risk assessment for `https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets`, known expected size within the 180 GB LIBERO-only budget, at least 250 GB disk remaining after acquisition, no token/login/payment/license click-through, and the approved target root `C:\assets\data\libero`.

Official LIBERO data acquisition command, only after the dry-run risk report is green:

```powershell
$env:ALLOW_DOWNLOADS="1"
powershell -ExecutionPolicy Bypass -File scripts\49_acquire_libero_data.ps1 -RemoteSizeCheck -Acquire
Remove-Item Env:\ALLOW_DOWNLOADS -ErrorAction SilentlyContinue
```

Use the source-resolution planner before setup:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\45_resolve_libero_robosuite_sources.ps1
```

For a safe next step without downloading demonstrations, run the metadata-only subset builder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\47_build_libero_metadata_subset.ps1
```

That builder reads BDDL/task metadata from the local LIBERO source checkout only. It can validate target/counterfactual split plumbing, but `ready_for_real_dataset_interface_smoke` remains false until actual demo files are present under `LIBERO_DATA_ROOT`.

To check whether a tiny local data file is structurally usable for offline interface smoke, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\48_plan_libero_offline_interface_smoke.ps1
```

The expected current decision is `stop` because only the no-full-dataset marker exists under `LIBERO_DATA_ROOT`.

Paper-grade standard success still requires simulator rollout evidence later. Offline subset metrics remain offline proxy evidence only.
