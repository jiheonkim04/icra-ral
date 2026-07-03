# LIBERO Data Acquisition Plan

The autonomous acquisition budget is raised only for the official LIBERO dataset source already recorded in `configs/libero_robosuite_sources.yaml`.

Allowed source:

```text
https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
```

Budget:

- expected size: 100 GB from the recorded source-resolution metadata,
- single-task LIBERO-only acquisition budget: 180 GB,
- required free disk after acquisition: at least 250 GB,
- target path: `C:\assets\data\libero`,
- cache path: `C:\assets\hf_home`.

Commands:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\49_acquire_libero_data.ps1
$env:ALLOW_DOWNLOADS="1"
powershell -ExecutionPolicy Bypass -File scripts\49_acquire_libero_data.ps1 -RemoteSizeCheck -Acquire
Remove-Item Env:\ALLOW_DOWNLOADS -ErrorAction SilentlyContinue
```

The acquisition command must stop if the source becomes ambiguous, token/login/license/payment is required, expected size exceeds 180 GB, free disk after acquisition would fall below 250 GB, the download tool is missing, simulator rollout or OpenVLA-OFT becomes required, or acquisition repeatedly fails.

This plan does not authorize simulator rollout, GPU jobs outside the bounded local pilot budget, OpenVLA-OFT, paper-grade claims, token/secret access, external upload, or committing files under `C:\assets`.
