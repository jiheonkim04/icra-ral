# LIBERO/RoboSuite Source Resolution

This report records the current official-source decision for the real LIBERO/RoboSuite setup gate.

## Official Sources

- LIBERO code: `https://github.com/Lifelong-Robot-Learning/LIBERO.git`
- RoboSuite code: `https://github.com/ARISE-Initiative/robosuite.git`
- LIBERO full demonstrations: `https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets`

The LIBERO GitHub repository documents the dataset download script and links to the Hugging Face dataset. The Hugging Face dataset card reports total file size as 100 GB. The current policy raises the autonomous budget for this official LIBERO data source only: it may proceed up to 180 GB if at least 250 GB free disk remains after acquisition and no token/login/payment/license click-through is required.

## Current Decision

Repo source setup is allowed after a green risk assessment:

- shallow clone LIBERO code into `C:\assets\repos\LIBERO`,
- shallow clone RoboSuite code into `C:\assets\repos\robosuite`,
- create `C:\assets\data\libero` as a path-ready data root only.

Full LIBERO dataset acquisition is allowed only through the dedicated LIBERO acquisition gate after a green risk report:

- expected size: 100 GB,
- LIBERO-only task budget: 180 GB,
- minimum disk remaining after acquisition: 250 GB,
- target path: `C:\assets\data\libero`,
- no simulator execution,
- no rollout,
- no paper-grade claim.

## Commands

Planning-only source resolution:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\45_resolve_libero_robosuite_sources.ps1
```

Bounded source checkout setup, only after the planner reports `ready_for_repo_setup=true`:

```powershell
$env:ALLOW_DOWNLOADS="1"
powershell -ExecutionPolicy Bypass -File scripts\46_prepare_libero_robosuite_sources.ps1
Remove-Item Env:\ALLOW_DOWNLOADS -ErrorAction SilentlyContinue
```

Then rerun:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\42_plan_libero_dataset_risk.ps1
powershell -ExecutionPolicy Bypass -File scripts\43_plan_simulator_readiness.ps1
```

Official LIBERO data acquisition, only after a green dry-run risk report:

```powershell
$env:ALLOW_DOWNLOADS="1"
powershell -ExecutionPolicy Bypass -File scripts\49_acquire_libero_data.ps1 -RemoteSizeCheck -Acquire
Remove-Item Env:\ALLOW_DOWNLOADS -ErrorAction SilentlyContinue
```

## Next Gate

Continue to tiny real/offline dataset interface smoke only after a real tiny subset or documented metadata-only subset exists under `LIBERO_DATA_ROOT`.

Do not run simulator import, render smoke, rollout, policy evaluation, training on real data, OpenVLA-OFT, or paper-grade claims from this source-resolution step.
