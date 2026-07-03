# Local Pilot Status

## Purpose

This report generator consolidates existing bounded local pilot outputs into one status artifact.

It is summary-only. It does not create new empirical evidence, standard success, rollout success, paper-grade evidence, or SOTA claims.

## Command

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\39_generate_local_pilot_status.ps1
```

It writes ignored runtime reports:

```text
reports\local_pilot_status_report.json
reports\local_pilot_status_report.md
```

## Bounds

The generator is standing-approved because it:

- reads existing local JSON reports,
- uses offline proxy status only,
- does not download assets,
- does not run GPU jobs,
- does not train,
- does not import heavy VLA models,
- does not load models,
- does not run model inference,
- does not rollout or execute simulators,
- does not execute OpenVLA-OFT,
- does not make paper claims.

The generator refuses execution gates, including `ALLOW_TINY_TRAINING`, because it is summary-only.
