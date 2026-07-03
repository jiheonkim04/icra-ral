# Head-Only Tiny Comparison

## Purpose

This bounded local pilot report compares the existing ActionMap head-only and TCA-Map head-only tiny smoke outputs.

It is an offline proxy diagnostic only. It is not standard success, not rollout success, not paper-grade evidence, and not a SOTA claim.

## Command

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\36_compare_head_only_tiny_pilot.ps1
```

It writes ignored runtime reports:

```text
reports\head_only_tiny_comparison_report.json
reports\head_only_tiny_comparison_report.md
```

## Bounds

The comparison is standing-approved because it:

- reads an existing bounded tiny smoke report,
- uses offline proxy metrics only,
- does not download assets,
- does not run GPU jobs,
- does not train,
- does not import heavy VLA models,
- does not run inference,
- does not rollout or execute simulators,
- does not execute OpenVLA-OFT,
- does not make paper claims.
