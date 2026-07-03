# Bounded Local Pilot Extension

This runner extends the cached-feature head-only smoke inside the local risk budget.

Run only after a green training risk assessment:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\44_bounded_local_pilot_extension.ps1 -PrepareDummyCache
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

It writes ignored runtime reports:

```text
reports\bounded_head_only_extension_report.json
reports\bounded_local_pilot_extension_report.json
reports\bounded_local_pilot_extension_report.md
```

This is still offline proxy/interface evidence only. It does not download assets, use GPU, import heavy VLA models, load models, run model inference, rollout, execute simulators, execute OpenVLA-OFT, use real datasets, or make paper-grade claims.

The broader local policy cap is 300 steps after smaller smoke is stable. This extension runner keeps a stricter 100-step cap by default and uses 64 steps unless overridden.
