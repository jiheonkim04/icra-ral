# ContactSet-VLA Autopilot State

## Current State

- Branch: `codex/contactset-vla-state1`
- Base: local `main` at `91c8778`
- Milestone: STATE 0 docs plus STATE 1 bounded offline action-head diagnostic
- Heavy training: not allowed
- Full VLA fine-tuning: not allowed
- OpenVLA-OFT: blocked
- GPU jobs: not part of STATE 1
- Simulator rollouts: not part of STATE 1
- Evidence label: exploratory offline action-head proxy

## STATE 1 Result

- Diagnostic report: `reports/contactset_vla_diagnostic_report.md` and `.json`
- Decision: `kill`
- Reason: full contact-set injection did not beat the active single-3D-point injection baseline.
- Training happened: yes, tiny CPU NumPy ridge action-head training only
- Loss computed: yes
- Replay/control metric happened: no
- GPU/download/OpenVLA-OFT happened: no / no / no
- Usable demos: `6`
- Train/eval records: `588 / 252`
- Source/destination/support observable: true / true / true
- Eval-label leakage detected: false
- Variants tested: `no_geometry_injection`, `single_3d_point_injection`, `source_object_point_only`, `destination_placement_point_only`, `source_destination_two_point_injection`, `full_contact_set_injection`
- No-geometry action L2: `0.851451`
- Single-point action L2: `0.930495702`
- Source-only action L2: `1.262017`
- Destination-only action L2: `0.86372`
- Source+destination action L2: `1.360487`
- Full contact-set action L2: `1.105028754`
- Contact-set beats single-point: false
- Simple point baselines matched contact-set: true

Interpretation: ContactSet-VLA should not proceed to full VLA fine-tuning or replay scale-up from this evidence. The contact-set encoder is executable and geometry was observable, but the active single-point and destination/no-geometry baselines were stronger on the first held-out action metric.

## Executable

Safe runner:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\200_contactset_vla_diagnostic.ps1
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

The runner trains tiny CPU NumPy ridge action heads over local HDF5 chunks only. It refuses download, GPU, rollout, simulator, heavy-import, runtime-install, and OpenVLA/OpenVLA-OFT gates.

## Validation Completed

- Targeted tests: `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests\test_contactset_vla_diagnostic.py -q` passed, `3 passed`.
- Diagnostic script: `ALLOW_TINY_TRAINING=1` plus `scripts\200_contactset_vla_diagnostic.ps1` passed and produced a kill decision.
- Safe runner: `scripts\40_cursor_safe_local_check.ps1` passed.
- Full pytest: `498 passed` inside the safe runner.
- Readiness checks: `scripts\11_check_real_assets.ps1`, `scripts\13_check_smolvla_adapter_smoke.ps1`, and `scripts\17_check_smolvla_runtime_deps.ps1` passed.

## Required Final Report Fields

- final main commit,
- training happened yes/no,
- loss computed yes/no,
- replay/control metric happened yes/no,
- GPU/download/OpenVLA-OFT yes/no,
- variants tested,
- single-point metric,
- contact-set metric,
- whether contact-set beats single-point,
- whether simple point baselines matched it,
- continue/kill decision.
