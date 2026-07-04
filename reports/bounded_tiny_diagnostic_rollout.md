# Bounded Tiny Diagnostic Rollout

This report defines the execution-only rung after the tiny diagnostic rollout risk planner is green.

Run:

```powershell
$env:ALLOW_TINY_ROLLOUT="1"
powershell -ExecutionPolicy Bypass -File scripts\63_bounded_tiny_diagnostic_rollout.ps1
Remove-Item Env:\ALLOW_TINY_ROLLOUT -ErrorAction SilentlyContinue
```

Scope:

- uses the existing WSL venv at `/home/jiheon/.venvs/tca_map_sim`,
- runs at most 5 toy MuJoCo diagnostic tasks,
- uses one episode per task and at most 5 steps per episode,
- uses no learned policy and no VLA inference,
- creates no LIBERO/RoboSuite benchmark environment,
- performs no training, GPU job, download, heavy VLA import, OpenVLA-OFT execution, token access, multi-seed rollout, benchmark/SOTA claim, or paper-grade claim.

This is simulator plumbing evidence only. It is not LIBERO success, standard success, benchmark evidence, or paper-grade evidence.
