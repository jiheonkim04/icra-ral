# Epoch 6 Terminal Handoff

Terminal state: `HARD_EXTERNAL_BLOCKER_REQUIRES_USER`
Campaign-state hash: `49FD9623C65A42CD28342545D6B0854E41925CA322EF6CC94080F040672F996D`
Scientific outcome rows exposed: `0`
Paper candidate: not reached
Paper generation: unauthorized

## Exact blocker

Every surviving ranked route needs the local LIBERO/WSL execution path. The
schedule-invariance route now fits in physical memory: complete v6/v7 model
forwards peaked at 73.8% and 73.6%, below the frozen 82% ceiling, with one
finite 30x20 action chunk, zero WSL swap, zero simulator actions, and no
reward/success/done read. It nevertheless failed host qualification because
Windows pagefile allocation grew by 8 MiB in v7 and WSL did not return freed
anonymous memory within the bounded teardown window.

The first backup independently reached the same external constraint before
scientific contact-label extraction. Four one-state smokes had exact state
round-trip error 0, resolved all 7 robot contact geoms, retained zero robot
edges, and used zero swap/actions/outcomes, but failed pagefile or teardown
qualification. The persistent-success backup requires the same simulator path,
so it cannot safely begin outcome-bearing replay.

This is an operational/resource conclusion only. Schedule dependence, contact
topology prevalence/headroom, persistent success, and every proposed method
remain scientifically unadjudicated.

## Authoritative evidence

- `reports/epoch6_campaign_state.json`
- `reports/epoch6_schedule_invariant_evaluation/operational_blocker.json`
- `reports/epoch6_contact_transition_topology/operational_blocker.json`
- `reports/epoch6_evidence_index.json`
- `reports/epoch6_closure_registry.json`
- `reports/epoch6_resource_inventory.json`

All immutable run directories referenced by those files remain under `runs/`.
The pre-existing untracked `rollouts/2026_07_17/` and
`rollouts/2026_07_18/` directories were preserved and excluded.

## Smallest condition that reopens progress

The human should save and close user-owned WSL shells and unrelated
applications, then reboot into a clean host session. The alternative is a
scientifically equivalent host with at least 32 GB RAM and pagefile-disabled
or demonstrably zero-pagefile execution. Codex did not close applications,
terminate the WSL shell, change Windows pagefile settings, reboot, purchase
compute, use a physical robot, or expose confirmatory identities.

After the condition changes, do not reuse a failed run directory. Create a new
immutable schedule Stage-0 run, repeat static preflight and outcome-suppressed
fixture capture, and execute the host smoke with the frozen protocol, model,
seed, thresholds, and cache-release monitor unchanged. Only
`EPOCH6_STAGE0_RESOURCE_SMOKE_PASS` authorizes the four action sequences.

## Resume command

```powershell
git switch codex/epoch6-ral-submission-convergence-v2
Get-Content reports/epoch6_campaign_state.json -Raw | ConvertFrom-Json | Out-Null
Get-FileHash -Algorithm SHA256 `
  reports/epoch6_schedule_invariant_evaluation/operational_blocker.json,`
  reports/epoch6_contact_transition_topology/operational_blocker.json,`
  reports/epoch6_evidence_index.json,`
  reports/epoch6_closure_registry.json,`
  reports/epoch6_resource_inventory.json
# After preparing a fresh immutable run with preflight and fixture:
powershell -ExecutionPolicy Bypass `
  -File scripts/monitor_epoch6_schedule_stage0_smoke.ps1 `
  -RunId <fresh_run_id> `
  -AllowWslCacheDropAfterChild
```
