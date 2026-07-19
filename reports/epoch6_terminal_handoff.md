# Epoch 6 terminal handoff

Terminal state: `HARD_EXTERNAL_BLOCKER_REQUIRES_USER`

Campaign-state hash: `2D68810CCF14174E1075DC47D12A60BE8F06E6DEB1B716C4B9BD48E04160F4D8`

Stage-0 decision: `ACTION_LEVEL_SCHEDULE_DEPENDENCE_GO`

Closed-loop decision: `INFRASTRUCTURE_OR_RESOURCE_BLOCKED`

Closed-loop scientific episodes exposed: `0 / 40`

Ours: unauthorized

Paper candidate: not reached

## Scientific status

The human clean-host condition successfully reopened Stage 0. The calibrated
resource smoke passed, then A and its cold restart matched on all 20 logical
keys, reversed-order B changed all 20, and the median normalized order effect
was 1.1129 times the independent-root reference. This establishes action-level
schedule dependence on the frozen outcome-suppressed panel. It does not yet
establish that scheduling changes official closed-loop task success.

The subsequent 40-episode protocol was frozen before any success outcome. It
uses 20 matched official reset identities under one canonical serial schedule
and one actual-arrival four-shard schedule, one shared X-VLA model, official
LIBERO horizons and success semantics, atomic query transactions, and
missing-suffix-only resume.

## Exact blocker

The required outcome-free resource smoke started four simultaneous official
LIBERO environment processes. Host use rose from 41.00% to 85.16% before the
single model could load, so the monitor stopped the run at the unchanged 82%
ceiling. There was no sustained Windows paging, WSL swap, OOM signature,
model forward, simulator action, or reward/success/done read. Controlled WSL
shutdown restored host use to 39.05%.

This was not duplicate model residency: the environments were spawned before
the sole model load. Reducing the number of live environments, replacing real
four-shard arrivals with state multiplexing, or changing the schedule would
alter the preregistered intervention rather than repair an implementation
defect.

A final outcome-free process-start audit measured 619,638,784 bytes of
LIBERO/Torch import residency per fresh process. Even granting perfect sharing
of all four duplicate copies leaves only 1,693,816,832 bytes below the ceiling,
versus a measured 6,923,243,520-byte model-active increment. Forking after
CUDA/EGL creation is unsafe, while pre-forking a live environment would no
longer be four independently constructed official environments. The candidate
repair was therefore rejected without changing code or running outcomes.

The conservative additive capacity projection is 28,664,598,528 bytes in use.
Keeping that below 82% needs about 34.957 decimal GB total. A 32 GiB host would
project to 83.42%; 48 GB is the smallest standard tier with defensible
headroom. The projection is for planning and is not an executed full-path
measurement.

## Authoritative evidence

- `reports/epoch6_campaign_state.json`
- `reports/epoch6_schedule_invariant_evaluation/stage0_result.json`
- `reports/epoch6_schedule_invariant_evaluation/closed_loop_execution_manifest.json`
- `reports/epoch6_schedule_invariant_evaluation/closed_loop_resource_blocker.json`
- `reports/epoch6_schedule_invariant_evaluation/closed_loop_process_start_repair_audit.json`
- `reports/epoch6_evidence_index.json`
- `reports/epoch6_closure_registry.json`
- `reports/epoch6_resource_inventory.json`

Raw immutable run artifacts remain under `runs/` and are hash-bound by the
tracked reports. The protected untracked `rollouts/2026_07_17/` and
`rollouts/2026_07_18/` directories remain untouched.

## Smallest condition that reopens progress

Provide or authorize a scientifically equivalent clean host with at least
48 GB physical RAM. Keep zero WSL swap, no CPU/disk model offload, the 82%
host ceiling, model, tasks, identities, schedules, seeds, action semantics,
horizons, and decision gates unchanged.

Alternatively, the human may explicitly authorize a genuinely new independent
schedule study with a lower-concurrency intervention. That study cannot be
reported as an unchanged continuation of the current four-shard protocol.

## Resume command

```powershell
git switch codex/epoch6-ral-submission-convergence-v2
git pull --ff-only
Get-Content reports/epoch6_campaign_state.json -Raw | ConvertFrom-Json | Out-Null
Get-FileHash -Algorithm SHA256 `
  reports/epoch6_schedule_invariant_evaluation/closed_loop_resource_blocker.json,`
  reports/epoch6_evidence_index.json,`
  reports/epoch6_closure_registry.json,`
  reports/epoch6_resource_inventory.json
# On a clean 48 GB-or-larger host, create a fresh immutable run, then:
powershell -ExecutionPolicy Bypass `
  -File scripts/monitor_epoch6_schedule_closed_loop_smoke.ps1 `
  -RunId <fresh_run_id>
```
