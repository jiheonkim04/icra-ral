# A2C2 Resource-Feasibility Continuation Result

Decision: `A2C2_RESOURCE_FUNDAMENTALLY_BLOCKED_ON_CURRENT_24GB_HOST`

This is a local resource decision, not an A2C2 scientific result. The
historical `NO_DEFENSIBLE_LOCAL_RESEARCH_PATH_FOUND` and
`PRIOR_INFRASTRUCTURE_BLOCKED` records remain intact. The implementation is
still `MECHANISM_FAITHFUL_A2C2_LOCAL_PORT`, and the frozen A2C2 hypothesis is
still scientifically unadjudicated.

The initial Windows load was 85.10%. Safely closing unrelated applications
and trimming reversible working sets produced permitted launch baselines
between 66.741% and 69.288%. These stayed below the hard 70% no-launch line,
but not the preferred 55% or acceptable 65% targets because essential
Cursor/Codex processes were retained. No game or research worker was active
at launch. The exact resource-only Base path then returned:

| WSL cap | Effective RAM | Baseline | Guard peak | Release-lag sample | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 6 GB | 6,066,260 KiB | 69.288% | 83.788% | 87.556% | Windows ceiling |
| 8 GB | 8,130,636 KiB | 68.144% | 85.480% | 89.681% | Windows ceiling |
| 10 GB | 10,182,740 KiB | 66.741% | 86.514% | 88.875% | Windows ceiling |
| 12 GB | 12,247,124 KiB | 68.256% | 86.758% | 89.635% | Windows ceiling |

Every run used zero WSL swap and no WSLg process. Pagefile current-use growth
and peak page writes were zero. No kernel OOM was observed before the monitor
guard, though kernel logs were unavailable after forced distro termination.
The exact model path began loading, but every run crossed the frozen 82% host
ceiling before a child trace or complete episode could be persisted. Thus
environment construction and model-forward/simulator-step counts remain
unknown, no success or reward was saved or counted, and the full frozen
Base/Prior panel was not authorized.

The optional 14 GB smoke was not run. Its required `<=40%` cleaned baseline
was not met, and the observed root was the Windows ceiling rather than an
isolated low WSL cap. No 16–20 GB cap was permitted.

## RAM adjudication

The largest physical-use sample was 22.305 decimal GB during termination lag.
Keeping that same load at or below 82% requires at least 27.201 decimal GB
before ordinary headroom. Therefore:

- 32 GB installed RAM is the practical minimum and is likely sufficient for
  this one sequential frozen evaluation with similar background control, but
  it is not a guarantee;
- 48 GB is preferred because it leaves useful Windows, Codex, driver, and
  filesystem-cache headroom without aggressive application closure; and
- 64 GB adds little to this single frozen evaluation, though it may help with
  simultaneous heavy applications, larger models, or parallel workloads.

No hardware purchase is automatic. The next permitted research action would
require a manual RAM upgrade and fresh explicit authority to rerun the same
unchanged resource smoke. Ours and Pivot Epoch 3 remain unauthorized.
