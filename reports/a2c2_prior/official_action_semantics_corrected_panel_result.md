# A2C2 Official Action-Semantics Corrected Panel Result

Date: 2026-07-19 KST

Exact decision: `CORRECTED_A2C2_NO_REPEATABLE_DELAY_GAP`.

The accepted rerun completed all 45 frozen rows with matched identities,
strict checkpoint loads, 323 Base forwards, 2,148 live Prior forwards, nonzero
corrections, finite 7-D actions, valid official controller-native handling,
zero controller rejection, zero episode exception, zero swap/offload/OOM, and
host decision `A2C2_CORRECTED_HOST_PANEL_PASS`. It remains an
`A2C2_FIDELITY_CORRECTED_LOCAL_PORT`, not an official reproduction.

Results were:

- Base standard: `11/15` (`4/5`, `3/5`, `4/5` by task 0/4/8)
- Base delayed: `9/15` (`3/5`, `3/5`, `3/5`)
- Prior delayed: `9/15` (`3/5`, `2/5`, `4/5`)

Base competence passed. There were four matched clean-to-delayed failures over
two tasks, but the aggregate success difference was only `2/15`; the frozen
repeatable-gap rule required at least `3/15`. The repeated-problem gate is
therefore false before Prior improvement is used to choose a research route.
This result does not establish corrected A2C2 improvement, saturation, or
no-improvement. It establishes that this local panel lacks a sufficiently
repeatable asynchronous-delay degradation for claim-specific method
development.

Raw nominal-bound exceedance remained diagnostic rather than an automatic
invalidity. All 2,725 exceedance events are persisted with task/reset/step/
chunk coordinates. Maximum raw magnitude was `1.077136397` for delayed Base
and `1.029590964` for Prior. Prior maximum exceedance was lower than matched
delayed Base on all 15 identities; its exceedance fraction was higher on nine
and lower on six. Native arm clipping occurred on four Base steps and zero
Prior steps. All native arm/gripper effective values, actuator commands, and
torques remained in bounds, and every simulator state was finite. No external
clip was added.

The first full attempt is preserved but quarantined because the inherited host
monitor mistook a 6 MiB system-wide pagefile reservation-counter drift for
paging despite zero sampled writes/output. One preregistered telemetry repair
was applied. The accepted full rerun returned pagefile growth/write/output all
zero and reproduced every compared scientific/action field exactly (zero
mismatches across 45 rows).

Peak allocated VRAM was `1532.542 MiB`, peak process RSS was `4406.098 MiB`,
peak WSL use was `4.011 GiB`, and host physical use peaked at `66.04%`. WSL
swap, pagefile writes/output, model offload, OOM, and host-ceiling termination
were all zero/false. Memory release passed.

Frozen route: close the local claim-specific asynchronous-delay thesis without
an additional Prior, Ours, Stage 0/A/B, or paper package. The historical raw-
bound invalidity remains preserved separately as
`HISTORICAL_LOCAL_STRICT_RAW_BOUND_GATE_RESULT`.
