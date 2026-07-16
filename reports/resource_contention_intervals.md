# Resource Contention Intervals

## 2026-07-15 Windows Efficiency Mode Interval

The user reported that Windows Efficiency Mode had temporarily been enabled on
`vmmemWSL` while the Goal UI was paused. Efficiency Mode was disabled before the
continuity audit. The exact start time was not supplied; the interval is
therefore conservatively recorded as start-unknown and ended no later than
`2026-07-15T16:23:29+09:00`.

No Linux research worker was alive at the audit. Campaign state contained no
active PID or running command. The newest durable rollout run was the already
completed EAC Stage B run at `runs/eac_vla_stage_b/20260714T202334Z`.

Audit result:

- wrapper PID `375`: dead;
- child PID `386`: dead;
- heartbeat/status: `completed`;
- exit code: `0`;
- partial JSON: parsed;
- completed/planned episodes: `200 / 200`;
- exceptions, timeouts, infrastructure failures: `0 / 0 / 0`;
- duplicate `(policy, suite, task_id, reset_seed)` keys: `0`;
- missing manifest keys / extra result keys: `0 / 0`;
- action-modified / invalid-action rows: `0 / 0`;
- manifest file SHA256:
  `CD90E53319A10693CFA898E0F8F9157959FE2B922EEACC10242C4678975BE46F`;
- simulator path: one environment with `use_async_envs=False` and synchronous
  `env.step` calls.

Decision: accept the existing EAC closed-loop task-success rows without rerun.
Do not resume or relaunch the completed run.

Evidence quarantine: latency, throughput, wall-clock efficiency, and resource
utilization measurements whose overlap with this start-unknown interval cannot
be excluded are not valid final paper evidence. Closed-loop task-success rows
remain valid because the simulator was synchronous, no timeout or exception
occurred, action semantics and frozen identities were unchanged, and no
duplicate or off-manifest rows were created.

## 2026-07-15 Goal Pause And Gaming Interval 2

The user reported a second start-unknown interval in which Windows gaming and
Efficiency Mode on `vmmemWSL` may have overlapped a detached WSL experiment.
Efficiency Mode was disabled before the continuity audit, which completed at
`2026-07-15T19:24:37+09:00`.

No Linux VLA or simulator worker was alive. The newest experiment was the
already completed FAMR endpoint at `runs/famr_vla/endpoint`:

- PID `387`: dead;
- heartbeat/status: `completed`;
- exit code: `0`;
- partial and final JSON: parsed;
- optimizer steps: `300 / 300`;
- discovery microbatches: `2400 / 2400`;
- exceptions and duplicate schedule keys: `0 / 0`;
- resume or relaunch: not performed.

The FAMR endpoint contains no closed-loop task-success rows, so the simulator,
timeout, reset-identity, and rollout-manifest acceptance conditions are not
applicable to it. Its fixed scientific endpoint adjudication remains unchanged,
but latency, throughput, wall-clock efficiency, and resource utilization are
quarantined because overlap with the start-unknown interval cannot be excluded.
The completed endpoint was accepted without rerun.

## 2026-07-16 Goal Pause Efficiency Mode Interval 4

The user reported another start-unknown interval in which Windows Efficiency
Mode on `vmmemWSL` may have overlapped a detached WSL experiment. Efficiency
Mode was disabled before the continuity audit, which completed at
`2026-07-16T13:55:28+09:00`.

No matching Linux VLA worker was alive. The newest durable experiment artifacts
were the already completed TSC-VLA Stage 0 files in `reports/tsc_vla`:

- PID `415`: dead;
- heartbeat/status: `completed`;
- exit code: `0`;
- partial and final JSON: parsed;
- model rows: `640 / 640`;
- final decision: `TSC_STAGE_0_NO_USABLE_HEADROOM`;
- exceptions: `0`;
- duplicate manifest / partial keys: `0 / 0`;
- missing manifest / extra partial keys: `0 / 0`;
- manifest and partial key sets equal: `true`;
- resume or relaunch: not performed.

TSC Stage 0 contains no closed-loop task-success rows, so simulator task-success
acceptance conditions are not applicable. The completed fixed-protocol result
was accepted without rerun. Latency, throughput, wall-clock efficiency, and
resource-utilization measurements overlapping this start-unknown interval, or
whose overlap cannot be excluded, remain ineligible for final paper evidence.
