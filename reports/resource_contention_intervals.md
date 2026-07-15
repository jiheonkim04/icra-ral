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

