# Epoch 6 closed-loop resource blocker

Decision: `INFRASTRUCTURE_OR_RESOURCE_BLOCKED`

Campaign state: `HARD_EXTERNAL_BLOCKER_REQUIRES_USER`

The frozen outcome-suppressed Stage 0 passed and established
`ACTION_LEVEL_SCHEDULE_DEPENDENCE_GO`. The authorized closed-loop resource
smoke then started four simultaneous official LIBERO environment processes,
with one shared X-VLA model scheduled to load only after all four environments
were ready.

The environment-only phase crossed the frozen host-RAM ceiling before model
load: physical use rose from 10,197,872,640 bytes (41.00%) to
21,178,970,112 bytes (85.16%) on the 24,871,014,400-byte host. The monitor
terminated the child at the 82% ceiling. There was no sustained paging, WSL
swap, OOM signature, model forward, simulator action, or outcome read. A
3 MiB pagefile-allocation change was allocation-only under the calibrated
rule. Controlled shutdown restored host use to 39.05%.

This is not a duplicated-model defect: the four workers use `spawn` and are
created before the sole model load. Reducing to one or two live environments,
multiplexing simulator states, or replacing actual four-shard arrival order
would change the preregistered schedule intervention. No scientifically
equivalent narrow repair remains on this host.

For capacity planning, adding the observed four-environment increment
(10,981,097,472 bytes) to the independently measured Stage-0 model increment
(7,485,628,416 bytes) projects 28,664,598,528 bytes in use. Holding that below
82% requires about 34.957 decimal GB total. A 32 GiB host would project to
83.42%, so it cannot be certified under the frozen ceiling; 48 GB is the
smallest standard tier with defensible headroom. This projection is not an
executed full-path measurement.

No closed-loop scientific episode was run, no success value was exposed, and
Ours remains unauthorized. Resume only on a clean 48 GB-or-larger equivalent
host with the same protocol, or under explicit authority for a genuinely new
independent schedule study.
