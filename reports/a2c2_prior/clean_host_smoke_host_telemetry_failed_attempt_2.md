# A2C2 Clean-Host Smoke Host-Telemetry Failed Attempt 2

Decision: `A2C2_RESOURCE_SMOKE_FAIL_UNRELATED_IMPLEMENTATION`

The identical 8 GB smoke with a 180-second outer timeout reproduced the
post-shutdown failure and revealed the exact root: seven host-payload fields
used bare `false` tokens rather than PowerShell `$false`. PowerShell therefore
raised `CommandNotFoundException` before writing the host JSON.

The internal smoke again completed 76 simulator steps and eight Base forwards
with finite actions, zero swap, no kernel OOM or offload, and successful
environment/model teardown. Its success/reward was neither persisted nor
counted, and its internal PASS is rejected as resource evidence because the
Windows peak record is absent.

The root-bounded `1 / 1` repair changes only those seven serializer literals
and adds a regression assertion that forbids bare boolean assignments. The
same frozen 8 GB stage is rerun under a fresh run id; no policy, environment,
identity, checkpoint, or scientific value changes.
