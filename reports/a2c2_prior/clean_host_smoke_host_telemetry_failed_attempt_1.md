# A2C2 Clean-Host Smoke Host-Telemetry Failed Attempt 1

Decision: `A2C2_RESOURCE_SMOKE_FAIL_UNRELATED_IMPLEMENTATION`

The first clean-host 8 GB invocation gave the orchestration shell only ten
seconds for a complete-episode monitor. The shell timed out while the monitor
and WSL child continued, and the detached monitor was reaped after child
completion before it wrote its Windows host-telemetry JSON.

The internal non-scientific smoke did complete: 76 simulator steps, eight Base
forwards, 4,175.73 MiB peak RSS, 932 MiB peak reserved VRAM, 48.8% peak WSL
use, zero swap, no kernel OOM or offload, and successful teardown. Its internal
PASS is not accepted because the required Windows peak record is missing. No
success/reward or scientific row was persisted or counted.

This first diagnosis was superseded when the identical 180-second invocation
reproduced the failure and printed the exact PowerShell serializer exception.
No code repair was made for the initial diagnosis. The authoritative root and
repair are recorded in `clean_host_smoke_host_telemetry_failed_attempt_2`.
