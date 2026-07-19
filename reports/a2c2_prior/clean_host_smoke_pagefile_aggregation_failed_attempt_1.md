# A2C2 Clean-Host Pagefile Aggregation Failed Attempt 1

Decision: `A2C2_RESOURCE_SMOKE_FAIL_UNRELATED_IMPLEMENTATION`

The 10 GB internal smoke completed, but its host report incorrectly emitted
`A2C2_RESOURCE_SMOKE_PASS`. The persisted raw samples show pagefile current use
at 55 MiB before launch and 59 MiB after WSL shutdown. The monitor had computed
growth before adding that shutdown sample, so its zero-growth field and PASS
are rejected. Recomputing the frozen rule gives
`A2C2_RESOURCE_SMOKE_FAIL_MEMORY_LEAK` with 4 MiB growth.

The root-bounded repair includes after-child and post-shutdown samples in all
peak accumulators before decision mapping and adds a focused regression
assertion. The identical 10 GB stage receives one verification under a fresh
run id. No scientific outcome was exposed or persisted, and no frozen value
changes.
