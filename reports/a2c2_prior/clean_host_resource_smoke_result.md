# A2C2 Clean-Host Resource-Smoke Result

Decision: `A2C2_RESOURCE_SMOKE_PASS`

The clean host began below 40% Windows RAM. Every accepted smoke used one
policy process, one policy instance, one LIBERO environment, batch size one,
no task parallelism, no prefetch, no video, no observation cache, and one full
model residency. No success or reward was persisted or counted.

| Cap | Windows baseline / peak | Pagefile growth | Episode evidence | Decision |
| ---: | ---: | ---: | --- | --- |
| 8 GB | 36.84% / 72.36% | 6 MiB | 76 steps, 8 forwards | `FAIL_MEMORY_LEAK` |
| 10 GB | 36.82% / 72.06% | 5 MiB | 76 steps, 8 forwards | `FAIL_MEMORY_LEAK` |
| 12 GB | 36.72% / 72.54% | 0 MiB | 76 steps, 8 forwards | `PASS` |

All three completed teardown with zero WSL swap, no kernel OOM, no CPU/disk
offload, and at most 932 MiB reserved VRAM. The strict pagefile gate rejected
8 and 10 GB. The corrected 12 GB run had zero pagefile growth/writes and full
post-shutdown memory release, making 12 GB the smallest passing cap. The 14 GB
smoke was not run because escalation after a smaller PASS is prohibited.

Two missing-host-telemetry attempts and one pagefile-aggregation defect are
preserved separately. Only serializer/telemetry evidence changed; no policy,
environment, identity, outcome, or frozen protocol value changed.
