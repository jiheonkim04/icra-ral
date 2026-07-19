# A2C2 10GB Resource-Only Actual-Path Smoke

Decision: `A2C2_RESOURCE_SMOKE_FAIL_WINDOWS_CEILING`

The verified 10 GB cap and zero swap started from a `66.741%` Windows
baseline. The exact Base load path reached `86.514%`, so the frozen 82% guard
directly terminated the distro. The `88.875%` after-sample is
termination-release lag. Pagefile growth and page writes were zero, and no
kernel OOM was observed before the guard. The child produced no internal
episode trace; consequently environment construction and forward/step counts
are unknown. No success or reward was persisted or counted.
