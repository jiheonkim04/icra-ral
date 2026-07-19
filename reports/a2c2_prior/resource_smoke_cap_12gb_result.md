# A2C2 12GB Resource-Only Actual-Path Smoke

Decision: `A2C2_RESOURCE_SMOKE_FAIL_WINDOWS_CEILING`

The verified 12 GB cap and zero swap started from a `68.256%` Windows
baseline. The exact Base load path reached `86.758%`, so the frozen 82% guard
directly terminated the distro. The `89.635%` after-sample is
termination-release lag. Pagefile growth and page writes were zero, and no
kernel OOM was observed before the guard. The child produced no internal
episode trace; consequently environment construction and forward/step counts
are unknown. No success or reward was persisted or counted.

The optional 14 GB smoke is not authorized: its required cleaned baseline was
at most 40%, while the accepted cap baselines were `66.741–69.288%`, and the
failures were the Windows ceiling rather than evidence that the WSL cap itself
was too low.
