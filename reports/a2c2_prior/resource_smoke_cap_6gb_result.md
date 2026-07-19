# A2C2 6GB Resource-Only Actual-Path Smoke

Decision: `A2C2_RESOURCE_SMOKE_FAIL_WINDOWS_CEILING`

The verified 6GB WSL cap started from a cleaned Windows baseline of `69.288%`.
During the exact Base load path, Windows RAM reached `83.788%`, above the
frozen `82%` runtime ceiling, so the host monitor stopped the run. Termination
lag produced an after sample of `87.556%` before WSL shutdown completed.

- WSL total: `6,066,260 kB`; swap: `0`; WSLg: disabled.
- Windows pagefile current-usage growth: `0 MiB`; page writes peak: `0/s`.
- Kernel OOM: none observed before the guard.
- Full episode: `false`; pre-episode environment/forward/step counts are unknown
  because the child was terminated before it persisted an internal trace.
- Task success and reward persisted or counted: `false`.
- Scientific interpretation: none. This is a local resource result only.

The next authorized cap is 8GB under the unchanged frozen protocol.
