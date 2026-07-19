# A2C2 Panel Host-Telemetry Repair Protocol

Date: 2026-07-19 KST

This protocol freezes the one permitted logging/telemetry repair before the
identical panel rerun. Windows pagefile `CurrentUsage` drift remains a required
diagnostic. Actual write activity is defined by nonzero sampled
`PageWrites/sec` or `PagesOutput/sec`. Host resource failure remains mandatory
for actual write activity, failed post-shutdown memory release, kernel OOM,
host use above 82%, nonzero WSL swap, or model offload.

The rerun starts from zero rows under a new run id and repeats the exact 45
frozen keys. No scientific or action-path field changes.
