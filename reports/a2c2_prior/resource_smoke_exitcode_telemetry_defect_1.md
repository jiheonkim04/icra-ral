# A2C2 Resource Smoke Exit-Code Telemetry Defect 1

Decision: `A2C2_RESOURCE_SMOKE_EXITCODE_TELEMETRY_DEFECT`

The redirected WSL process produced a null PowerShell `ExitCode`, even though
the internal smoke report correctly recorded failure. The one telemetry-only
repair uses the internal decision as a 0/1 fallback and returns `125` when
neither source exists. This cannot change model, simulator, or task outcomes.
