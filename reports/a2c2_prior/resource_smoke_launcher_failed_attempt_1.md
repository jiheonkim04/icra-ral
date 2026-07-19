# A2C2 Resource Smoke Launcher Failed Attempt 1

Decision: `A2C2_RESOURCE_SMOKE_LAUNCHER_FAILED_BEFORE_EXECUTION`

Classification: `INFRASTRUCTURE_NULL_DEFECT`

The default Windows PowerShell execution policy rejected the checked-in host
monitor before WSL, the model, or the simulator launched. No task outcome was
observed, persisted, or counted. The scientific protocol and 6GB cap are
unchanged.

The single repair for this launcher root is to invoke the same monitor through
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File` and rerun the same
6GB resource-only stage.
