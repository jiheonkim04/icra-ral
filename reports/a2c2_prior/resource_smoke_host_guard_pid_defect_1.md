# A2C2 Resource Smoke Host-Guard PID Defect 1

Decision: `A2C2_RESOURCE_SMOKE_HOST_GUARD_PID_DEFECT`

Stopping only the Windows `wsl.exe` client did not synchronously terminate the
Linux child. The guard fired at `83.788%`, but termination lag allowed an after
sample of `87.556%`. WSL was subsequently shut down and no worker remained.

The one launcher/PID repair terminates the `Ubuntu-22.04` distro immediately,
maps the guard to exit code `82`, and refuses stale internal smoke output.
