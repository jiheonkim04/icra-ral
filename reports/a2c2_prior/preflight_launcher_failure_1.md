# A2C2 Preflight Launcher Failure 1

Date: `2026-07-19 KST`

Classification: `EXECUTION_TRANSPORT_FAILURE`

The orchestration call closed its stdout pipe after a five-second timeout.
The still-running child reached the frozen A2C2 prior construction, then
torchvision's ResNet18 download progress writer raised `BrokenPipeError`.
This was not an implementation or scientific-contract failure: there was no
training or rollout, and no code, environment variable, panel, budget, or
decision rule changed. The identical command was rerun with its stdout pipe
retained; its accepted result is `reports/a2c2_prior/preflight_result.json`.
