# A2C2 Setup Preflight Failed Attempt 1

Date: `2026-07-19 KST`

Decision: `A2C2_SETUP_PREFLIGHT_FAILED`

Primary classification: `INFRASTRUCTURE_NULL_DEFECT`

The frozen SmolVLA loaded and produced its action-chunk computation, but the
LeRobot `0.4.4` implementation invokes `vlm_with_expert.forward(...)`
directly. Direct invocation bypasses PyTorch's `Module.__call__` forward-hook
dispatcher, so the read-only compatibility hook did not receive the prefix
tensor and the preflight stopped before A2C2 forward, optimizer, or rollout.

The one permitted narrow repair changes only the observation of the same
bound `forward` return value: the temporary wrapper records the first prefix
token and returns the original output unchanged. It changes no panel,
identity, condition, action chunk, Prior graph, training budget, threshold, or
decision rule. The exact machine-readable failed attempt is preserved in
`reports/a2c2_prior/preflight_failed_attempt_1.json`.
