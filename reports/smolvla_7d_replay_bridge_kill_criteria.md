# SmolVLA 7D Adapter Replay Bridge Kill Criteria

Final decision must be one of:
- `READY_FOR_METHOD_AFTER_REPLAY_BRIDGE`
- `NEEDS_EXECUTABLE_ADAPTER_FIX`
- `OFFLINE_TO_CONTROL_GAP`
- `MEAN_OR_MLP_REPLAY_DOMINATED`
- `EXPERT_REPLAY_BLOCKED`
- `TOO_HEAVY_LOCAL`

Stop if the learned 7D adapter cannot be executed, the env action interface mismatch returns, expert replay fails, mean-action or ridge/MLP matches or beats learned replay/progress, adapter actions are mostly clipped/invalid, offline L2 improves without replay/progress transfer, or the runner becomes unstable or unbounded.
