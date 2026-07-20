# Frozen Persistent-Completion Problem Protocol

This is a ten-task, pre-policy, official-demonstration replay gate. The JSON companion is authoritative.

For each LIBERO-Goal task, test demonstrations in numeric order under exact-init standard replay and select the lowest-index replay that reaches the unchanged official success predicate. Persistence outcomes cannot affect demonstration selection.

Replay the selected demonstration in two primary branches. The immediate branch switches at the first success hit to 30 zero-motion controller actions with the last gripper-command sign latched. The recoverability branch instead completes the unused expert action suffix and then applies the same neutral hold. Persistent success means the official predicate remains true at every hold step. A last-action-repeat branch is secondary only.

The gate requires valid rows for all tasks, native success on at least eight tasks and all four broad mechanisms, at least three immediate-persistence failures across two mechanisms and at least 20% of native successes, and suffix-recoverable persistence on at least two disagreement tasks across two mechanisms. No policy, Ours, training, or confirmatory outcome is permitted.
