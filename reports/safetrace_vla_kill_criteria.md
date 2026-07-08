# SafeTrace-VLA Kill Criteria

Kill or source-block if no official or local source can produce temporal safety metrics, no observable temporal property exists, only toy/synthetic symbolic metrics are available, no real data/replay/rollout-backed metric appears, no nontrivial preference pairs exist, safe actions collapse to no-op/stop, or utility cannot be preserved.

Kill if safety-only, stop-on-risk, clipping-only, reward-penalty, generic DPO/preference, task-success-only BC, or any trivial monitor/filter baseline matches the intended SafeTrace effect.

Source-block if official safety benchmark assets require unresolved login, token, payment, license click-through, large unapproved download, unsupported simulator setup, OpenVLA-OFT, full VLA training, or a human-only blocker.

STATE 2 is forbidden unless STATE 1 is green.

