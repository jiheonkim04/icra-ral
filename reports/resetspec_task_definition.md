# ResetSpec-Retarget Task Definition

Long title: Object-Relative Executable Action Retargeting for Robust VLA Replay under Initial-State and Object-Pose Mismatch.

Hypothesis: object-relative and EEF-relative replay retargeting should recover under reset/object-pose mismatch better than raw replay and simple action-only baselines.

Scope:
- one local LIBERO/RoboSuite task first,
- HDF5 expert actions and observations,
- exact-init expert replay as the bridge sanity upper bound,
- default-reset raw replay as the mismatch probe,
- object/EEF state from simulator observations when available,
- no eval success labels, BDDL target metadata, dataset target labels, task IDs, filenames, or manifest target fields as inference-time target proxies.

Evidence level: bounded replay diagnostic only, not a policy rollout claim and not paper-grade evidence.
