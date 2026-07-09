# SmolVLA 7D Adapter Replay Bridge Experiment Plan

STATE 1: export or reload the best fixed 7D SmolVLA adapter, verify 7D shape, train-split-only normalization, learned gripper output, unnormalization, and action validity.

STATE 2: compare expert, mean-action, executable ridge, and SmolVLA 7D adapter actions on the first held-out replay demo before any simulator stepping.

STATE 3: if `ALLOW_SMOLVLA_7D_REPLAY_BRIDGE_REPLAY=1` is set and LIBERO/RoboSuite imports, run exact-init replay on one held-out demo with a capped horizon.

Stop immediately if the adapter cannot execute, expert replay is blocked, simple baselines dominate replay/progress, actions are invalid/clipped, or offline L2 fails to transfer to control progress.
