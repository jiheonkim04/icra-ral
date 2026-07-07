# Phase-Locked Retiming Reusable Artifacts

Keep these pieces as diagnostic infrastructure:
- bounded WSL replay script: `scripts\180_phase_locked_retiming_diagnostic.ps1`,
- diagnostic module: `tca_map.phase_locked.retiming`,
- event-anchor extraction from actions, EEF trajectories, gripper transitions, and object observations when available,
- perturbation generator for gripper delay/advance, lift delay/advance, chunk shift, time stretch/compression, and boundary offset,
- exact-init replay/control runner,
- per-failure-mode baseline table,
- event timing, gripper timing, trajectory drift, EEF-object distance, object movement, controller-valid action, and clip-rate metrics,
- concise result report: `reports\phase_locked_retiming_state1_result.md`.

Do not reuse Phase-Locked Retiming as a main method claim unless a future route beats both the best single simple baseline and the best per-failure-mode simple baseline under predeclared criteria.
