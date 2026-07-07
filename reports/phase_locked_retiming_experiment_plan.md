# Phase-Locked Retiming Experiment Plan

STATE 1 runs one exact-init LIBERO HDF5 demonstration replay and injects temporal phase mismatches:
- gripper close delayed,
- gripper close early,
- lift phase delayed,
- lift phase early,
- chunk shifted forward,
- chunk shifted backward,
- time stretch,
- time compression,
- chunk boundary offset.

For each perturbation, compare:
- raw perturbed replay,
- fixed time shift,
- repeat-last/hold,
- gripper-only timing correction,
- global scale,
- diagonal affine,
- linear time warp,
- nearest-progress demo if object/EEF progress is observable,
- event-locked retiming.

Metrics:
- reward, success, done index,
- replay recovery rate,
- event timing error,
- gripper timing error,
- trajectory drift,
- EEF-object distance change,
- object movement,
- controller-valid action rate,
- clip rate,
- improvement over raw perturbed replay,
- improvement over best simple baseline.

The diagnostic must write `reports\phase_locked_retiming_state1_result.json` and `reports\phase_locked_retiming_state1_result.md`.
