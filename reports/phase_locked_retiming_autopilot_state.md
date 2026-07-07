# Phase-Locked Retiming Autopilot State

- branch: `codex/phase-locked-retiming-state0-state1`
- state: `STATE 1 complete`
- evidence level: `bounded_libero_phase_retiming_replay_diagnostic`
- first metric target: exact-init replay/control metric within one local task
- training allowed: `false`
- loss computation allowed: `false`
- downloads allowed: `false`
- GPU jobs allowed: `false`
- OpenVLA-OFT allowed: `false`
- paper-grade claims allowed: `false`

STATE 1 result:
- replay/control metric happened: `true`
- demos/tasks: `1 / 1`
- perturbations tested: `9`
- baselines per perturbation: `9`
- total variants: `82`
- simulator steps: `22248`
- exact-init expert replay reward/success/done: `1.0 / true / 260`
- phase perturbations degraded replay: `9 / 9`
- event-locked retiming recovered over raw: `0 / 9`
- event-locked retiming beat best simple baseline: `0 / 9`
- simple baseline matched or beat event-locked retiming: `3 / 9`
- decision: `kill`
- next state: `archive_or_reframe_phase_locked_retiming`

Key limitation: the selected HDF5 demo exposed EEF positions but no object-position trajectory keys, so object-motion anchors were unavailable from the demo file. The replay still used live simulator object observations for progress metrics and target-object state.
