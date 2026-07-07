# AMP-GD Autopilot State

- branch: `codex/active-micro-probe-state0-state1`
- current stage: `STATE 2`
- last completed stage: `STATE 1`
- current evidence level: `toy_control_rollout_metric`
- rollout/control metric happened: `true`
- simulator used: `toy_2d_point_world_control_diagnostic`
- trials: `60`
- target classes: `dotted`, `striped`
- distractor configurations: `front_back`, `left_right`
- training happened: `false`
- loss computed: `false`
- GPU/download/heavy VLA/OpenVLA-OFT happened: `false`
- continue/kill decision: `continue_to_state2_scale_diagnostic`
- next command: `powershell -ExecutionPolicy Bypass -File scripts\168_amp_gd_minimal_probe_diagnostic.ps1 -Trials 150 -Seeds "11,23,37,53,71"`

State 1 was green because AMP-GD reduced wrong-target rate versus no-probe, random-probe, safety-only/clipping-only, and nearest-target baselines with bounded probe/path cost. State 2 should scale this diagnostic and begin the LIBERO/RoboSuite object-observable port; it must still kill the route if random-probe, safety-only, or nearest-target catches up.
