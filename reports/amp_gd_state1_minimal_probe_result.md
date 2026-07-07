# AMP-GD State 1 Minimal Probe Result

Status: completed as a bounded toy rollout/control diagnostic.

Execution boundary:
- rollout/control metric happened: yes, in a local toy 2D point-world diagnostic.
- LIBERO/RoboSuite rollout happened: no.
- training happened: no.
- LoRA training happened: no.
- loss was computed: no.
- GPU, downloads, heavy VLA imports, OpenVLA-OFT, benchmark rollouts, and paper-grade claims: no.

Diagnostic scope: `60` trials, seeds `[11, 23, 37]`, target classes `['dotted', 'striped']`, distractor configurations `['front_back', 'left_right']`.

- `no_probe_greedy`: target acc `0.5`, wrong-target `0.5`, success `0.066666667`, unsafe `0.816666667`, probe cost `0.0`.
- `random_probe`: target acc `0.533333333`, wrong-target `0.466666667`, success `0.533333333`, unsafe `0.0`, probe cost `0.12`.
- `safety_only_clipping`: target acc `0.5`, wrong-target `0.5`, success `0.5`, unsafe `0.0`, probe cost `0.0`.
- `nearest_target`: target acc `0.516666667`, wrong-target `0.483333333`, success `0.066666667`, unsafe `0.833333333`, probe cost `0.0`.
- `amp_gd_micro_probe`: target acc `1.0`, wrong-target `0.0`, success `1.0`, unsafe `0.0`, probe cost `0.12`.

Key comparison:
- AMP-GD wrong-target reduction vs no-probe: `0.5`.
- AMP-GD wrong-target reduction vs random-probe: `0.466666667`.
- AMP-GD wrong-target reduction vs safety-only: `0.5`.
- AMP-GD extra path length vs no-probe: `0.318929988`.
- AMP-GD utility drop vs no-probe: `-1.718107002`.

Decision: `continue_to_state2_scale_diagnostic`.

Reason: AMP-GD reduced wrong-target decisions against no-probe, random-probe, and safety-only with bounded probe/path cost.

Limitation: this is toy control evidence. Continue only by scaling the diagnostic and moving the same predeclared active-probe/baseline structure toward LIBERO/RoboSuite object-observable exact-init scenes.
