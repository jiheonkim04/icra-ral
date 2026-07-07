# ResetSpec-Retarget Autopilot State

- branch: `codex/resetspec-retarget-state0-state1`
- current stage: `KILL_OR_REFRAME`
- last completed stage: `STATE 1`
- evidence level: `bounded_libero_replay_retarget_diagnostic`
- replay/control metric happened: `true`
- training happened: `false`
- LoRA training happened: `false`
- loss computed: `false`
- GPU/download/heavy VLA/OpenVLA-OFT happened: `false`
- final decision: `kill`

Key result:
- exact-init expert replay: reward/success `1.0 / true`, first done `260`.
- default-reset raw replay: reward/success `0.0 / false`.
- diagonal-affine and clipping baselines matched raw and failed.
- fixed global-scale replay from default reset succeeded: reward/success `1.0 / true`, first done `257`.
- object-relative translation retargeting improved progress and trajectory drift but did not reach reward/success.
- object-relative translation plus gripper-phase retargeting also failed reward/success.

Conclusion: ResetSpec-Retarget has a real reset-mismatch gap and useful object-relative diagnostic plumbing, but it fails the simple-baseline gate because fixed global scaling beats it on the tested task.
