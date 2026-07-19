# A2C2 official action-semantics audit

Decision: `OFFICIAL_ACTION_SEMANTICS_VERIFIED`

The released evaluator sends each Base or corrected-Prior output directly to
`OffScreenRenderEnv.step`; it adds no clip. Both checkpoints use identical
seven-dimensional action mean/std statistics and return unnormalized
environment actions before that step.

LIBERO delegates unchanged to robosuite. The nominal environment action spec
is `[-1,1]^7`. `SingleArm` routes dimensions 0..5 to the OSC controller and
dimension 6 to the Panda gripper:

- OSC natively clips its six inputs to `[-1,1]` and scales translations to
  `[-0.05,0.05]` and rotations to `[-0.5,0.5]`.
- Panda gripper processing uses the sign of the seventh action, increments its
  two-finger native state by `0.01`, saturates that state to `[-1,1]`, and
  maps it to actuator control ranges.
- Generated joint torques are natively clipped to robot torque limits.

The same post-policy path applies to Base and Prior. Native arm, gripper,
actuator, torque, and simulator-state values can be observed passively by
calling the original native methods exactly once and returning their identical
outputs. No external `clip` is added.

The checkpoint-compatible prior is still a declared local port: its historical
author graph predicts a normalized final refined action, then unnormalizes it.
That deviation and the historical strict-raw-bound invalidity remain preserved;
neither prevents verifying the post-policy official action path.
