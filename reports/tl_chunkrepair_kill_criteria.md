# TL-ChunkRepair Kill Criteria

Kill immediately in STATE 1 if:
- no real replay/control metric is produced,
- temporal perturbations do not degrade replay/progress,
- TL-ChunkRepair does not reduce temporal property violations versus no repair,
- TL-ChunkRepair does not beat the best single simple baseline,
- TL-ChunkRepair does not beat the best per-failure-mode simple baseline,
- gripper-only, fixed shift, linear time warp, clipping-only, safety-only, abort-to-stop, repeat-last, or hold matches the method with similar or lower utility cost,
- improvements appear only in symbolic property scores and not in reward, success, done index, safe-success, or trajectory progress,
- the method requires privileged labels, BDDL oracle fields, eval labels, future success labels, or task IDs at inference,
- it only works on one handcrafted perturbation with no nontrivial generalization path.

Continue only if temporal perturbations degrade real replay, TL-ChunkRepair improves replay/control metrics, and both simple-baseline gates are passed with acceptable utility cost.
