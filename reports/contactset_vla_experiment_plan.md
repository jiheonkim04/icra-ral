# ContactSet-VLA Experiment Plan

## STATE 1 Diagnostic

Run the smallest executable local diagnostic before any VLA fine-tuning.

Data:

- local LIBERO HDF5 action chunks,
- EEF position/orientation from HDF5 observations,
- object points from HDF5 `*_pos` observations when present,
- object free-joint positions from HDF5 simulator state plus embedded MuJoCo XML when needed,
- destination/support points from instruction-selected observable object traces or static XML bodies/sites,
- natural language instruction only for selecting source/destination names,
- no reward, done, success, eval label, task-id, or filename target-label leakage for geometry features.

Required variants:

1. `no_geometry_injection`
2. `single_3d_point_injection`
3. `source_object_point_only`
4. `destination_placement_point_only`
5. `source_destination_two_point_injection`
6. `full_contact_set_injection`

Contact-set encoding:

- source object point,
- destination/placement point,
- support/contact surface point,
- optional safety/avoidance point,
- optional normal/orientation cue,
- permutation-aware set summary over role-tagged point features relative to the EEF.

Action head:

- tiny CPU NumPy ridge action head,
- base features: time phase, current EEF state, gripper aperture, instruction hash,
- injected features: variant-specific point-set embedding,
- deterministic per-demo time holdout split,
- exploratory only, not confirmatory.

Metrics:

- 7D action L2,
- translation L2,
- rotation L2,
- gripper error,
- target-directed movement proxy,
- source consistency,
- destination consistency,
- contact/placement consistency,
- exact-init replay progress: recorded as not happened in STATE 1 unless a separate risk assessment authorizes it.

Continue only if:

- full contact-set injection beats active single-3D-point injection on held-out action L2,
- source-only, destination-only, and source+destination simple baselines do not match full contact-set injection,
- source, placement/support, and contact-set points are observable without leakage.

Kill if:

- single 3D point injection matches full contact set,
- source object point only matches full contact set,
- destination point only matches full contact set,
- source+destination two-point injection matches full contact set,
- contact set requires oracle/eval target labels,
- no object/placement/contact points are observable,
- no real action loss or meaningful contact metric appears,
- full contact set improves only a proxy that cannot later be connected to replay/progress.

Next state after continue: one separately risk-assessed exact-init replay/progress diagnostic, not full VLA fine-tuning.

