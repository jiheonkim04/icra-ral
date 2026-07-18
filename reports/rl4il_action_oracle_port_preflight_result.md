# RL4IL Action-Oracle Prior Port Preflight Result

- Decision: `RL4IL_ACTION_ORACLE_PORT_PREFLIGHT_PASS`
- Dataset root: `/mnt/c/assets/data/libero`
- Identity mapping: `initial_state_index = reset_identity - 20260711`
- Action-oracle config: resample `64` steps; length-penalty weight `0.01`
- Training, optimizer, checkpoint, simulator rollout, and Ours work: none.

## Why this preflight exists

The official RL4IL release assigns a constant scalar label to every demonstration, but the paper-level IL mechanism requires the recorded action sequence to serve as the action signal. This preflight validates the replacement action-sequence oracle before any prior training or rollout is armed.

## Results

| suite/task | demos | action lengths | unique nearest action-oracles | min off-diagonal distance | mean off-diagonal distance |
|---|---:|---|---:|---:|---:|
| `libero_goal/task0` | 50 | 116..196 | 27 | 0.004562416728585959 | 0.037908262177564774 |
| `libero_object/task0` | 50 | 136..196 | 30 | 0.006711575442912595 | 0.07503705708699998 |
| `libero_spatial/task5` | 50 | 86..154 | 30 | 0.015192028361396128 | 0.12074494501250232 |

All checked demos have 7D actions, nonzero off-diagonal action distances, and nondegenerate nearest action-oracle structure.

## Reset-state finding

The HDF5 demo with the same numeric index as an X-VLA reset identity does not match the official LIBERO reset state:

- Goal task0 same-index L2 to official init: `0.321216349955591..0.7745020951020343`
- Object task0 same-index L2 to official init: `1.0786177508593875..1.1397043511822882`
- Spatial task5 same-index L2 to official init: `2.98560735958861..2.992202742791687`

Therefore the prior runner must not pretend HDF5 `demo_i` is the paired query for reset identity `20260711 + i`. It must query live initial observations from the official reset states, then retrieve/replay training-demo actions.

## Next action

Implement or launch the bounded RL4IL action-sequence-oracle prior runner with live initial-observation queries.
