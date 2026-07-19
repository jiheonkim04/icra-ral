# Frozen Problem/Supervision Protocol: Contact-Transition Topology

This is a discovery-only, pre-method protocol. It freezes all task identities,
demo IDs, graph semantics, controls, thresholds, and decisions before any
Epoch 6 contact label is extracted. The machine-readable JSON companion is
authoritative.

## Scope and legal data

Use demo IDs 0--5 from the ten frozen LIBERO tasks. Five development-train
tasks, one development-tune task, and four untouched validation tasks are named
in the JSON. Stage 0A may read only flattened simulator states and actions.
Stage 0B may additionally read the listed RGB and robot-observation arrays.
`rewards` and `dones` are forbidden, as are simulator success checks and policy
rollouts. The official LIBERO repository states that the demonstration datasets
are CC BY 4.0.

## Stage 0A: label feasibility

Restore each recorded state directly, call only `sim.forward`, and read active
MuJoCo contacts. Resolve every collision geom published by the robot model and
discard any edge touching those IDs. Collapse remaining geoms to bodies, type
bodies as free, articulated, or static from their joint ancestry, retain edges
with at least one non-static endpoint, and emit typed birth/death labels after a
two-state persistence debounce.

The gate requires exact replay, complete robot-geom resolution, zero retained
robot edges, noncollapsed validation transitions, multiple supported typed
birth/death bins, a material fraction of transitions outside a +/-2-frame
gripper-change window, deterministic cold-repeat hashes, zero forbidden data
access, zero actions, zero outcomes, zero swap, and bounded host resources. A
failure is a data/supervision or implementation/resource failure, not a method
result.

## Stage 0B: predictability and action headroom

Only after Stage 0A GO, train the frozen small visual probe on development
tasks and select its epoch on the single tune task. It must predict transitions
on held-out tasks beyond prevalence and nonvisual proprio/action/time controls.
Separately, a privileged oracle ridge diagnostic asks whether typed topology
contains arm-action information beyond base history, gripper/stage, and binary
contact-transition controls. A within-demo shuffled-topology control must erase
most of the gain.

Stage 0B GO is still pre-method evidence. It authorizes a new contract, not VLA
training, closed-loop access, an “Ours” label, or a paper.

## Failure and rotation

Stage 0A NO-GO archives this exact contact-topology formulation and activates
the persistent-success backup. Stage 0B NO-GO does the same after recording the
narrow headroom or equivalence conclusion. Thresholds and identities cannot be
changed after labels are exposed.
