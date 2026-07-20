# Epoch 9B v2 Task-Preservation Protocol

Frozen: 2026-07-21T00:56:58+09:00

Status: `FROZEN_BEFORE_ANY_EPOCH9B_CONTROLLER_OUTCOME`

This rule was frozen before any Epoch 9B controller outcome. The old 3 cm v1 gate remains unchanged for every historical v1 result and is retained as a reported reference only.

## Geometric basis

The bowl's collision geometry has a maximum planar radius of `0.053949 m`. Across untouched first-probe states from development demos 30..39, front reset XY spans are `[0.0408095767794618, 0.03521896494823876]` and back spans are `[0.039823674462334535, 0.04501543365031199]`.

The v2 absolute displacement limit is `0.050 m`, below one collision radius. It is coupled to lane membership and task completion and was not selected from the old 4.19 cm result.

| slot | center-x lane (m) | center-y lane (m) |
|---|---:|---:|
| front | [0.065, 0.18] | [0.115, 0.175] |
| back | [-0.185, -0.07] | [0.015, 0.09] |

## Validity rule

- intended candidate physically contacted or excited (simulator contact is evaluation-only)
- all raw 7-D actions finite and within [-1,1]
- candidate center remains inside its frozen lane at every sampled step
- candidate displacement <= 0.05 m and is reported continuously
- candidate remains upright with center z in the reachable envelope
- no candidate-candidate or candidate-distractor collision, identity swap, fall, or workspace exit
- audited RGB tracker remains on the intended instance with template quality >= 0.50
- end effector returns within 0.05 m of episode-neutral pose and final z >= 1.10 m

At scene level, both instances must remain trackable and reachable and the pose-adaptive scripted oracle must complete the property-conditioned task under the official success predicate. The retained canonical X-VLA path is run separately from the identical post-probe state as a practical-headroom endpoint.

## Frozen 24-scene panel

The manifest contains 24 fresh procedural development scenes and 48 probes. Heavy slot, probe order, and heaviest/lightest instruction are a fully crossed 2x2x2 design with three scenes per cell. Both candidate placements are independently permuted inside their original BDDL regions. The task generator uses development demo 30 only as a robot/fixture base state, replaces both candidate free-joint XY coordinates, and never reads identities 40..49.

The complete manifest, exact gates, continuous metrics, and information boundary are in the companion JSON.
