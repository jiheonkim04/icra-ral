# ECHO-VLA Effect Predicate Schema

Date: 2026-07-11 KST

## Explicit Effect Vector

ECHO uses an explicit numeric effect vector, not future RGB frames.

| Component | Symbol | Meaning | Visible/proprioceptive at inference | Privileged training label | Predicted by effect head |
| --- | --- | --- | --- | --- | --- |
| end-effector displacement | `eef_delta_norm` | norm of EEF pose displacement over chunk | yes, from robot state | no | yes |
| target-distance change | `target_distance_delta` | target object distance to goal/support before minus after | no, needs object pose | yes | yes |
| contact transition | `contact_transition` | new contact between gripper and target object | partly visible, not reliable | yes | yes |
| gripper transition | `gripper_transition` | open/close state change and command consistency | yes | no | yes |
| object retention | `object_retained` | target remains controlled after grasp/lift | partly visible | yes | yes |
| object lift | `object_lift_delta` | vertical target-object displacement | partly visible | yes | yes |
| object-target displacement | `object_goal_delta` | object moved closer to final target/support | no | yes | yes |
| placement alignment | `placement_alignment` | object pose aligns with receptacle/placement target | partly visible | yes | yes |
| release stability | `release_stability` | object remains placed after release window | no during candidate selection | yes | yes |

## Phase Set

| Phase | Required effect target |
| --- | --- |
| approach | reduce target distance / produce appropriate EEF displacement without premature contact |
| grasp_contact | create contact, close gripper, begin object retention |
| lift | maintain retention and increase object height/support clearance |
| transport | reduce object-goal distance while maintaining retention |
| placement | improve placement alignment and reduce object-goal distance |
| release | open gripper while preserving release stability |

## Non-Privileged Deployment Rule

At deployment:

- phase is inferred from non-privileged observation, instruction, proprioception, and history;
- candidate effects are predicted before execution;
- realized effects are used only after execution for logging/history, not to choose the already-selected chunk;
- no BDDL predicate or simulator state may be read by the selector.

## Lightweight Head Targets

The first prototype trains:

- phase head: multiclass phase target;
- effect head: mixed regression/classification over the effect vector;
- compatibility head: scalar phase-effect compatibility;
- ranking objective: pairwise preference within same-state intervention groups.

## Schema Version

`echo_vla_effect_schema_v1`
