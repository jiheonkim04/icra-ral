# TL-ChunkRepair Task Definition

Short name: TL-ChunkRepair.

Long title: Temporal-Logic-Guided Action Chunk Repair for Safe Vision-Language-Action Robot Manipulation.

Problem: given a proposed 7D robot manipulation action chunk, current execution state, observable event predicates, and a temporal manipulation property, detect whether executing the chunk would violate the property and minimally repair the chunk before execution.

Initial properties:
- grasp before lift,
- keep grasp until placement or containment,
- do not release before target region/contact,
- do not move/transport object while gripper is open,
- avoid forbidden contact before safe phase when observable,
- mechanism/action onset order when observable.

Method hypothesis: a finite-state event monitor can locate the causal violation boundary inside the chunk and apply interpretable edits such as hold, delayed release, forced grasp maintenance, unsafe-segment removal, or gripper timing patching while preserving more utility than clipping, hold, abort, fixed shift, or gripper-only fixes.

Evidence boundary: STATE 1 is bounded exact-init replay/control evidence only. It is not benchmark success, paper-grade evidence, or VLA policy competence.
