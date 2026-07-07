# TL-ChunkRepair Experiment Plan

STATE 1 diagnostic:
- use local LIBERO/RoboSuite exact-init HDF5 replay if the readiness gate is green,
- replay one validated expert chunk as the upper bound,
- create temporal chunk perturbations from the same chunk,
- compare no repair, clipping-only, safety-only one-step filter, gripper-only timing fix, fixed delay/shift, linear time warp, abort-to-stop, repeat-last/hold, and TL-ChunkRepair,
- report temporal property violations, safe-success, reward/success/done/progress, edit distance, intervention rate, utility drop, gripper timing error, and false positive/negative repair indicators where observable.

Perturbations:
- early gripper release,
- delayed gripper close,
- lift before grasp,
- transport with gripper open,
- premature place/release,
- chunk truncation,
- phase skip,
- inserted unsafe contact action.

Execution constraints:
- no OpenVLA-OFT,
- no VLA fine-tuning,
- no GPU,
- no downloads,
- no broad planning-only expansion,
- no paper-grade claim,
- no privileged reward/success/task-label use for repair action selection.
