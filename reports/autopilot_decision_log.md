# Autopilot Decision Log

## 2026-07-06 - 7D Bridge And Limited Fixed-Prior Rollout Diagnostic

Decision: preserve full `7D` HDF5 actions for rollout candidates while keeping the existing `ACTION_PREFIX_DIM=4` offline proxy path unchanged.

Rationale: the previous rollout gate was red because the rollout path reused `4D` offline proxy records. Padding or inventing gripper/rotation values would make the diagnostic invalid. The local HDF5 demos already contain `7D` LIBERO actions, so the safe fix is to preserve those actions only for rollout readiness/diagnostics.

Outcome: readiness gate became green. A bounded fixed-prior diagnostic ran `30` simulator steps across ActionMap-style mean, fixed semantic target-prior TCA, and oracle upper-bound variants. Reward and success stayed zero for all variants. The result supports the action bridge enough to proceed with cautious diagnosis, but does not support a rollout performance claim.

Integrity note: no training, LoRA training, loss computation, GPU job, download, heavy VLA import, OpenVLA-OFT execution, benchmark rollout, or paper-grade claim occurred.

