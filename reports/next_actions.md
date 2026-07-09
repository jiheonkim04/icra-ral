# Next Actions

Date: 2026-07-09 KST

Current decision: `FEATURE_PATH_MISMATCH`

## Immediate Next Action

Fix the live closed-loop feature schema so replay uses HDF5-compatible ee_states features, then rerun teacher-forced and replay diagnostics before any method work.

Do not start a new method unless the control diagnosis decision is `READY_FOR_METHOD_AFTER_CONTROL_DIAGNOSIS`.
