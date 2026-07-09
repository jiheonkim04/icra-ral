# Next Actions

Date: 2026-07-09 KST

Current decision: `EXPERT_REPLAY_BLOCKED`

## Immediate Next Action

Install or activate the local `mujoco` Python dependency for LIBERO/RoboSuite in the `tca_map` environment, then rerun this same replay bridge; do not start a new method.

Do not start a new method until this replay bridge is green. If exact-init replay is blocked by simulator dependencies, fix the simulator/import stack first and rerun this same bridge; do not switch routes.
