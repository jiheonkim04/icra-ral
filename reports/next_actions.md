# Next Actions

Date: 2026-07-09 KST

Current decision: `OFFICIAL_VLA_RECIPE_REPRODUCTION_REQUIRED`

## Immediate Next Action

Stop custom SmolVLA 7D adapter iteration. The next valid VLA step is official SmolVLA/LeRobot/OpenVLA-style baseline reproduction with official preprocessing, normalization, action/gripper conventions, and eval/replay stack.

Do not start a new RA-L method unless an official baseline succeeds first. If official reproduction is not feasible under local constraints, use `STOP_VLA_METHOD_SEARCH_UNDER_CURRENT_SETUP`.
