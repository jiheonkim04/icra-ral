# Gripper-Close Compatibility Diagnostic Plan

This report defines a narrow planning gate for the next learned-policy compatibility hypothesis.

The offline LIBERO HDF5 adapter reproduction check found that the first demonstration action is reproduced exactly by appending the gripper-close command to the 6D policy-sized action, while the zero-hold gripper default mismatches that demonstration action. That is a compatibility clue, not rollout evidence.

The planner added in `scripts/96_plan_gripper_close_compat_diagnostic.ps1` reads the offline reproduction report, the previous adapter-strategy diagnostic report if it exists, and the rollout bridge source. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

If no previous close-strategy rollout diagnostic exists, the planner may authorize a separately gated one-task diagnostic with:

- task-local gate: `ALLOW_GRIPPER_CLOSE_COMPAT_DIAGNOSTIC=1`,
- one LIBERO task,
- at most 10 steps,
- CPU/WSL topology only,
- gripper strategy `policy_6d_delta_pose_plus_gripper_close`,
- diagnostic evidence label only.

If an equivalent close-strategy diagnostic already ran and still produced zero reward and zero diagnostic success, the planner must not recommend rerunning the same variant. It should reduce scope toward an HDF5-aligned compatibility check covering task selection, initial-state convention, replay/action-sign assumptions, and demonstration-to-rollout mismatch.

This plan does not unblock rollout scaling, multi-seed evaluation, OpenVLA-OFT execution, full fine-tuning, standard-success claims, SOTA claims, or paper-grade claims.
