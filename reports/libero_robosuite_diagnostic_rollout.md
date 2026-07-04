# LIBERO/RoboSuite Diagnostic Rollout

This rung moves beyond toy MuJoCo plumbing into one bounded LIBERO/RoboSuite diagnostic rollout.

Run the planner first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\64_plan_libero_robosuite_diagnostic_rollout.ps1
```

If the planner is green, run the bounded diagnostic with a task-local gate:

```powershell
$env:ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT="1"
powershell -ExecutionPolicy Bypass -File scripts\65_bounded_libero_robosuite_diagnostic_rollout.ps1
Remove-Item Env:\ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT -ErrorAction SilentlyContinue
```

Scope:

- uses the existing WSL venv at `/home/jiheon/.venvs/tca_map_sim`,
- uses the local official LIBERO and RoboSuite source checkouts,
- uses at most 5 LIBERO tasks, one diagnostic episode per task, and at most 5 zero-action steps per task,
- uses no learned policy, no VLA inference, no training, no GPU job, no downloads, no OpenVLA-OFT, no multi-seed evaluation, and no paper claim.

This is simulator diagnostic evidence only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.

## Current Local Result

The bounded local diagnostic has passed on WSL with:

- WSL venv: `/home/jiheon/.venvs/tca_map_sim`,
- LIBERO source: `C:\assets\repos\LIBERO`,
- RoboSuite source: `C:\assets\repos\robosuite` checked out to the LIBERO-compatible `v1.4.0` tag,
- LIBERO data root: `C:\assets\data\libero`,
- MuJoCo Python package aligned to `mujoco==2.3.7` in the WSL venv,
- local diagnostic: 1 `libero_10` task, 3 zero-action steps, 64x64 offscreen observation.

The runner created a LIBERO/RoboSuite environment, reset it, stepped it three times with a zero action, observed a finite `agentview_image`, and closed the environment. It did not run learned policy inference, train, use GPU, download assets, execute OpenVLA-OFT, run multi-seed evaluation, or make paper claims.

The WSL venv dependency fixes were limited to simulator readiness and were applied after green risk assessments. They include the LIBERO/RoboSuite-compatible runtime packages needed for this diagnostic path, including `bddl==1.0.1`, `future==0.18.2`, `easydict==1.9`, `matplotlib==3.5.3`, `numpy==1.22.4`, `cloudpickle==2.1.0`, `gym==0.25.2`, and `mujoco==2.3.7`.

Benchmark rollout, learned-policy rollout, standard-success reporting, multi-seed rollout, OpenVLA-OFT execution, and paper-grade claims remain blocked behind separate risk assessment and policy gates.
