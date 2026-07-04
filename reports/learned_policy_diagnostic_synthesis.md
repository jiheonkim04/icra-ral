# Learned-Policy Diagnostic Synthesis

This report records the report-only synthesis after the bounded learned-policy diagnostic ladder:

- zero-action comparison,
- adapter strategy,
- action scale,
- prompt format,
- camera source,
- state sufficiency.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\92_generate_learned_policy_diagnostic_synthesis.ps1
```

The synthesis reads existing diagnostic reports only. It does not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

Runtime outputs are ignored:

- `reports\learned_policy_diagnostic_synthesis_report.json`,
- `reports\learned_policy_diagnostic_synthesis_report.md`.

## Current Local Result

Latest synthesis result: `no_go_rollout_scaling`.

The diagnostic ladder is complete:

- zero-action comparison: passed, learned-policy reward `0.0`, diagnostic success `0.0`,
- adapter strategy: passed, best variant `policy_6d_delta_pose_plus_gripper_close`, reward `0.0`, diagnostic success `0.0`,
- action scale: passed, best variant `1.0`, reward `0.0`, diagnostic success `0.0`,
- prompt format: passed, best variant `bddl_language_period`, reward `0.0`, diagnostic success `0.0`,
- camera source: passed, best variant `all_agentview`, reward `0.0`, diagnostic success `0.0`,
- state sufficiency: passed, best variant `eef_pos_zero_rot`, reward `0.0`, diagnostic success `0.0`.

Synthesis decision:

- positive diagnostic signal found: false,
- rollout scaling ready: false,
- paper-grade claim ready: false,
- benchmark claim: false,
- SOTA claim: false.

No-go reason:

```text
All bounded learned-policy diagnostic axes completed, but none produced nonzero reward or diagnostic success. Every diagnostic report keeps ready_for_rollout_scaling=false.
```

Recommended next step:

Create a bounded environment-policy compatibility audit focused on task/checkpoint alignment, action convention, and observation convention before another one-task diagnostic. Do not scale learned-policy rollouts from the current evidence.

## Execution-First Diagnostic Update

Additional bounded learned-policy diagnostics were run after the bridge wiring and zero-reward result:

- gripper strategy diagnostic: completed 3 variants; best strategy `policy_6d_delta_pose_plus_gripper_close`; diagnostic success `0.0`; reward `0.0`,
- action scale diagnostic: completed scales `0.25`, `0.5`, and `1.0`; best scale `1.0`; diagnostic success `0.0`; reward `0.0`,
- prompt format diagnostic: completed 3 variants; best prompt `bddl_language_period`; diagnostic success `0.0`; reward `0.0`,
- camera source diagnostic: completed 3 variants; best alias strategy `all_agentview`; diagnostic success `0.0`; reward `0.0`,
- state sufficiency diagnostic: completed 3 variants; best state adapter `eef_pos_zero_rot`; diagnostic success `0.0`; reward `0.0`,
- HDF5 init-state learned-policy recheck: passed execution on one LIBERO task for 5 steps with the demo initial state set in the environment; success `false`; reward `0.0`.

The init-state recheck used local SmolVLA policy inference on CPU and the local LIBERO/RoboSuite environment. It reported:

- policy action shape: `[1, 6]`,
- environment action dimension: `7`,
- explicit adapter: `policy_6d_delta_pose_plus_gripper_close`,
- gripper value: `-1.0`,
- action scale: `1.0`,
- implicit padding: false,
- truncation: false,
- HDF5 initial state set in environment: true.

Concrete failure diagnosis:

The zero-reward outcome is not explained by a simple action-shape bridge failure. The explicit 6D policy-action to 7D environment-action adapter ran, gripper/scale/prompt/camera/state variants executed, and HDF5 demo initial-state replay/recheck succeeded without silent padding or truncation. The current evidence points instead to checkpoint/task/action-stat provenance, policy competence under the current LIBERO setup, or horizon/task difficulty mismatch. Do not scale learned-policy rollouts or make benchmark/paper claims from this checkpoint state.

Recommended next execution-first step:

Run the fixed-integrity ActionMap vs TCA-Map tiny offline training/evaluation comparison on real LIBERO HDF5 snippets. This next task should produce training loss and offline proxy metrics, not another planner-only gate, unless execution is directly blocked.
