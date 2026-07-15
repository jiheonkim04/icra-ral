# LIFT-VLA Prototype Protocol

Date: 2026-07-15 KST

Proposal hash:
`3D263AA6FF73B342523D85AD4854145AF4D79DE2B90C6119F417D37A8B08F55F`

Preregistration: `reports/lift_vla/preregistration.md`

Decision: `LIFT_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`

Reviewer status: `APPROVE_WITH_FIXED_EMPIRICAL_RISKS`

## Environment

- repository: `C:\Users\jiheo\tca_map`;
- branch: `codex/autonomous-until-paper-governance-v2`;
- canonical runtime: WSL Ubuntu 22.04;
- Python:
  `/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python`;
- GPU: NVIDIA GeForce RTX 5080, 16GB;
- checkpoint: `C:\assets\checkpoints\smolvla` mounted in WSL;
- LIBERO: `C:\assets\repos\LIBERO` mounted in WSL;
- RoboSuite: `C:\assets\repos\robosuite` mounted in WSL;
- no download, training, LoRA, QLoRA, or cloud execution.

## Frozen Command

From Windows PowerShell:

```powershell
wsl -d Ubuntu-22.04 bash -lc "cd /mnt/c/Users/jiheo/tca_map && /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_lift_vla_stage0.py --mode audit"
```

Unit tests use:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests/test_lift_vla.py -q
```

## Required Outputs

- `reports/lift_vla/counterfactual_manifest.json`
- `reports/lift_vla/discovery_thresholds.json`
- `reports/lift_vla/stage_0_result.json`
- `reports/lift_vla/stage_0_result.md`
- `reports/lift_vla/implementation_blocker.json` on implementation failure

All JSON writes are atomic. Partial work may resume only missing manifest keys.

## Fixed Inputs

- `libero_goal` target partitions: discovery `[0,1,2,3]`, validation `[4,5,6]`,
  confirmatory `[7,8,9]`;
- initial action-noise seed derived by SHA-256 from the full manifest key;
- empty task text exactly `""`;
- guidance scales `[1.25, 1.50, 2.00]`;
- identity scale `1.00` for audit only;
- one frozen SmolVLA checkpoint;
- ten Euler steps;
- native chunk `[1,50,32]`;
- canonical policy chunk `[1,50,7]`.

## Implementation Boundary

The implementation may:

- subclass or wrap the local SmolVLA sampling path;
- construct conditioned and empty-language prefix caches;
- expose per-step vector fields and counts;
- implement Base, native-space CAG, full LIFT, and matched-compute ablation;
- schedule branches sequentially for memory if numerically equivalent;
- persist metrics and manifests.

The implementation may not:

- edit installed LeRobot source or checkpoint weights;
- change tokenizer, null prompt, step count, integrator, noise, postprocessor, or
  7D bridge;
- add training, adapters, LoRA, QLoRA, schedules, gates, losses, or policies;
- decode confirmatory policy observations or outcomes in Stage 0;
- use old cross-scene offline counterfactual pairs;
- tune thresholds or scales after validation or confirmatory outcomes.

## Stage 0 Execution Boundary

The audit command may use discovery and validation BDDL/reset identities. It may
hash confirmatory metadata and initial-state identities solely to freeze and
prove partition separation. It must report
`confirmatory_policy_observations_decoded = 0` and
`confirmatory_policy_actions_computed = 0`.

Baseline-only development rollouts for headroom are allowed only after source,
shape, identity, action-validity, and compute gates pass. No LIFT scale may be
selected in Stage 0.

## Hard Stops

Stop immediately on:

- invalid or unscoreable counterfactual manifest;
- any partition overlap;
- native shape, output shape, or flow-step mismatch;
- Base identity error above `1e-5`;
- invalid action fraction below `1.0`;
- peak allocated memory at or above `15.5 GiB`;
- CPU fallback or CUDA OOM;
- LIFT latency above `4x` Base;
- Base failure below `0.20` or CAG residual failure below `0.10`;
- practical equivalence under the frozen discovery-threshold formulas.

## No Rescue

If Stage 0 stops, do not change task partitions, null prompt, scales, thresholds,
flow steps, noise coupling, postprocessing location, baseline list, or ablation.
An implementation bug may be repaired only when the repair restores this exact
protocol and preserves all raw artifacts.

## Next Step

Implement the four-policy flow sampler and Stage 0 runner, run unit tests, then
execute the frozen audit command. Validation search remains forbidden until a
valid `LIFT_STAGE_0_PASS_TO_BOUNDED_VALIDATION` result.

