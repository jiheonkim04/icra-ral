# DAGR-VLA Prototype Protocol

Date: 2026-07-14 KST

Method: `DAGR-VLA`.

Proposal hash: `BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89`

## Prototype Policies

The first serious comparison contains exactly five policies:

1. `frozen_smolvla`
   - Frozen official SmolVLA policy.

2. `dam_static_component_proxy`
   - Closest external-prior proxy.
   - Static arm/gripper component-weighted residual adapter.
   - Faithful transparent local proxy, not an official DAM-VLA reproduction.

3. `dagr_full`
   - Frozen SmolVLA plus identity-preserving dynamic group route gates and clipped group residuals.
   - Must use the selected validation config and disk-reloaded checkpoint.

4. `dagr_no_dynamic_route_ablation`
   - Same residual target source and comparable parameter budget.
   - Removes group-specific dynamic route gates and uses a shared residual intervention.

5. `gripper_transition_heuristic`
   - One strongest simple reviewer-killer baseline.
   - Uses only a bounded gripper timing bias near predicted gripper transitions.

No additional internal controls may precede this five-policy comparison unless Stage 0 exposes a concrete implementation ambiguity that would otherwise invalidate one of the five policies.

## Stage 0 Command

Planned command:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_dagr_vla_development.py --mode audit
```

Stage 0 writes:

- `reports/dagr_vla/development_audit.json`
- `reports/dagr_vla/development_audit.md`
- `reports/dagr_vla/route_label_manifest.json`
- `reports/dagr_vla/split_manifest.json`

Stage 0 must not train a policy or launch closed-loop rollout.

## Validation Search Command

Planned command after Stage 0 pass:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_dagr_vla_development.py --mode validation
```

The bounded search may try at most the six preregistered configurations in `reports/dagr_vla/preregistration.md`. It must save all tried configurations and negative results.

## Training Artifacts

Every trained policy must have:

- config JSON;
- training seed;
- source split manifest;
- route-label manifest;
- checkpoint path;
- checksum;
- base checkpoint identity;
- dataset identity;
- validation metrics;
- action-delta metrics;
- route activation metrics;
- disk-reload verification.

Stage A cannot begin until `dagr_full`, `dam_static_component_proxy`, and `dagr_no_dynamic_route_ablation` have disk-reloadable identities or the policy is explicitly nontrainable by design.

## Closed-Loop Manifest

Stage A and Stage B must use matched paired manifests:

- identical task keys across policies;
- identical reset identities across policies;
- no overlap with validation identities;
- no duplicate `(policy, task, reset)` keys;
- official LIBERO success condition;
- no post-result task or reset selection.

The manifest is frozen before each stage begins.

## Metrics

Primary:

- official closed-loop task success;
- task-balanced success;
- paired full-minus-baseline success deltas.

Secondary:

- paired wins/losses/ties;
- paired bootstrap confidence interval;
- relative failure-rate reduction;
- per-task success;
- clean-retention success;
- route activation by group;
- translation, rotation, and gripper delta from Base;
- residual norm;
- gate values;
- action validity;
- latency;
- VRAM;
- training time.

## Resume Policy

For long-running WSL training or rollout:

- run detached;
- save Linux PID;
- save heartbeat/status JSON;
- save stdout and stderr logs;
- save partial result JSON;
- save exact resume command;
- resume only missing `(policy, task, reset)` keys after interruption.

## Scientific Decisions

DAGR cannot become a paper candidate unless `dagr_full` beats Base, the DAM-style static component proxy, the no-dynamic-route ablation, and the gripper-transition heuristic under the matched protocol.

If `dam_static_component_proxy` wins, DAGR is not a useful local extension of the closest prior.

If `dagr_no_dynamic_route_ablation` wins, the dynamic route mechanism is not useful.

If `gripper_transition_heuristic` wins, the result is explained by simple gripper timing.

## Nonretroactivity

DAGR cannot change MTF, RAC, CAVM, RCV, PSE, or earlier results. All prior kills and non-GO decisions remain fixed.

