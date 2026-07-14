# DAGR-VLA Policy Identity Training

Date: `2026-07-14 KST`

Final decision: `DAGR_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY`

- closed-loop experiment happened: `False`
- confirmatory-test identities used: `False`
- training happened: `True`
- Stage A allowed: `True`
- config: `dagr_a020_route_mlp`
- seed: `101`
- epochs: `250`
- checkpoint root: `runs\dagr_vla_checkpoints\dagr_a020_route_mlp`

Variants:

| variant | decision | reload | delta p95 | clean p95 | validity | checkpoint |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `dagr_full` | `DAGR_POLICY_CHECKPOINT_VERIFIED` | `True` | 0.008576558902859688 | 0.006687765941023827 | 1.0 | `runs\dagr_vla_checkpoints\dagr_a020_route_mlp\dagr_full\seed_101` |
| `dam_static_component_proxy` | `DAGR_POLICY_CHECKPOINT_VERIFIED` | `True` | 0.016259152442216873 | 0.009240802377462387 | 1.0 | `runs\dagr_vla_checkpoints\dagr_a020_route_mlp\dam_static_component_proxy\seed_101` |
| `dagr_no_dynamic_route_ablation` | `DAGR_POLICY_CHECKPOINT_VERIFIED` | `True` | 0.006147781852632761 | 0.0048079947009682655 | 1.0 | `runs\dagr_vla_checkpoints\dagr_a020_route_mlp\dagr_no_dynamic_route_ablation\seed_101` |
| `gripper_transition_heuristic` | `NONTRAINABLE_HEURISTIC_READY` | `True` | 0.0 | 0.0 | 1.0 | `runs\dagr_vla_checkpoints\dagr_a020_route_mlp\gripper_transition_heuristic\seed_101` |

Distinction:

```json
{
  "full_vs_shared_delta_l2_mean_abs": 0.00047564832493662834,
  "full_vs_static_delta_l2_mean_abs": 0.0035665808245539665
}
```

Next step: Freeze the DAGR Stage A matched manifest before any rollout.
