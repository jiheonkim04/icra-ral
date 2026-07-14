# MARC-VLA Policy Identity Training

Date: `2026-07-15 KST`

Final decision: `MARC_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY`

- closed-loop experiment happened: `False`
- confirmatory-test identities used: `False`
- training happened: `True`
- Stage A allowed: `True`
- config: `marc_a020_gate_mlp`
- seed: `101`
- epochs: `250`
- checkpoint root: `runs\marc_vla_checkpoints\marc_a020_gate_mlp`
- static mixture remains live reviewer-killer: `False`

Variants:

| variant | decision | reload | delta p95 | clean p95 | validity | target L2 | checkpoint |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `openvla_oft_l1_proxy` | `MARC_POLICY_CHECKPOINT_VERIFIED` | `True` | 0.2307613492012024 | 0.18917657434940338 | 1.0 | 0.13093115008724351 | `runs\marc_vla_checkpoints\marc_a020_gate_mlp\openvla_oft_l1_proxy\seed_101` |
| `marc_full` | `MARC_POLICY_CHECKPOINT_VERIFIED` | `True` | 0.010693175718188286 | 0.008896797895431519 | 1.0 | 0.08629934170250374 | `runs\marc_vla_checkpoints\marc_a020_gate_mlp\marc_full\seed_101` |
| `marc_no_disagreement_gate_ablation` | `MARC_POLICY_CHECKPOINT_VERIFIED` | `True` | 0.12246084958314896 | 0.10007624328136444 | 1.0 | 0.10173239223806452 | `runs\marc_vla_checkpoints\marc_a020_gate_mlp\marc_no_disagreement_gate_ablation\seed_101` |
| `static_l1_mixture_baseline` | `MARC_POLICY_CHECKPOINT_VERIFIED` | `True` | 0.07999999821186066 | 0.07970031350851059 | 1.0 | 0.09530899882343835 | `runs\marc_vla_checkpoints\marc_a020_gate_mlp\static_l1_mixture_baseline\seed_101` |

Full-policy distinctions:

```json
{
  "marc_full_vs_marc_no_disagreement_gate_ablation_mean_l2": 0.04372206702828407,
  "marc_full_vs_openvla_oft_l1_proxy_mean_l2": 0.08430124074220657,
  "marc_full_vs_static_l1_mixture_baseline_mean_l2": 0.032826922833919525
}
```

Next step: Freeze the MARC Stage A matched manifest before any rollout.
