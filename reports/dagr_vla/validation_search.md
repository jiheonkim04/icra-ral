# DAGR-VLA Validation Search

Date: `2026-07-14`

Proposal hash: `BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89`

Final decision: `VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING`

- closed-loop experiment happened: `False`
- lightweight validation training happened: `True`
- confirmatory-test tuning happened: `False`
- audit final decision: `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`
- search budget: `6 configs: residual alpha in {0.05, 0.10, 0.20} x route architecture in {linear, mlp}`
- tried configs: `6`
- selected config: `dagr_a020_route_mlp`
- selected residual alpha: `0.2`
- selected route architecture: `mlp`
- selected score: `0.8571740870493018`
- selected delta L2 p95: `0.008609326556324959`
- selected clean delta L2 p95: `0.00672802422195673`
- selected action validity: `1.0`

Score weights:

```json
{
  "action_validity_and_group_delta": 0.15,
  "clean_action_retention_and_bounded_deltas": 0.25,
  "compute_overhead": 0.1,
  "full_versus_proxy_and_ablation_distinction": 0.2,
  "route_predictability_above_majority": 0.3
}
```

Selected config:

```json
{
  "checkpoint_path": "reports\\dagr_vla\\validation_checkpoints\\dagr_a020_route_mlp.pt",
  "checkpoint_reload_max_abs_diff": 0.0,
  "config_id": "dagr_a020_route_mlp",
  "final_decision": "VALIDATION_CONFIG_PASS",
  "first_gradient_norms": {
    "residual": 0.00111672033395391,
    "route": 0.09741910273562175,
    "trunk": 0.05258959470359266
  },
  "hard_stop_reasons": [],
  "initial_delta_p95": 0.0,
  "loss_final_train": {
    "clean": 1.645963266128092e-06,
    "delta": 1.997219806071371e-05,
    "residual": 0.0011103303404524922,
    "route": 0.5111390352249146,
    "total": 0.5122515559196472
  },
  "loss_initial": {
    "clean": 0.0,
    "delta": 0.0,
    "residual": 0.0011733261635527015,
    "route": 0.7019369006156921,
    "total": 0.7031102180480957
  },
  "loss_validation": {
    "clean": 1.4742300891157356e-06,
    "delta": 2.084582047245931e-05,
    "residual": 0.0016336414264515042,
    "route": 0.634952962398529,
    "total": 0.6365888118743896
  },
  "proposal_hash": "BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89",
  "residual_alpha": 0.2,
  "route_architecture": "mlp",
  "route_metrics": {
    "gripper": {
      "accuracy": 0.7325,
      "accuracy_margin": 0.21000000000000008,
      "majority_accuracy": 0.5225,
      "mean_probability": 0.36505594849586487,
      "predicted_positive_fraction": 0.315
    },
    "rotation": {
      "accuracy": 0.58,
      "accuracy_margin": 0.05499999999999994,
      "majority_accuracy": 0.525,
      "mean_probability": 0.5029714703559875,
      "predicted_positive_fraction": 0.475
    },
    "translation": {
      "accuracy": 0.61,
      "accuracy_margin": 0.0675,
      "majority_accuracy": 0.5425,
      "mean_probability": 0.5142354369163513,
      "predicted_positive_fraction": 0.5125
    }
  },
  "score_terms": {
    "action_validity_and_group_delta": 0.9877009620623929,
    "clean_retention_and_bounded_delta": 0.9663598788902164,
    "compute_overhead": 0.95,
    "full_proxy_ablation_distinction": 0.36214486508694377,
    "route_predictability": 1.0,
    "total": 0.8571740870493018
  },
  "validation_metrics": {
    "action_validity": 1.0,
    "clean_delta_l2_p95": 0.00672802422195673,
    "delta_l2_mean": 0.0037601334042847157,
    "delta_l2_p95": 0.008609326556324959,
    "full_vs_shared_mean_l2": 0.0018881356809288263,
    "full_vs_static_mean_l2": 0.0017333129699406114,
    "gate_activation_fraction_by_group": {
      "gripper": 0.315,
      "rotation": 0.475,
      "translation": 0.5125
    },
    "gate_mean_by_group": {
      "gripper": 0.36505594849586487,
      "rotation": 0.5029714703559875,
      "translation": 0.5142354369163513
    }
  }
}
```

Tried configurations:

| config | decision | alpha | arch | route | clean | distinction | validity | compute | total |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dagr_a005_route_linear` | `VALIDATION_CONFIG_PASS` | 0.05 | `linear` | 1.0 | 0.9890046555083245 | 0.07109522606290566 | 0.9954376775505287 | 1.0 | 0.8107858607222416 |
| `dagr_a010_route_linear` | `VALIDATION_CONFIG_PASS` | 0.1 | `linear` | 1.0 | 0.9805900093633682 | 0.13894437309932162 | 0.9915783140542251 | 1.0 | 0.8216731240688402 |
| `dagr_a020_route_linear` | `VALIDATION_CONFIG_PASS` | 0.2 | `linear` | 0.8750000000000003 | 0.9586969427764416 | 0.2510074158626636 | 0.9840522584106242 | 1.0 | 0.7999835576282367 |
| `dagr_a005_route_mlp` | `VALIDATION_CONFIG_PASS` | 0.05 | `mlp` | 1.0 | 0.9925706017529592 | 0.08289859560380072 | 0.9971510672808758 | 0.95 | 0.8092950296511313 |
| `dagr_a010_route_mlp` | `VALIDATION_CONFIG_PASS` | 0.1 | `mlp` | 1.0 | 0.9842651654034853 | 0.18161258206926467 | 0.9943914743406432 | 0.95 | 0.8265475289158208 |
| `dagr_a020_route_mlp` | `VALIDATION_CONFIG_PASS` | 0.2 | `mlp` | 1.0 | 0.9663598788902164 | 0.36214486508694377 | 0.9877009620623929 | 0.95 | 0.8571740870493018 |

Next step: Freeze the selected DAGR config and train disk-reloadable policy identities for the five-policy comparison before Stage A.
