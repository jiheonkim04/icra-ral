# EAC-VLA Validation Search

Date: `2026-07-15`

Proposal hash: `A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E`

Final decision: `EAC_VALIDATION_SEARCH_SELECT_CONFIG_STAGE_A_MANIFEST_READY`

- closed-loop experiment happened: `False`
- training happened: `False`
- validation search happened: `True`
- confirmatory-test tuning happened: `False`
- confirmatory records used for tuning: `False`
- tried config count: `6`
- selected config: `eac_q33_aggressive_1_4_50`
- selected validation score: `0.7530415186081504`
- selected commitment counts: `{'1': 132, '4': 136, '50': 132}`
- selected policy calls per step proxy: `0.4216`
- selected oscillation fraction: `0.6388888888888888`
- selected risk exposure reduction proxy: `0.9032794643799159`

Validation score weights:

```json
{
  "clean_action_value_passthrough": 0.2,
  "latency_penalty": -0.05,
  "mechanism_activation": 0.15,
  "oscillation_penalty": -0.05,
  "risk_exposure_reduction_proxy": 0.45,
  "runtime_action_validity": 0.1
}
```

Tried configurations:

```json
[
  {
    "commitment_counts": {
      "1": 132,
      "4": 136,
      "50": 132
    },
    "commitment_map": {
      "long": 50,
      "medium": 4,
      "short": 1
    },
    "config_id": "eac_q33_aggressive_1_4_50",
    "max_commitment_share": 0.34,
    "oscillation_fraction": 0.6388888888888888,
    "policy_calls_per_step_proxy": 0.4216,
    "quantile_margin": 0.33,
    "risk_monotonicity_short_gt_long": true,
    "risk_summary_by_commitment": {
      "1": {
        "count": 132,
        "max": 1.0,
        "mean": 0.5626796837122628,
        "min": 0.30860313236146664,
        "p50": 0.5019669481835786,
        "p95": 0.9832699363175449
      },
      "4": {
        "count": 136,
        "max": 0.3085894120013512,
        "mean": 0.21484555177296905,
        "min": 0.13938054468316916,
        "p50": 0.2115970243684425,
        "p95": 0.29514483289083077
      },
      "50": {
        "count": 132,
        "max": 0.1364078421539268,
        "mean": 0.06894869902880432,
        "min": 0.0,
        "p50": 0.07300336823166828,
        "p95": 0.13082922764875465
      }
    },
    "score_components": {
      "clean_action_value_passthrough": 1.0,
      "latency_penalty": 0.4097959183673469,
      "mechanism_activation": 0.6599999999999999,
      "oscillation_penalty": 0.6388888888888888,
      "risk_exposure_reduction_proxy": 0.9032794643799159,
      "runtime_action_validity": 1.0
    },
    "validation_score": 0.7530415186081504
  },
  {
    "commitment_counts": {
      "2": 132,
      "50": 132,
      "8": 136
    },
    "commitment_map": {
      "long": 50,
      "medium": 8,
      "short": 2
    },
    "config_id": "eac_q33_balanced_2_8_50",
    "max_commitment_share": 0.34,
    "oscillation_fraction": 0.6388888888888888,
    "policy_calls_per_step_proxy": 0.21409999999999996,
    "quantile_margin": 0.33,
    "risk_monotonicity_short_gt_long": true,
    "risk_summary_by_commitment": {
      "2": {
        "count": 132,
        "max": 1.0,
        "mean": 0.5626796837122628,
        "min": 0.30860313236146664,
        "p50": 0.5019669481835786,
        "p95": 0.9832699363175449
      },
      "50": {
        "count": 132,
        "max": 0.1364078421539268,
        "mean": 0.06894869902880432,
        "min": 0.0,
        "p50": 0.07300336823166828,
        "p95": 0.13082922764875465
      },
      "8": {
        "count": 136,
        "max": 0.3085894120013512,
        "mean": 0.21484555177296905,
        "min": 0.13938054468316916,
        "p50": 0.2115970243684425,
        "p95": 0.29514483289083077
      }
    },
    "score_components": {
      "clean_action_value_passthrough": 1.0,
      "latency_penalty": 0.19806122448979588,
      "mechanism_activation": 0.6599999999999999,
      "oscillation_penalty": 0.6388888888888888,
      "risk_exposure_reduction_proxy": 0.8686327134161742,
      "runtime_action_validity": 1.0
    },
    "validation_score": 0.7480372153683441
  },
  {
    "commitment_counts": {
      "1": 100,
      "4": 200,
      "50": 100
    },
    "commitment_map": {
      "long": 50,
      "medium": 4,
      "short": 1
    },
    "config_id": "eac_q25_aggressive_1_4_50",
    "max_commitment_share": 0.5,
    "oscillation_fraction": 0.5583333333333333,
    "policy_calls_per_step_proxy": 0.38,
    "quantile_margin": 0.25,
    "risk_monotonicity_short_gt_long": true,
    "risk_summary_by_commitment": {
      "1": {
        "count": 100,
        "max": 1.0,
        "mean": 0.6333343630362923,
        "min": 0.3856193255130583,
        "p50": 0.5934759558830542,
        "p95": 1.0
      },
      "4": {
        "count": 200,
        "max": 0.38507629288331835,
        "mean": 0.2205002359634548,
        "min": 0.10466272874924337,
        "p50": 0.2115970243684425,
        "p95": 0.3526183492286971
      },
      "50": {
        "count": 100,
        "max": 0.1045453171824287,
        "mean": 0.05160458066624453,
        "min": 0.0,
        "p50": 0.05529888407633501,
        "p95": 0.10058396791023658
      }
    },
    "score_components": {
      "clean_action_value_passthrough": 1.0,
      "latency_penalty": 0.3673469387755102,
      "mechanism_activation": 0.5,
      "oscillation_penalty": 0.5583333333333333,
      "risk_exposure_reduction_proxy": 0.9301875318790702,
      "runtime_action_validity": 1.0
    },
    "validation_score": 0.7473003757401393
  },
  {
    "commitment_counts": {
      "2": 100,
      "50": 100,
      "8": 200
    },
    "commitment_map": {
      "long": 50,
      "medium": 8,
      "short": 2
    },
    "config_id": "eac_q25_balanced_2_8_50",
    "max_commitment_share": 0.5,
    "oscillation_fraction": 0.5583333333333333,
    "policy_calls_per_step_proxy": 0.1925,
    "quantile_margin": 0.25,
    "risk_monotonicity_short_gt_long": true,
    "risk_summary_by_commitment": {
      "2": {
        "count": 100,
        "max": 1.0,
        "mean": 0.6333343630362923,
        "min": 0.3856193255130583,
        "p50": 0.5934759558830542,
        "p95": 1.0
      },
      "50": {
        "count": 100,
        "max": 0.1045453171824287,
        "mean": 0.05160458066624453,
        "min": 0.0,
        "p50": 0.05529888407633501,
        "p95": 0.10058396791023658
      },
      "8": {
        "count": 200,
        "max": 0.38507629288331835,
        "mean": 0.2205002359634548,
        "min": 0.10466272874924337,
        "p50": 0.2115970243684425,
        "p95": 0.3526183492286971
      }
    },
    "score_components": {
      "clean_action_value_passthrough": 1.0,
      "latency_penalty": 0.17602040816326534,
      "mechanism_activation": 0.5,
      "oscillation_penalty": 0.5583333333333333,
      "risk_exposure_reduction_proxy": 0.8867347235593015,
      "runtime_action_validity": 1.0
    },
    "validation_score": 0.7373129385268558
  },
  {
    "commitment_counts": {
      "1": 160,
      "4": 80,
      "50": 160
    },
    "commitment_map": {
      "long": 50,
      "medium": 4,
      "short": 1
    },
    "config_id": "eac_q40_aggressive_1_4_50",
    "max_commitment_share": 0.4,
    "oscillation_fraction": 0.575,
    "policy_calls_per_step_proxy": 0.45799999999999996,
    "quantile_margin": 0.4,
    "risk_monotonicity_short_gt_long": true,
    "risk_summary_by_commitment": {
      "1": {
        "count": 160,
        "max": 1.0,
        "mean": 0.5140861597817086,
        "min": 0.26700433192315176,
        "p50": 0.4496966988668247,
        "p95": 0.9471965848771726
      },
      "4": {
        "count": 80,
        "max": 0.2656128329844134,
        "mean": 0.21210608461768957,
        "min": 0.16653962381014037,
        "p50": 0.2115970243684425,
        "p95": 0.2561825504548404
      },
      "50": {
        "count": 160,
        "max": 0.1656823621163232,
        "mean": 0.08357293267785074,
        "min": 0.0,
        "p50": 0.0872913030137839,
        "p95": 0.1591794310146856
      }
    },
    "score_components": {
      "clean_action_value_passthrough": 1.0,
      "latency_penalty": 0.44693877551020406,
      "mechanism_activation": 0.6,
      "oscillation_penalty": 0.575,
      "risk_exposure_reduction_proxy": 0.8720130410794238,
      "runtime_action_validity": 1.0
    },
    "validation_score": 0.7313089297102304
  },
  {
    "commitment_counts": {
      "2": 160,
      "50": 160,
      "8": 80
    },
    "commitment_map": {
      "long": 50,
      "medium": 8,
      "short": 2
    },
    "config_id": "eac_q40_balanced_2_8_50",
    "max_commitment_share": 0.4,
    "oscillation_fraction": 0.575,
    "policy_calls_per_step_proxy": 0.233,
    "quantile_margin": 0.4,
    "risk_monotonicity_short_gt_long": true,
    "risk_summary_by_commitment": {
      "2": {
        "count": 160,
        "max": 1.0,
        "mean": 0.5140861597817086,
        "min": 0.26700433192315176,
        "p50": 0.4496966988668247,
        "p95": 0.9471965848771726
      },
      "50": {
        "count": 160,
        "max": 0.1656823621163232,
        "mean": 0.08357293267785074,
        "min": 0.0,
        "p50": 0.0872913030137839,
        "p95": 0.1591794310146856
      },
      "8": {
        "count": 80,
        "max": 0.2656128329844134,
        "mean": 0.21210608461768957,
        "min": 0.16653962381014037,
        "p50": 0.2115970243684425,
        "p95": 0.2561825504548404
      }
    },
    "score_components": {
      "clean_action_value_passthrough": 1.0,
      "latency_penalty": 0.21734693877551023,
      "mechanism_activation": 0.6,
      "oscillation_penalty": 0.575,
      "risk_exposure_reduction_proxy": 0.8448017077173144,
      "runtime_action_validity": 1.0
    },
    "validation_score": 0.7305434215340159
  }
]
```

Reference baselines:

```json
{
  "fixed_short_replan_baseline_commitment_1": {
    "commitment": 1,
    "oscillation_fraction": 0.0,
    "policy_calls_per_step_proxy": 1.0,
    "risk_exposure_reduction_proxy": 1.0
  },
  "fixed_short_replan_baseline_commitment_2": {
    "commitment": 2,
    "oscillation_fraction": 0.0,
    "policy_calls_per_step_proxy": 0.5,
    "risk_exposure_reduction_proxy": 0.9795918367346937
  },
  "frozen_smolvla_fixed_queue": {
    "commitment": 50,
    "oscillation_fraction": 0.0,
    "policy_calls_per_step_proxy": 0.019999999999999997,
    "risk_exposure_reduction_proxy": 0.0
  }
}
```

Hard stop reasons:
- none

Next step: Freeze the EAC Stage A matched manifest and preflight the five policy identities.
