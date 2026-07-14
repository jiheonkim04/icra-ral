# EAC-VLA Stage A Runner Validation

Date: `2026-07-15`

Final decision: `EAC_STAGE_A_RUNNER_VALIDATED_READY_FOR_ROLLOUT`

- closed-loop experiment happened: `False`
- training happened: `False`
- validation search happened: `False`
- confirmatory-test tuning happened: `False`
- planned episode count: `50`
- policy count: `5`
- action values modified: `False`
- rollout allowed: `True`

Calibration:

```json
{
  "aac_dispersion_quantile_thresholds": {
    "high": 0.2708144477625418,
    "low": 0.10427305027305488,
    "quantile_margin": 0.33
  },
  "confirmatory_test_tuning_happened": false,
  "dispersion_norm_summary": {
    "count": 400,
    "max": 1.0,
    "mean": 0.26728377216858673,
    "min": 0.0,
    "p50": 0.15717823462461084,
    "p95": 0.996716754592918
  },
  "eac_quantile_thresholds": {
    "high": 0.3085939397201893,
    "low": 0.1383995528485192,
    "quantile_margin": 0.33
  },
  "frozen_validation_only": true,
  "normalizer": {
    "dispersion_p05": 4.948173661796488e-05,
    "dispersion_p95": 0.0007983036317792467,
    "transition_p05": 0.023662913284092264,
    "transition_p95": 0.2050179072447333
  },
  "risk_summary": {
    "count": 400,
    "max": 1.0,
    "mean": 0.2814848539073616,
    "min": 0.0,
    "p50": 0.2115970243684425,
    "p95": 0.827539654595269
  },
  "risk_weights": {
    "dispersion": 0.67,
    "transition": 0.33
  },
  "selected_config": {
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
  "source_canonical_artifact": "reports/canonical_frozen_base_prediction_artifact.json",
  "source_canonical_artifact_sha256": "26297b56850269849cb51b8b6e21a8d53748f69ec249aa2c03ceb77221bbc6a4",
  "transition_norm_summary": {
    "count": 400,
    "max": 1.0,
    "mean": 0.3103173531951773,
    "min": 0.0,
    "p50": 0.2298368169320893,
    "p95": 0.999334217896133
  },
  "validation_frame_count": 400
}
```

Policy validation records:

```json
[
  {
    "action_values_modified": false,
    "base_chunk_sha256": "7f9288defee3832b11b4f2ad17b1798f4d5123efbe31d774955891a497554c31",
    "commitment": 50,
    "policy": "frozen_smolvla_fixed_queue",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "7f9288defee3832b11b4f2ad17b1798f4d5123efbe31d774955891a497554c31",
    "prefix_shape": [
      50,
      7
    ],
    "risk": {
      "chunk_sample_count": 1,
      "dispersion_norm": 0.0,
      "first_transition_l2": 0.07337958999056414,
      "first_two_dispersion": 0.0,
      "risk": 0.09046623395822438,
      "transition_norm": 0.2741401029037102
    },
    "role": "base",
    "runtime_sample_count": 1,
    "scheduler": "fixed_commitment",
    "scheduler_value": null,
    "thresholds": null
  },
  {
    "action_values_modified": false,
    "base_chunk_sha256": "7f9288defee3832b11b4f2ad17b1798f4d5123efbe31d774955891a497554c31",
    "commitment": 2,
    "policy": "aac_entropy_proxy",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "3af24bd6ab6ee19858b9e388317b2efc777b18692645dde44a65533ea9c46263",
    "prefix_shape": [
      2,
      7
    ],
    "risk": {
      "chunk_sample_count": 2,
      "dispersion_norm": 1.0,
      "first_transition_l2": 0.10158403790212397,
      "first_two_dispersion": 0.001237947248756342,
      "risk": 0.811788050951225,
      "transition_norm": 0.42966076045825746
    },
    "role": "closest_prior_proxy",
    "runtime_sample_count": 2,
    "scheduler": "dispersion_only_quantile_proxy",
    "scheduler_value": 1.0,
    "thresholds": {
      "high": 0.2708144477625418,
      "low": 0.10427305027305488,
      "quantile_margin": 0.33
    }
  },
  {
    "action_values_modified": false,
    "base_chunk_sha256": "7f9288defee3832b11b4f2ad17b1798f4d5123efbe31d774955891a497554c31",
    "commitment": 1,
    "policy": "eac_full",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "19485d60fa982b95551269a7429426a53a158c3d6faee82f9112f18e2833f1a4",
    "prefix_shape": [
      1,
      7
    ],
    "risk": {
      "chunk_sample_count": 2,
      "dispersion_norm": 1.0,
      "first_transition_l2": 0.10158403790212397,
      "first_two_dispersion": 0.001237947248756342,
      "risk": 0.811788050951225,
      "transition_norm": 0.42966076045825746
    },
    "role": "ours",
    "runtime_sample_count": 2,
    "scheduler": "selected_validation_config",
    "scheduler_value": 0.811788050951225,
    "thresholds": {
      "high": 0.3085939397201893,
      "low": 0.1383995528485192,
      "quantile_margin": 0.33
    }
  },
  {
    "action_values_modified": false,
    "base_chunk_sha256": "7f9288defee3832b11b4f2ad17b1798f4d5123efbe31d774955891a497554c31",
    "commitment": 1,
    "policy": "eac_no_calibration_no_hysteresis_ablation",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "19485d60fa982b95551269a7429426a53a158c3d6faee82f9112f18e2833f1a4",
    "prefix_shape": [
      1,
      7
    ],
    "risk": {
      "chunk_sample_count": 2,
      "dispersion_norm": 1.0,
      "first_transition_l2": 0.10158403790212397,
      "first_two_dispersion": 0.001237947248756342,
      "risk": 0.811788050951225,
      "transition_norm": 0.42966076045825746
    },
    "role": "key_ablation",
    "runtime_sample_count": 2,
    "scheduler": "raw_risk_fixed_threshold_ablation",
    "scheduler_value": 0.811788050951225,
    "thresholds": {
      "high": 0.6666666666666666,
      "low": 0.3333333333333333
    }
  },
  {
    "action_values_modified": false,
    "base_chunk_sha256": "7f9288defee3832b11b4f2ad17b1798f4d5123efbe31d774955891a497554c31",
    "commitment": 1,
    "policy": "fixed_short_replan_baseline",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "19485d60fa982b95551269a7429426a53a158c3d6faee82f9112f18e2833f1a4",
    "prefix_shape": [
      1,
      7
    ],
    "risk": {
      "chunk_sample_count": 1,
      "dispersion_norm": 0.0,
      "first_transition_l2": 0.07337958999056414,
      "first_two_dispersion": 0.0,
      "risk": 0.09046623395822438,
      "transition_norm": 0.2741401029037102
    },
    "role": "simple_killer",
    "runtime_sample_count": 1,
    "scheduler": "fixed_commitment",
    "scheduler_value": null,
    "thresholds": null
  }
]
```

Errors:
- none

Next step: Launch the frozen EAC Stage A rollout with the validated runner.
