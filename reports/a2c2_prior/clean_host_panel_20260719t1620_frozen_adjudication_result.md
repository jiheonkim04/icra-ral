# A2C2 Official-Prior-First Problem Verification

Date: `2026-07-19 KST`

Fidelity label: `MECHANISM_FAITHFUL_A2C2_LOCAL_PORT`

Final decision: `NO_DIAGNOSTIC_HEADROOM`

```json
{
  "conditions": {
    "BASE_DELAYED_E40_D10": {
      "execution_horizon": 40,
      "inference_delay": 10
    },
    "BASE_STANDARD_E10_D0": {
      "execution_horizon": 10,
      "inference_delay": 0
    },
    "PRIOR_DELAYED_E40_D10": {
      "execution_horizon": 40,
      "inference_delay": 10
    }
  },
  "counts": {
    "clean_successes": 10,
    "clean_to_delayed_failure_count": 7,
    "clean_to_prior_residual_count": 7,
    "delayed_base_successes": 4,
    "delayed_prior_successes": 3,
    "prior_recovery_count": 1,
    "prior_regression_count": 2
  },
  "date": "2026-07-19 KST",
  "fidelity_label": "MECHANISM_FAITHFUL_A2C2_LOCAL_PORT",
  "final_decision": "NO_DIAGNOSTIC_HEADROOM",
  "frozen_panel": {
    "official_init_state_ids": [
      0,
      1,
      2,
      3,
      4
    ],
    "suite": "libero_spatial",
    "tasks": [
      {
        "global_task_index": 34,
        "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "task_id": 0
      },
      {
        "global_task_index": 31,
        "instruction": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "task_id": 4
      },
      {
        "global_task_index": 36,
        "instruction": "pick up the black bowl next to the plate and place it on the plate",
        "task_id": 8
      }
    ]
  },
  "gates": {
    "base_competent": true,
    "diagnostic_or_task_headroom": false,
    "prior_improves": false,
    "prior_saturates": false,
    "repeatable_problem": true,
    "residual_is_not_infrastructure_artifact": true,
    "residual_is_not_single_accidental_episode": true,
    "residual_remains": false
  },
  "identity_lists": {
    "clean_to_delayed_failures": [
      [
        0,
        0
      ],
      [
        0,
        4
      ],
      [
        4,
        1
      ],
      [
        4,
        4
      ],
      [
        8,
        0
      ],
      [
        8,
        1
      ],
      [
        8,
        3
      ]
    ],
    "clean_to_prior_residuals": [
      [
        0,
        0
      ],
      [
        0,
        3
      ],
      [
        0,
        4
      ],
      [
        4,
        1
      ],
      [
        4,
        4
      ],
      [
        8,
        1
      ],
      [
        8,
        3
      ]
    ],
    "prior_recoveries": [
      [
        8,
        0
      ]
    ],
    "prior_regressions": [
      [
        0,
        3
      ],
      [
        8,
        2
      ]
    ]
  },
  "job_classification": "REPORT_ONLY",
  "next_step": "Do not design or execute Ours for this thesis; follow the frozen pivot-closure rule.",
  "official_commit": "54dd088302a0ef3f50c4add3ec927ab94d76a406",
  "ours_designed_or_executed": false,
  "schema_version": 1,
  "source_reports": {
    "base_rollout": "reports/a2c2_prior/clean_host_panel_20260719t1620_base_closed_loop_result.json",
    "cache": "reports/a2c2_prior/cached_feature_result.json",
    "prior_rollout": "reports/a2c2_prior/clean_host_panel_20260719t1620_prior_closed_loop_result.json",
    "training": "reports/a2c2_prior/prior_module_training_result.json"
  },
  "thresholds": {
    "base_competence": "clean successes >=8/15 and >=1 success on every task",
    "prior_improvement": "prior-delayed >=2/15, >=2 delayed failures recovered, <=1 delayed success regressed, nonzero live prior forwards and correction",
    "prior_saturation": "prior improves and prior >= clean-1 or <=1 matched clean-success/prior-failure remains",
    "repeatable_problem": "clean-delayed >=3/15, >=3 matched clean-to-delayed failures, spanning >=2 tasks",
    "residual": "prior improves and clean-prior >=2/15 with >=2 matched residuals spanning >=2 tasks"
  },
  "validity": {
    "infrastructure_valid": true,
    "manifest_valid": true
  }
}
```
