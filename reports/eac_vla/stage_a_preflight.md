# EAC-VLA Stage A Manifest And Preflight

Date: `2026-07-15`

Final decision: `EAC_STAGE_A_PREFLIGHT_PASS_RUNNER_IMPLEMENTATION_PENDING`

- closed-loop experiment happened: `False`
- training happened: `False`
- validation search happened: `False`
- confirmatory-test tuning happened: `False`
- planned episode count: `50`
- paired cases per policy: `10`
- reset seeds: `[20261211, 20261212]`
- policies: `['frozen_smolvla_fixed_queue', 'aac_entropy_proxy', 'eac_full', 'eac_no_calibration_no_hysteresis_ablation', 'fixed_short_replan_baseline']`
- canonical payload sha256: `63E96D0629F3D34E4801EB1084D094CB287EC4F2F2FCD96373981787EDA9954C`

Policy identities:

```json
[
  {
    "commitment": 50,
    "policy": "frozen_smolvla_fixed_queue",
    "proxy_or_reproduction_label": "unmodified frozen SmolVLA action values with fixed full queue commitment",
    "role": "base",
    "scheduler": "fixed_commitment"
  },
  {
    "commitment": 8,
    "commitment_map": {
      "long": 50,
      "medium": 8,
      "short": 2
    },
    "policy": "aac_entropy_proxy",
    "proxy_or_reproduction_label": "faithful transparent local proxy, not an official AAC reproduction",
    "quantile_margin": 0.33,
    "role": "closest_prior_proxy",
    "scheduler": "dispersion_only_quantile_proxy"
  },
  {
    "commitment": 4,
    "commitment_map": {
      "long": 50,
      "medium": 4,
      "short": 1
    },
    "config_id": "eac_q33_aggressive_1_4_50",
    "policy": "eac_full",
    "proxy_or_reproduction_label": "ours",
    "quantile_margin": 0.33,
    "role": "ours",
    "scheduler": "selected_validation_config"
  },
  {
    "commitment": 4,
    "commitment_map": {
      "long": 50,
      "medium": 4,
      "short": 1
    },
    "policy": "eac_no_calibration_no_hysteresis_ablation",
    "proxy_or_reproduction_label": "key ablation",
    "raw_thresholds": {
      "high": 0.6666666666666666,
      "low": 0.3333333333333333
    },
    "role": "key_ablation",
    "scheduler": "raw_risk_fixed_threshold_ablation"
  },
  {
    "commitment": 1,
    "policy": "fixed_short_replan_baseline",
    "proxy_or_reproduction_label": "strong simple fixed short-replan baseline",
    "role": "simple_killer",
    "scheduler": "fixed_commitment"
  }
]
```

Preflight records:

```json
[
  {
    "action_values_modified": false,
    "commitment": 50,
    "policy": "frozen_smolvla_fixed_queue",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "e53c3b84489bdc60ff91bb9c950e91cde95b64606ac87b8639721ca535007285",
    "prefix_shape": [
      50,
      7
    ],
    "proxy_or_reproduction_label": "unmodified frozen SmolVLA action values with fixed full queue commitment",
    "role": "base",
    "scheduler": "fixed_commitment"
  },
  {
    "action_values_modified": false,
    "commitment": 8,
    "policy": "aac_entropy_proxy",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "05e862c61e56114046fb94aadcb3294772b48a3d5b6809ba480f6386a9a407d3",
    "prefix_shape": [
      8,
      7
    ],
    "proxy_or_reproduction_label": "faithful transparent local proxy, not an official AAC reproduction",
    "role": "closest_prior_proxy",
    "scheduler": "dispersion_only_quantile_proxy"
  },
  {
    "action_values_modified": false,
    "commitment": 4,
    "policy": "eac_full",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "d8260f2b9de9d43b11f681c6d8cbe87ca01b3b4ddf450f7db6d3106fc4b5d85a",
    "prefix_shape": [
      4,
      7
    ],
    "proxy_or_reproduction_label": "ours",
    "role": "ours",
    "scheduler": "selected_validation_config"
  },
  {
    "action_values_modified": false,
    "commitment": 4,
    "policy": "eac_no_calibration_no_hysteresis_ablation",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "d8260f2b9de9d43b11f681c6d8cbe87ca01b3b4ddf450f7db6d3106fc4b5d85a",
    "prefix_shape": [
      4,
      7
    ],
    "proxy_or_reproduction_label": "key ablation",
    "role": "key_ablation",
    "scheduler": "raw_risk_fixed_threshold_ablation"
  },
  {
    "action_values_modified": false,
    "commitment": 1,
    "policy": "fixed_short_replan_baseline",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "cbde4b485fa6d2f5b6e19f6eb03b0ca7f325905f7883439ae63bcd06900458be",
    "prefix_shape": [
      1,
      7
    ],
    "proxy_or_reproduction_label": "strong simple fixed short-replan baseline",
    "role": "simple_killer",
    "scheduler": "fixed_commitment"
  }
]
```

Errors:
- none

Next step: Implement the minimal EAC Stage A runner and launch only after runner validation.
