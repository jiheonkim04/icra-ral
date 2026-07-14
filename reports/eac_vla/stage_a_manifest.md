# EAC-VLA Stage A Manifest And Preflight

Date: `2026-07-15`

Final decision: `EAC_STAGE_A_PLAN_FROZEN_PREFLIGHT_PENDING`

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
[]
```

Errors:
- none

Next step: Run the EAC Stage A policy preflight before any rollout.
