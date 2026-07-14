# RAC-VLA Validation Search

Date: `2026-07-14`

Proposal hash: `71ABA93E37FC725C1A2E5EAE6E1461BC77AACDAFF9B0711C37F17D5C0AB0902F`

Final decision: `VALIDATION_SEARCH_SELECT_CONFIG`

Search budget: `6 configs: H in {2, 4}, alpha in {0.05, 0.10, 0.20}`

## Selected Config

```json
{
  "action_only_validation_accuracy": 0.3588530465949821,
  "clean_action_delta_p95": 0.0,
  "config_id": "rac_h4_a0.05",
  "final_decision": "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH",
  "full_validation_accuracy": 0.6032497013142174,
  "full_vs_best_baseline_accuracy_margin": 0.24439665471923533,
  "full_vs_best_baseline_shifted_accuracy_margin": 0.20161290322580638,
  "gate_positive_fraction": 0.17204301075268819,
  "hard_stop_reasons": [],
  "history_horizon": 4,
  "no_consequence_validation_accuracy": 0.3511589008363202,
  "residual_alpha": 0.05,
  "score_terms": {
    "action_validity": 1.0,
    "clean_retention": 1.0,
    "full_vs_best_baseline_margin": 0.24439665471923533,
    "mechanism_activation": 0.8198924731182795,
    "shifted_proxy_gain": 0.20161290322580638,
    "total": 0.5089259259259259
  },
  "shifted_action_delta_p95": 0.005000000000000001,
  "validation_action_validity": 1.0
}
```

## Tried Configs

| config | decision | full acc | margin | shifted margin | gate | clean p95 | shifted p95 | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rac_h2_a0.05` | `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH` | 0.5857448325017819 | 0.21126158232359227 | 0.1660133048229983 | 0.16830601092896175 | 0.0 | 0.005000000000000001 | 0.48983012592064623 |
| `rac_h2_a0.10` | `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH` | 0.5857448325017819 | 0.21126158232359227 | 0.1660133048229983 | 0.16830601092896175 | 0.0 | 0.010000000000000002 | 0.48983012592064623 |
| `rac_h2_a0.20` | `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH` | 0.5857448325017819 | 0.21126158232359227 | 0.1660133048229983 | 0.16830601092896175 | 0.0 | 0.020000000000000004 | 0.48983012592064623 |
| `rac_h4_a0.05` | `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH` | 0.6032497013142174 | 0.24439665471923533 | 0.20161290322580638 | 0.17204301075268819 | 0.0 | 0.005000000000000001 | 0.5089259259259259 |
| `rac_h4_a0.10` | `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH` | 0.6032497013142174 | 0.24439665471923533 | 0.20161290322580638 | 0.17204301075268819 | 0.0 | 0.010000000000000002 | 0.5089259259259259 |
| `rac_h4_a0.20` | `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH` | 0.6032497013142174 | 0.24439665471923533 | 0.20161290322580638 | 0.17204301075268819 | 0.0 | 0.020000000000000004 | 0.5089259259259259 |

Next step: Freeze selected config and implement Stage A runner.
