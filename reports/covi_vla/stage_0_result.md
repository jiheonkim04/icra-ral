# COVI-VLA Stage 0 Result

Date: `2026-07-15`

Proposal hash: `338430D2C6CF1D82410C036D79102ED3F38B2367BB35B9AE2811161698A3E621`

Final decision: `IMPLEMENTATION_OR_DATA_FAILURE`

- implementation and data valid: `False`
- diagnostic headroom exists: `False`
- identity and safety passed: `False`
- candidate margin: `-0.05313794852281253`
- strongest comparator: `covi_no_imagined_view_ablation`
- validation records: `400`
- independent validation episodes: `40`
- bootstrap interval: `{'episode_count': 40, 'record_count': 400, 'iterations': 5000, 'low': -0.10168302523568187, 'high': -0.004541124240623629, 'mean': -0.052093271395757236}`
- initial action delta p95: `0.0`
- trained action delta p95: `0.003527037193998693`
- clean retention delta p95: `0.0`
- output valid fraction: `0.2`
- test records decoded: `0`

The Stage 0 occlusion is a synthetic development proxy. It does not establish the final physical-occlusion claim.

False-negative adjudication:

```json
{
  "bootstrap_interval": {
    "episode_count": 40,
    "high": -0.004541124240623629,
    "iterations": 5000,
    "low": -0.10168302523568187,
    "mean": -0.052093271395757236,
    "record_count": 400
  },
  "confidence": "low",
  "evidence_class": "IMPLEMENTATION_OR_DATA_FAILURE",
  "exact_evidence_required_for_permanent_kill": "valid data and implementation, safe acting mechanism, 40 independent episodes, resolved normalization sensitivity, and bootstrap upper bound below 0.02 against both VIM proxy and random-cutout",
  "false_negative_risk": "one validation episode per task and normalization sensitivity can hide a useful small effect",
  "false_positive_risk": "synthetic development occlusion may overstate physical-occlusion transfer",
  "independent_episode_count": 40,
  "narrowest_publishable_claim": "bounded complementary-feature adaptation may improve physical scene-induced occlusion only if later physical validation succeeds",
  "normalization_sensitivity": {
    "margins": {
      "l2_normalized": -0.053137947282571506,
      "raw": -0.04495361852076281,
      "train_z_scored": -0.04580498427370086
    },
    "range": 0.008184328761808697,
    "resolved": true,
    "sign_consistent": true
  },
  "one_fixed_check_allowed": false,
  "practical_effect_threshold": 0.02,
  "record_count": 400,
  "small_point_estimate_alone_used_for_kill": false,
  "strongest_fair_interpretation": "identity-preserving frozen-SmolVLA complementary-feature adapter under a development occlusion proxy"
}
```

Next command: `adjudicate_and_archive_or_repair_under_current_governance`
