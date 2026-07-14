# EAC-VLA Stage 0 Audit

Date: `2026-07-15`

Proposal hash: `A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E`

Final decision: `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`

- closed-loop experiment happened: `False`
- training happened: `False`
- validation search happened: `False`
- confirmatory-test tuning happened: `False`
- scoreable validation records: `2000`
- validation unique frames: `400`
- reserved records not used for tuning: `6000`
- queue helper present: `True`
- expected chunk shape recorded: `True`
- full chunk values available in artifact: `False`
- runtime full-chunk check required before validation search: `True`
- first-two dispersion p95: `0.0007983036317792467`
- first-two dispersion nonzero fraction: `1.0`
- commitment counts: `{2: 136, 8: 132, 50: 132}`
- max commitment share: `0.34`
- passthrough max abs error: `5.070000000384489e-07`

Split manifest:

```json
{
  "confirmatory_records_used_for_tuning": false,
  "confirmatory_reserved_split": "test",
  "reserved_sample_count": 6000,
  "reserved_unique_frame_count": 1200,
  "split_counts_records": {
    "test": 6000,
    "val": 2000
  },
  "validation_reserved_frame_overlap": 0,
  "validation_reserved_sample_overlap": 0,
  "validation_sample_count": 2000,
  "validation_split": "val",
  "validation_unique_frame_count": 400
}
```

Queue surface manifest:

```json
{
  "canonical_artifact_chunk_shapes": {
    "[50, 7]": 2000
  },
  "chunk_shape_ok": true,
  "expected_chunk_shape": [
    50,
    7
  ],
  "first_two_preview_available": true,
  "full_chunk_values_available_in_artifact": false,
  "previous_preflight_chunk_shape": [
    50,
    7
  ],
  "queue_helper_present": true,
  "runtime_full_chunk_check_required_before_validation_search": true
}
```

Dispersion manifest summary:

```json
{
  "audited_frames_with_repeated_eval_seeds": 400,
  "commitment_counts": {
    "2": 136,
    "8": 132,
    "50": 132
  },
  "eval_seed_count_distribution": {
    "5": 400
  },
  "first_transition_l2_summary": {
    "count": 400,
    "max": 2.019235135709165,
    "mean": 0.11235769422908842,
    "min": 0.009625775145696307,
    "nonzero_fraction": 1.0,
    "p50": 0.06630401280398343,
    "p95": 0.22224630799806686,
    "std": 0.25644047733858477
  },
  "first_two_dispersion_summary": {
    "count": 400,
    "max": 0.046720446452109814,
    "mean": 0.0008318929853238895,
    "min": 1.8177096435385518e-05,
    "nonzero_fraction": 1.0,
    "p50": 0.00016718024014767058,
    "p95": 0.0007983036317792467,
    "std": 0.005065356524949223
  },
  "max_commitment_share": 0.34,
  "risk_monotonicity": {
    "long_commitment_dispersion_mean": 9.2099138223779e-05,
    "short_commitment_dispersion_mean": 0.002179729077336901,
    "short_gt_long": true
  },
  "risk_summary": {
    "count": 400,
    "max": 1.0,
    "mean": 0.27798851578072303,
    "min": 0.0,
    "nonzero_fraction": 0.9825,
    "p50": 0.20967199371748713,
    "p95": 0.8040613962279417,
    "std": 0.23593148755423402
  },
  "source": "canonical_frozen_base_prediction_artifact_first_two_chunk_previews",
  "unique_validation_frames": 400
}
```

Stage 0 limitations:
- `canonical artifact stores first-two chunk previews and chunk hashes, not all 50 postprocessed actions`
- `runtime full-chunk equality and queue-prefix execution must be checked before validation search artifacts are accepted`

Hard stop reasons:
- none

Next step: Proceed to bounded validation search only after implementing the runtime full-chunk/queue-prefix check.
