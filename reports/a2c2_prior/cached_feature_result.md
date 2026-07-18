# A2C2 Cached Feature Probe

Date: `2026-07-19 KST`

Fidelity label: `MECHANISM_FAITHFUL_A2C2_LOCAL_PORT`

Final decision: `A2C2_CACHED_FEATURES_ACCEPTED`

```json
{
  "anchor_stride": 8,
  "base_load_info": {
    "cuda_memory_after_load": {
      "allocated_mb": 864.648,
      "max_allocated_mb": 864.648
    }
  },
  "base_model_forward_count_this_run": 82,
  "cache_path": "runs/a2c2_prior/a2c2_cached_features.h5",
  "cache_sha256": "E6BD933692F18F2AEDC3C7F9AFC87EB66311C175B5D4FAB949F8A8866DB4F3C3",
  "closed_loop_rollout_happened": false,
  "completed_anchor_count_this_run": 82,
  "cumulative_unique_anchor_count": 615,
  "cumulative_base_model_forward_count_across_durable_attempts": 615,
  "date": "2026-07-19 KST",
  "elapsed_seconds": 50.506,
  "exceptions": [],
  "fidelity_label": "MECHANISM_FAITHFUL_A2C2_LOCAL_PORT",
  "final_decision": "A2C2_CACHED_FEATURES_ACCEPTED",
  "frozen_training_episode_ids": [
    1261,
    1274,
    1277,
    1291,
    1262,
    1263,
    1268,
    1276,
    1264,
    1265,
    1266,
    1271,
    1267,
    1269,
    1270,
    1308,
    1272,
    1273,
    1275,
    1282,
    1278,
    1279,
    1303,
    1319,
    1280,
    1285,
    1293,
    1307,
    1281,
    1286,
    1294,
    1297,
    1283,
    1287,
    1289,
    1299,
    1290,
    1295,
    1296,
    1306
  ],
  "job_classification": "CACHED_FEATURE_PROBE",
  "official_commit": "54dd088302a0ef3f50c4add3ec927ab94d76a406",
  "offset_rule": "sorted unique {0, floor(max/3), floor(2max/3), max}, max=min(49, remaining episode steps)",
  "peak_resource_snapshot": {
    "cuda_pid": 282,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "pid": 282,
    "rss_mb": 2484.973,
    "system_ram_limit_fraction": 0.82,
    "system_ram_total_gib": 3.33,
    "system_ram_used_fraction": 0.754,
    "system_ram_used_gib": 2.512,
    "vram_allocated_mib": 1003.703,
    "vram_reserved_fraction": 0.067351,
    "vram_reserved_limit_fraction": 0.88,
    "vram_reserved_mib": 1098.0,
    "vram_total_mib": 16302.562
  },
  "peak_rss_mb": 2484.973,
  "windows_host_resource_observation": {
    "peak_used_fraction_observed_during_accepted_run": 0.8107,
    "used_fraction_after_run": 0.7993,
    "limit_fraction": 0.82,
    "wslconfig_sha256": "B4EFBE7E62D1EAFC6A08E34EE17FB7C8F4A0E2BAEE49635C2123B9DBFA20D0BC",
    "swap_bytes": 0
  },
  "peak_vram": {
    "allocated_bytes": 1052459008,
    "allocated_mb": 1003.703,
    "max_allocated_bytes": 1109607424,
    "max_allocated_mb": 1058.204
  },
  "row_count": 2438,
  "rows_before_resume": 2115,
  "schema_version": 1,
  "task_counts": {
    "30": 233,
    "31": 271,
    "32": 219,
    "33": 283,
    "34": 196,
    "35": 212,
    "36": 250,
    "37": 244,
    "38": 241,
    "39": 289
  },
  "training_happened": false,
  "unique_episode_count": 40,
  "vlm_hidden_dim": 960
}
```
