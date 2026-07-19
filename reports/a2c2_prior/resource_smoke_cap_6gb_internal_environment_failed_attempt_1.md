# A2C2 Resource-Only Actual-Path Smoke

Date: `2026-07-19 KST`

Fidelity label: `MECHANISM_FAITHFUL_A2C2_LOCAL_PORT`

Final decision: `A2C2_RESOURCE_SMOKE_INTERNAL_FAIL`

```json
{
  "after_resources": {
    "pid": 287,
    "rss_mb": 449.375,
    "system_ram_limit_fraction": 0.82,
    "system_ram_total_gib": 5.785,
    "system_ram_used_fraction": 0.109,
    "system_ram_used_gib": 0.631
  },
  "base_policy_load_audit": null,
  "before_resources": {
    "pid": 287,
    "rss_mb": 448.641,
    "system_ram_limit_fraction": 0.82,
    "system_ram_total_gib": 5.785,
    "system_ram_used_fraction": 0.109,
    "system_ram_used_gib": 0.63
  },
  "condition": "BASE_STANDARD_E10_D0",
  "date": "2026-07-19 KST",
  "elapsed_seconds": 0.135,
  "environment_constructed": false,
  "episode_completed": false,
  "exception": {
    "message": "No module named 'peft'",
    "traceback": [
      "Traceback (most recent call last):",
      "  File \"/mnt/c/Users/jiheo/tca_map/scripts/run_a2c2_resource_smoke.py\", line 172, in main",
      "    loaded = frozen._load_policy_and_processors(args, frozen.PolicySpec(\"frozen_base\"))",
      "  File \"/mnt/c/Users/jiheo/tca_map/tca_map/smolvla/official_wsl_libero_rollout.py\", line 242, in _load_policy_and_processors",
      "    from peft import PeftConfig, PeftModel",
      "ModuleNotFoundError: No module named 'peft'"
    ],
    "type": "ModuleNotFoundError"
  },
  "execution_horizon": 10,
  "execution_type": "VLA_INFERENCE",
  "fidelity_label": "MECHANISM_FAITHFUL_A2C2_LOCAL_PORT",
  "final_decision": "A2C2_RESOURCE_SMOKE_INTERNAL_FAIL",
  "frozen_runner_path": "scripts/run_a2c2_problem_verification.py",
  "global_task_index": 34,
  "inference_delay": 0,
  "internal_pass": false,
  "kernel_oom_after": {
    "available": true,
    "matching_lines": [],
    "returncode": 0
  },
  "kernel_oom_before": {
    "available": true,
    "matching_lines": [],
    "returncode": 0
  },
  "max_steps": 220,
  "meminfo_after": {
    "mem_total_bytes": 6211846144,
    "swap_free_bytes": 0,
    "swap_total_bytes": 0
  },
  "meminfo_before": {
    "mem_total_bytes": 6211846144,
    "swap_free_bytes": 0,
    "swap_total_bytes": 0
  },
  "new_kernel_oom_lines": [],
  "no_cpu_or_disk_model_offload": false,
  "official_commit": "54dd088302a0ef3f50c4add3ec927ab94d76a406",
  "official_init_state_id": 0,
  "ours_designed_or_executed": false,
  "peak_resources": {
    "exceptions": [],
    "interval_seconds": 0.25,
    "peak_rss_mb": 448.734,
    "peak_vram_allocated_mib": 0.0,
    "peak_vram_reserved_mib": 0.0,
    "peak_wsl_used_fraction": 0.109,
    "peak_wsl_used_gib": 0.63,
    "samples": 1
  },
  "prior_retrained": false,
  "purpose": "RESOURCE_ONLY_ACTUAL_PATH_SMOKE",
  "resource_cap_gib_requested": 6,
  "reward_persisted": false,
  "schema_version": 1,
  "scientific_episode_row_persisted": false,
  "scientific_protocol_changed": false,
  "task_id": 0,
  "task_success_counted": false,
  "task_success_persisted": false,
  "teardown": {
    "environment_closed": false,
    "exceptions": [],
    "success": false
  },
  "trace_evidence_without_outcome": null,
  "wslconfig_sha256": "62395E1525E50770C6AA8DB1F6FD5E78B301BC786764D8D29B4DDEAB0AF80259"
}
```
