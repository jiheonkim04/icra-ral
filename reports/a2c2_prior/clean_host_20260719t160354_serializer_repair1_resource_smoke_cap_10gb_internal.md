# A2C2 Resource-Only Actual-Path Smoke

Date: `2026-07-19 KST`

Fidelity label: `MECHANISM_FAITHFUL_A2C2_LOCAL_PORT`

Final decision: `A2C2_RESOURCE_SMOKE_INTERNAL_PASS`

```json
{
  "after_resources": {
    "cuda_pid": 287,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "pid": 287,
    "rss_mb": 3766.039,
    "system_ram_limit_fraction": 0.82,
    "system_ram_total_gib": 9.711,
    "system_ram_used_fraction": 0.349,
    "system_ram_used_gib": 3.386,
    "vram_allocated_mib": 873.773,
    "vram_reserved_fraction": 0.056187,
    "vram_reserved_limit_fraction": 0.88,
    "vram_reserved_mib": 916.0,
    "vram_total_mib": 16302.562
  },
  "base_policy_load_audit": {
    "action_chunk_device": "cuda:0",
    "action_chunk_dtype": "torch.float32",
    "action_chunk_finite": true,
    "action_chunk_shape": [
      1,
      50,
      7
    ],
    "amp_fp16_or_bf16_active": false,
    "autocast": {
      "cpu": false,
      "cuda": false
    },
    "control_mode": "relative",
    "cuda_memory": {
      "allocated_bytes": 917834240,
      "allocated_mb": 875.315,
      "max_allocated_bytes": 950344704,
      "max_allocated_mb": 906.319
    },
    "empty_cameras": 1,
    "input_tensor_devices": {
      "observation.images.camera1": "cuda:0",
      "observation.images.camera2": "cuda:0",
      "observation.language.attention_mask": "cuda:0",
      "observation.language.tokens": "cuda:0",
      "observation.state": "cuda:0"
    },
    "input_tensor_shapes": {
      "observation.images.camera1": [
        1,
        3,
        256,
        256
      ],
      "observation.images.camera2": [
        1,
        3,
        256,
        256
      ],
      "observation.language.attention_mask": [
        1,
        48
      ],
      "observation.language.tokens": [
        1,
        48
      ],
      "observation.state": [
        1,
        8
      ]
    },
    "load_seconds": 15.938,
    "old_custom_libero_7d_route_used": false,
    "paligemma_import_stub_used": false,
    "parameter": {
      "device": "cuda:0",
      "dtype": "torch.bfloat16",
      "numel": 589824
    },
    "peft": {
      "used": false
    },
    "policy_class": "SmolVLAPolicy",
    "policy_name": "frozen_base",
    "rename_map": {
      "observation.images.image": "observation.images.camera1",
      "observation.images.image2": "observation.images.camera2"
    }
  },
  "before_resources": {
    "cuda_pid": 287,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "pid": 287,
    "rss_mb": 840.875,
    "system_ram_limit_fraction": 0.82,
    "system_ram_total_gib": 9.711,
    "system_ram_used_fraction": 0.083,
    "system_ram_used_gib": 0.81,
    "vram_allocated_mib": 0.0,
    "vram_reserved_fraction": 0.0,
    "vram_reserved_limit_fraction": 0.88,
    "vram_reserved_mib": 0.0,
    "vram_total_mib": 16302.562
  },
  "condition": "BASE_STANDARD_E10_D0",
  "date": "2026-07-19 KST",
  "elapsed_seconds": 33.764,
  "environment_constructed": true,
  "episode_completed": true,
  "exception": null,
  "execution_horizon": 10,
  "execution_type": "VLA_INFERENCE",
  "fidelity_label": "MECHANISM_FAITHFUL_A2C2_LOCAL_PORT",
  "final_decision": "A2C2_RESOURCE_SMOKE_INTERNAL_PASS",
  "frozen_runner_path": "scripts/run_a2c2_problem_verification.py",
  "global_task_index": 34,
  "inference_delay": 0,
  "internal_pass": true,
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
    "mem_total_bytes": 10427125760,
    "swap_free_bytes": 0,
    "swap_total_bytes": 0
  },
  "meminfo_before": {
    "mem_total_bytes": 10427125760,
    "swap_free_bytes": 0,
    "swap_total_bytes": 0
  },
  "new_kernel_oom_lines": [],
  "no_cpu_or_disk_model_offload": true,
  "official_commit": "54dd088302a0ef3f50c4add3ec927ab94d76a406",
  "official_init_state_id": 0,
  "ours_designed_or_executed": false,
  "peak_resources": {
    "exceptions": [],
    "interval_seconds": 0.25,
    "peak_rss_mb": 4081.875,
    "peak_vram_allocated_mib": 906.057,
    "peak_vram_reserved_mib": 932.0,
    "peak_wsl_used_fraction": 0.38,
    "peak_wsl_used_gib": 3.694,
    "samples": 121
  },
  "prior_retrained": false,
  "purpose": "RESOURCE_ONLY_ACTUAL_PATH_SMOKE",
  "resource_cap_gib_requested": 10,
  "reward_persisted": false,
  "schema_version": 1,
  "scientific_episode_row_persisted": false,
  "scientific_protocol_changed": false,
  "task_id": 0,
  "task_success_counted": false,
  "task_success_persisted": false,
  "teardown": {
    "environment_closed": true,
    "exceptions": [],
    "success": true
  },
  "trace_evidence_without_outcome": {
    "action_finite": true,
    "base_model_forward_count": 8,
    "elapsed_seconds": 9.252,
    "episode_length": 76,
    "peak_vram": {
      "allocated_bytes": 918085632,
      "allocated_mb": 875.555,
      "max_allocated_bytes": 950596096,
      "max_allocated_mb": 906.559
    },
    "prior_module_forward_count": 0,
    "reward_persisted": false,
    "rss_mb_at_trace_return": 4029.113,
    "simulator_step_count": 76,
    "task_success_persisted": false
  },
  "wslconfig_sha256": "69139246A90A2994961837BDFA63373D6D75D2F08AEB90C2259E68C12893039F"
}
```
