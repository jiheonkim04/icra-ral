# A2C2 Prior Closed Loop

Date: `2026-07-19 KST`

Fidelity label: `MECHANISM_FAITHFUL_A2C2_LOCAL_PORT`

Final decision: `A2C2_PRIOR_CLOSED_LOOP_ACCEPTED`

```json
{
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
      "allocated_bytes": 1047712768,
      "allocated_mb": 999.177,
      "max_allocated_bytes": 1080223232,
      "max_allocated_mb": 1030.181
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
    "load_seconds": 15.27,
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
    "cuda_pid": 289,
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "pid": 289,
    "rss_mb": 843.438,
    "system_ram_limit_fraction": 0.82,
    "system_ram_total_gib": 11.68,
    "system_ram_used_fraction": 0.071,
    "system_ram_used_gib": 0.829,
    "vram_allocated_mib": 0.0,
    "vram_reserved_fraction": 0.0,
    "vram_reserved_limit_fraction": 0.88,
    "vram_reserved_mib": 0.0,
    "vram_total_mib": 16302.562
  },
  "completed_episode_rows": 15,
  "conditions": {
    "PRIOR_DELAYED_E40_D10": {
      "execution_horizon": 40,
      "inference_delay": 10
    }
  },
  "date": "2026-07-19 KST",
  "elapsed_seconds": 319.016,
  "episodes": [
    {
      "action_finite": true,
      "base_model_forward_count": 6,
      "condition": "PRIOR_DELAYED_E40_D10",
      "elapsed_seconds": 17.61,
      "episode_length": 220,
      "exception": null,
      "execution_horizon": 40,
      "global_task_index": 34,
      "inference_delay": 10,
      "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
      "max_reward": 0.0,
      "max_steps": 220,
      "official_init_state_id": 0,
      "peak_vram": {
        "allocated_bytes": 1049671680,
        "allocated_mb": 1001.045,
        "max_allocated_bytes": 1082182144,
        "max_allocated_mb": 1032.049
      },
      "prior_max_mean_abs_correction": 0.031784013,
      "prior_mean_abs_correction": 0.014761891,
      "prior_module_forward_count": 220,
      "rss_mb": 4337.637,
      "success": false,
      "suite": "libero_spatial",
      "sum_reward": 0.0,
      "task_id": 0,
      "uses_expert_action_at_live_inference": false,
      "uses_prior": true
    },
    {
      "action_finite": true,
      "base_model_forward_count": 2,
      "condition": "PRIOR_DELAYED_E40_D10",
      "elapsed_seconds": 9.383,
      "episode_length": 79,
      "exception": null,
      "execution_horizon": 40,
      "global_task_index": 34,
      "inference_delay": 10,
      "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
      "max_reward": 1.0,
      "max_steps": 220,
      "official_init_state_id": 1,
      "peak_vram": {
        "allocated_bytes": 1049671680,
        "allocated_mb": 1001.045,
        "max_allocated_bytes": 1082182144,
        "max_allocated_mb": 1032.049
      },
      "prior_max_mean_abs_correction": 0.022878351,
      "prior_mean_abs_correction": 0.011744561,
      "prior_module_forward_count": 79,
      "rss_mb": 4429.082,
      "success": true,
      "suite": "libero_spatial",
      "sum_reward": 1.0,
      "task_id": 0,
      "uses_expert_action_at_live_inference": false,
      "uses_prior": true
    },
    {
      "action_finite": true,
      "base_model_forward_count": 6,
      "condition": "PRIOR_DELAYED_E40_D10",
      "elapsed_seconds": 17.727,
      "episode_length": 220,
      "exception": null,
      "execution_horizon": 40,
      "global_task_index": 34,
      "inference_delay": 10,
      "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
      "max_reward": 0.0,
      "max_steps": 220,
      "official_init_state_id": 2,
      "peak_vram": {
        "allocated_bytes": 1049671680,
        "allocated_mb": 1001.045,
        "max_allocated_bytes": 1082182144,
        "max_allocated_mb": 1032.049
      },
      "prior_max_mean_abs_correction": 0.029122028,
      "prior_mean_abs_correction": 0.017432278,
      "prior_module_forward_count": 220,
      "rss_mb": 4446.434,
      "success": false,
      "suite": "libero_spatial",
      "sum_reward": 0.0,
      "task_id": 0,
      "uses_expert_action_at_live_inference": false,
      "uses_prior": true
    },
    {
      "action_finite": true,
      "base_model_forward_count": 6,
      "condition": "PRIOR_DELAYED_E40_D10",
      "elapsed_seconds": 17.672,
      "episode_length": 220,
      "exception": null,
      "execution_horizon": 40,
      "global_task_index": 34,
      "inference_delay": 10,
      "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
      "max_reward": 0.0,
      "max_steps": 220,
      "official_init_state_id": 3,
      "peak_vram": {
        "allocated_bytes": 1049671680,
        "allocated_mb": 1001.045,
        "max_allocated_bytes": 1082182144,
        "max_allocated_mb": 1032.049
      },
      "prior_max_mean_abs_correction": 0.028227862,
      "prior_mean_abs_correction": 0.01191475,
      "prior_module_forward_count": 220,
      "rss_mb": 4468.977,
      "success": false,
      "suite": "libero_spatial",
      "sum_reward": 0.0,
      "task_id": 0,
      "uses_expert_action_at_live_inference": false,
      "uses_prior": true
    },
    {
      "action_finite": true,
      "base_model_forward_count": 6,
      "condition": "PRIOR_DELAYED_E40_D10",
      "elapsed_seconds": 17.442,
      "episode_length": 220,
      "exception": null,
      "execution_horizon": 40,
      "global_task_index": 34,
      "inference_delay": 10,
      "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
      "max_reward": 0.0,
      "max_steps": 220,
      "official_init_state_id": 4,
      "peak_vram": {
        "allocated_bytes": 1049671680,
        "allocated_mb": 1001.045,
        "max_allocated_bytes": 1082182144,
        "max_allocated_mb": 1032.049
      },
      "prior_max_mean_abs_correction": 0.035403702,
      "prior_mean_abs_correction": 0.014630128,
      "prior_module_forward_count": 220,
      "rss_mb": 4473.605,
      "success": false,
      "suite": "libero_spatial",
      "sum_reward": 0.0,
      "task_id": 0,
      "uses_expert_action_at_live_inference": false,
      "uses_prior": true
    },
    {
      "action_finite": true,
      "base_model_forward_count": 4,
      "condition": "PRIOR_DELAYED_E40_D10",
      "elapsed_seconds": 14.048,
      "episode_length": 125,
      "exception": null,
      "execution_horizon": 40,
      "global_task_index": 31,
      "inference_delay": 10,
      "instruction": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
      "max_reward": 1.0,
      "max_steps": 220,
      "official_init_state_id": 0,
      "peak_vram": {
        "allocated_bytes": 1049696256,
        "allocated_mb": 1001.068,
        "max_allocated_bytes": 1082182144,
        "max_allocated_mb": 1032.049
      },
      "prior_max_mean_abs_correction": 0.031483836,
      "prior_mean_abs_correction": 0.014101132,
      "prior_module_forward_count": 125,
      "rss_mb": 4461.887,
      "success": true,
      "suite": "libero_spatial",
      "sum_reward": 1.0,
      "task_id": 4,
      "uses_expert_action_at_live_inference": false,
      "uses_prior": true
    },
    {
      "action_finite": true,
      "base_model_forward_count": 6,
      "condition": "PRIOR_DELAYED_E40_D10",
      "elapsed_seconds": 18.858,
      "episode_length": 220,
      "exception": null,
      "execution_horizon": 40,
      "global_task_index": 31,
      "inference_delay": 10,
      "instruction": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
      "max_reward": 0.0,
      "max_steps": 220,
      "official_init_state_id": 1,
      "peak_vram": {
        "allocated_bytes": 1049671680,
        "allocated_mb": 1001.045,
        "max_allocated_bytes": 1082182144,
        "max_allocated_mb": 1032.049
      },
      "prior_max_mean_abs_correction": 0.030838618,
      "prior_mean_abs_correction": 0.013014652,
      "prior_module_forward_count": 220,
      "rss_mb": 4447.242,
      "success": false,
      "suite": "libero_spatial",
      "sum_reward": 0.0,
      "task_id": 4,
      "uses_expert_action_at_live_inference": false,
      "uses_prior": true
    },
    {
      "action_finite": true,
      "base_model_forward_count": 6,
      "condition": "PRIOR_DELAYED_E40_D10",
      "elapsed_seconds": 19.064,
      "episode_length": 220,
      "exception": null,
      "execution_horizon": 40,
      "global_task_index": 31,
      "inference_delay": 10,
      "instruction": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
      "max_reward": 0.0,
      "max_steps": 220,
      "official_init_state_id": 2,
      "peak_vram": {
        "allocated_bytes": 1049671680,
        "allocated_mb": 1001.045,
        "max_allocated_bytes": 1082182144,
        "max_allocated_mb": 1032.049
      },
      "prior_max_mean_abs_correction": 0.033920031,
      "prior_mean_abs_correction": 0.010511388,
      "prior_module_forward_count": 220,
      "rss_mb": 4464.527,
      "success": false,
      "suite": "libero_spatial",
      "sum_reward": 0.0,
      "task_id": 4,
      "uses_expert_action_at_live_inference": false,
      "uses_prior": true
    },
    {
      "action_finite": true,
      "base_model_forward_count": 6,
      "condition": "PRIOR_DELAYED_E40_D10",
      "elapsed_seconds": 18.4,
      "episode_length": 220,
      "exception": null,
      "execution_horizon": 40,
      "global_task_index": 31,
      "inference_delay": 10,
      "instruction": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
      "max_reward": 0.0,
      "max_steps": 220,
      "official_init_state_id": 3,
      "peak_vram": {
        "allocated_bytes": 1049671680,
        "allocated_mb": 1001.045,
        "max_allocated_bytes": 1082182144,
        "max_allocated_mb": 1032.049
      },
      "prior_max_mean_abs_correction": 0.032949466,
      "prior_mean_abs_correction": 0.013648326,
      "prior_module_forward_count": 220,
      "rss_mb": 4497.777,
      "success": false,
      "suite": "libero_spatial",
      "sum_reward": 0.0,
      "task_id": 4,
      "uses_expert_action_at_live_inference": false,
      "uses_prior": true
    },
    {
      "action_finite": true,
      "base_model_forward_count": 6,
      "condition": "PRIOR_DELAYED_E40_D10",
      "elapsed_seconds": 18.079,
      "episode_length": 220,
      "exception": null,
      "execution_horizon": 40,
      "global_task_index": 31,
      "inference_delay": 10,
      "instruction": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
      "max_reward": 0.0,
      "max_steps": 220,
      "official_init_state_id": 4,
      "peak_vram": {
        "allocated_bytes": 1049671680,
        "allocated_mb": 1001.045,
        "max_allocated_bytes": 1082182144,
        "max_allocated_mb": 1032.049
      },
      "prior_max_mean_abs_correction": 0.027873294,
      "prior_mean_abs_correction": 0.011998608,
      "prior_module_forward_count": 220,
      "rss_mb": 4443.945,
      "success": false,
      "suite": "libero_spatial",
      "sum_reward": 0.0,
      "task_id": 4,
      "uses_expert_action_at_live_inference": false,
      "uses_prior": true
    },
    {
      "action_finite": true,
      "base_model_forward_count": 3,
      "condition": "PRIOR_DELAYED_E40_D10",
      "elapsed_seconds": 9.916,
      "episode_length": 92,
      "exception": null,
      "execution_horizon": 40,
      "global_task_index": 36,
      "inference_delay": 10,
      "instruction": "pick up the black bowl next to the plate and place it on the plate",
      "max_reward": 1.0,
      "max_steps": 220,
      "official_init_state_id": 0,
      "peak_vram": {
        "allocated_bytes": 1049671680,
        "allocated_mb": 1001.045,
        "max_allocated_bytes": 1082182144,
        "max_allocated_mb": 1032.049
      },
      "prior_max_mean_abs_correction": 0.023575844,
      "prior_mean_abs_correction": 0.010314866,
      "prior_module_forward_count": 92,
      "rss_mb": 4466.352,
      "success": true,
      "suite": "libero_spatial",
      "sum_reward": 1.0,
      "task_id": 8,
      "uses_expert_action_at_live_inference": false,
      "uses_prior": true
    },
    {
      "action_finite": true,
      "base_model_forward_count": 6,
      "condition": "PRIOR_DELAYED_E40_D10",
      "elapsed_seconds": 17.673,
      "episode_length": 220,
      "exception": null,
      "execution_horizon": 40,
      "global_task_index": 36,
      "inference_delay": 10,
      "instruction": "pick up the black bowl next to the plate and place it on the plate",
      "max_reward": 0.0,
      "max_steps": 220,
      "official_init_state_id": 1,
      "peak_vram": {
        "allocated_bytes": 1049671680,
        "allocated_mb": 1001.045,
        "max_allocated_bytes": 1082182144,
        "max_allocated_mb": 1032.049
      },
      "prior_max_mean_abs_correction": 0.01859506,
      "prior_mean_abs_correction": 0.009447925,
      "prior_module_forward_count": 220,
      "rss_mb": 4494.109,
      "success": false,
      "suite": "libero_spatial",
      "sum_reward": 0.0,
      "task_id": 8,
      "uses_expert_action_at_live_inference": false,
      "uses_prior": true
    },
    {
      "action_finite": true,
      "base_model_forward_count": 6,
      "condition": "PRIOR_DELAYED_E40_D10",
      "elapsed_seconds": 17.891,
      "episode_length": 220,
      "exception": null,
      "execution_horizon": 40,
      "global_task_index": 36,
      "inference_delay": 10,
      "instruction": "pick up the black bowl next to the plate and place it on the plate",
      "max_reward": 0.0,
      "max_steps": 220,
      "official_init_state_id": 2,
      "peak_vram": {
        "allocated_bytes": 1049671680,
        "allocated_mb": 1001.045,
        "max_allocated_bytes": 1082182144,
        "max_allocated_mb": 1032.049
      },
      "prior_max_mean_abs_correction": 0.027319843,
      "prior_mean_abs_correction": 0.013727216,
      "prior_module_forward_count": 220,
      "rss_mb": 4471.543,
      "success": false,
      "suite": "libero_spatial",
      "sum_reward": 0.0,
      "task_id": 8,
      "uses_expert_action_at_live_inference": false,
      "uses_prior": true
    },
    {
      "action_finite": true,
      "base_model_forward_count": 6,
      "condition": "PRIOR_DELAYED_E40_D10",
      "elapsed_seconds": 17.933,
      "episode_length": 220,
      "exception": null,
      "execution_horizon": 40,
      "global_task_index": 36,
      "inference_delay": 10,
      "instruction": "pick up the black bowl next to the plate and place it on the plate",
      "max_reward": 0.0,
      "max_steps": 220,
      "official_init_state_id": 3,
      "peak_vram": {
        "allocated_bytes": 1049671680,
        "allocated_mb": 1001.045,
        "max_allocated_bytes": 1082182144,
        "max_allocated_mb": 1032.049
      },
      "prior_max_mean_abs_correction": 0.027525906,
      "prior_mean_abs_correction": 0.012700078,
      "prior_module_forward_count": 220,
      "rss_mb": 4488.93,
      "success": false,
      "suite": "libero_spatial",
      "sum_reward": 0.0,
      "task_id": 8,
      "uses_expert_action_at_live_inference": false,
      "uses_prior": true
    },
    {
      "action_finite": true,
      "base_model_forward_count": 6,
      "condition": "PRIOR_DELAYED_E40_D10",
      "elapsed_seconds": 17.772,
      "episode_length": 220,
      "exception": null,
      "execution_horizon": 40,
      "global_task_index": 36,
      "inference_delay": 10,
      "instruction": "pick up the black bowl next to the plate and place it on the plate",
      "max_reward": 0.0,
      "max_steps": 220,
      "official_init_state_id": 4,
      "peak_vram": {
        "allocated_bytes": 1049671680,
        "allocated_mb": 1001.045,
        "max_allocated_bytes": 1082182144,
        "max_allocated_mb": 1032.049
      },
      "prior_max_mean_abs_correction": 0.027463973,
      "prior_mean_abs_correction": 0.013234902,
      "prior_module_forward_count": 220,
      "rss_mb": 4487.973,
      "success": false,
      "suite": "libero_spatial",
      "sum_reward": 0.0,
      "task_id": 8,
      "uses_expert_action_at_live_inference": false,
      "uses_prior": true
    }
  ],
  "exceptions": [],
  "expert_action_replay_counted_as_success": false,
  "fidelity_label": "MECHANISM_FAITHFUL_A2C2_LOCAL_PORT",
  "final_decision": "A2C2_PRIOR_CLOSED_LOOP_ACCEPTED",
  "job_classification": "VLA_CLOSED_LOOP_ROLLOUT",
  "official_commit": "54dd088302a0ef3f50c4add3ec927ab94d76a406",
  "official_init_state_ids": [
    0,
    1,
    2,
    3,
    4
  ],
  "official_reset_states": true,
  "ours_executed": false,
  "peak_vram": {
    "allocated_bytes": 1046298624,
    "allocated_mb": 997.828,
    "max_allocated_bytes": 1082182144,
    "max_allocated_mb": 1032.049
  },
  "planned_episode_rows": 15,
  "prior_checkpoint": "runs/a2c2_prior/checkpoints/step_040000.pt",
  "prior_checkpoint_step": 40000,
  "rss_mb": 4231.77,
  "schema_version": 1,
  "successful_episode_rows": 3,
  "summary": {
    "PRIOR_DELAYED_E40_D10": {
      "base_model_forward_count": 81,
      "episodes": 15,
      "per_task_successes": {
        "0": 1,
        "4": 1,
        "8": 1
      },
      "prior_module_forward_count": 2936,
      "success_rate": 0.2,
      "successes": 3
    }
  },
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
  ],
  "with_prior": true
}
```
