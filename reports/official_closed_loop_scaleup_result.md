# Official Closed-Loop Scaleup Result

Date: 2026-07-11 KST

- final decision: `OFFLINE_ONLINE_MISMATCH_CONFIRMED`
- planned episodes: `400`
- completed episodes: `400`
- infrastructure failures: `0`
- successful episodes: `282`
- unsuccessful episodes preserved: `118`
- elapsed seconds: `6881.777`
- official route: WSL LeRobot/LIBERO, relative control, official preprocessing/postprocessing/action queue
- static-mix duplicate rollouts: skipped because canonical alpha is `0.0`
- old custom `LIBERO_7D` route used: `false`

## CUDA And Route Audit

| Policy | Parameter device | Input tensor devices | Action chunk device | Autocast fp16/bf16 active | Peak load VRAM MB | Episode peak VRAM MB |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `frozen_base` | `cuda:0` | `cuda:0` | `cuda:0` | `false` | `925.984` | `926.638` |
| `rank4_lora_seed_11` | `cuda:0` | `cuda:0` | `cuda:0` | `false` | `928.331` | `928.365` |
| `rank4_lora_seed_22` | `cuda:0` | `cuda:0` | `cuda:0` | `false` | `928.331` | `928.365` |
| `rank4_lora_seed_33` | `cuda:0` | `cuda:0` | `cuda:0` | `false` | `928.331` | `928.365` |

Preflight saw `NVIDIA GeForce RTX 5080`, `torch 2.10.0+cu128`, and CUDA `12.8`. CPU fallback was not observed.

## Policy Success

| Policy | Successes | Total | Rate | 95% CI |
| --- | ---: | ---: | ---: | --- |
| `frozen_base` | `74` | `100` | `74.0%` | `[0.646288, 0.815954]` |
| `rank4_lora_seed_11` | `74` | `100` | `74.0%` | `[0.646288, 0.815954]` |
| `rank4_lora_seed_22` | `68` | `100` | `68.0%` | `[0.583372, 0.76331]` |
| `rank4_lora_seed_33` | `66` | `100` | `66.0%` | `[0.562775, 0.745386]` |

## Compute And Latency

| Policy | Avg policy latency s | Avg env-step latency s | Avg episode length | Peak VRAM MB |
| --- | ---: | ---: | ---: | ---: |
| `frozen_base` | `0.007874` | `0.064543` | `206.30` | `926.638` |
| `rank4_lora_seed_11` | `0.009584` | `0.062154` | `209.59` | `928.365` |
| `rank4_lora_seed_22` | `0.009453` | `0.062770` | `221.40` | `928.365` |
| `rank4_lora_seed_33` | `0.009773` | `0.063461` | `222.77` | `928.365` |

## Suite Difficulty Across Policies

| Suite | Successes | Total | Rate | Failures |
| --- | ---: | ---: | ---: | ---: |
| `libero_10` | `45` | `100` | `45.0%` | `55` |
| `libero_goal` | `71` | `100` | `71.0%` | `29` |
| `libero_spatial` | `79` | `100` | `79.0%` | `21` |
| `libero_object` | `87` | `100` | `87.0%` | `13` |

## Strongest Repeated Failure Signals

The weakest task by closed-loop success was `libero_10/task_4`, with `5/20` successes and `15/20` failures across all policies and reset seeds. Other low-success task slices were `libero_spatial/task_4` at `6/20`, `libero_10/task_6` at `7/20`, `libero_10/task_8` at `7/20`, and `libero_goal/task_6` at `8/20`.

The strongest repeated all-policy task/reset failures were:

| Task/reset pair | Failed policies |
| --- | --- |
| `libero_10/task_4/seed_20260713` | all four policies |
| `libero_10/task_4/seed_20260715` | all four policies |
| `libero_10/task_6/seed_20260715` | all four policies |
| `libero_10/task_8/seed_20260712` | all four policies |
| `libero_goal/task_6/seed_20260711` | all four policies |
| `libero_goal/task_8/seed_20260712` | all four policies |
| `libero_goal/task_8/seed_20260715` | all four policies |
| `libero_object/task_2/seed_20260712` | all four policies |
| `libero_object/task_8/seed_20260713` | all four policies |
| `libero_spatial/task_4/seed_20260712` | all four policies |
| `libero_spatial/task_4/seed_20260713` | all four policies |
| `libero_spatial/task_4/seed_20260714` | all four policies |
| `libero_spatial/task_8/seed_20260713` | all four policies |

These are closed-loop-important failures, but no confident phase mechanism is claimed because this scaleup did not capture visual evidence.

## Paired Difference Versus Frozen Base

```json
{
  "reset_level": {
    "rank4_lora_seed_11": {
      "loss": 9,
      "tie": 82,
      "win": 9
    },
    "rank4_lora_seed_22": {
      "loss": 14,
      "tie": 78,
      "win": 8
    },
    "rank4_lora_seed_33": {
      "loss": 12,
      "tie": 84,
      "win": 4
    }
  },
  "task_level": {
    "rank4_lora_seed_11": {
      "loss": 4,
      "tie": 13,
      "win": 3
    },
    "rank4_lora_seed_22": {
      "loss": 6,
      "tie": 11,
      "win": 3
    },
    "rank4_lora_seed_33": {
      "loss": 7,
      "tie": 12,
      "win": 1
    }
  }
}
```

## Per-Suite Success

```json
{
  "frozen_base": {
    "libero_10": {
      "success_percent": 44.0,
      "successes": 11,
      "total": 25
    },
    "libero_goal": {
      "success_percent": 80.0,
      "successes": 20,
      "total": 25
    },
    "libero_object": {
      "success_percent": 88.0,
      "successes": 22,
      "total": 25
    },
    "libero_spatial": {
      "success_percent": 84.0,
      "successes": 21,
      "total": 25
    }
  },
  "rank4_lora_seed_11": {
    "libero_10": {
      "success_percent": 56.0,
      "successes": 14,
      "total": 25
    },
    "libero_goal": {
      "success_percent": 76.0,
      "successes": 19,
      "total": 25
    },
    "libero_object": {
      "success_percent": 84.0,
      "successes": 21,
      "total": 25
    },
    "libero_spatial": {
      "success_percent": 80.0,
      "successes": 20,
      "total": 25
    }
  },
  "rank4_lora_seed_22": {
    "libero_10": {
      "success_percent": 48.0,
      "successes": 12,
      "total": 25
    },
    "libero_goal": {
      "success_percent": 60.0,
      "successes": 15,
      "total": 25
    },
    "libero_object": {
      "success_percent": 92.0,
      "successes": 23,
      "total": 25
    },
    "libero_spatial": {
      "success_percent": 72.0,
      "successes": 18,
      "total": 25
    }
  },
  "rank4_lora_seed_33": {
    "libero_10": {
      "success_percent": 32.0,
      "successes": 8,
      "total": 25
    },
    "libero_goal": {
      "success_percent": 68.0,
      "successes": 17,
      "total": 25
    },
    "libero_object": {
      "success_percent": 84.0,
      "successes": 21,
      "total": 25
    },
    "libero_spatial": {
      "success_percent": 80.0,
      "successes": 20,
      "total": 25
    }
  }
}
```

## Per-Task Success

```json
{
  "frozen_base": {
    "libero_10/task_0": {
      "success_percent": 60.0,
      "successes": 3,
      "total": 5
    },
    "libero_10/task_2": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_10/task_4": {
      "success_percent": 20.0,
      "successes": 1,
      "total": 5
    },
    "libero_10/task_6": {
      "success_percent": 40.0,
      "successes": 2,
      "total": 5
    },
    "libero_10/task_8": {
      "success_percent": 20.0,
      "successes": 1,
      "total": 5
    },
    "libero_goal/task_0": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_goal/task_2": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_goal/task_4": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_goal/task_6": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_goal/task_8": {
      "success_percent": 60.0,
      "successes": 3,
      "total": 5
    },
    "libero_object/task_0": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_object/task_2": {
      "success_percent": 60.0,
      "successes": 3,
      "total": 5
    },
    "libero_object/task_4": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_object/task_6": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_object/task_8": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_spatial/task_0": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_spatial/task_2": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_spatial/task_4": {
      "success_percent": 40.0,
      "successes": 2,
      "total": 5
    },
    "libero_spatial/task_6": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_spatial/task_8": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    }
  },
  "rank4_lora_seed_11": {
    "libero_10/task_0": {
      "success_percent": 60.0,
      "successes": 3,
      "total": 5
    },
    "libero_10/task_2": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_10/task_4": {
      "success_percent": 40.0,
      "successes": 2,
      "total": 5
    },
    "libero_10/task_6": {
      "success_percent": 20.0,
      "successes": 1,
      "total": 5
    },
    "libero_10/task_8": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_goal/task_0": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_goal/task_2": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_goal/task_4": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_goal/task_6": {
      "success_percent": 40.0,
      "successes": 2,
      "total": 5
    },
    "libero_goal/task_8": {
      "success_percent": 60.0,
      "successes": 3,
      "total": 5
    },
    "libero_object/task_0": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_object/task_2": {
      "success_percent": 60.0,
      "successes": 3,
      "total": 5
    },
    "libero_object/task_4": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_object/task_6": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_object/task_8": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_spatial/task_0": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_spatial/task_2": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_spatial/task_4": {
      "success_percent": 20.0,
      "successes": 1,
      "total": 5
    },
    "libero_spatial/task_6": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_spatial/task_8": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    }
  },
  "rank4_lora_seed_22": {
    "libero_10/task_0": {
      "success_percent": 40.0,
      "successes": 2,
      "total": 5
    },
    "libero_10/task_2": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_10/task_4": {
      "success_percent": 40.0,
      "successes": 2,
      "total": 5
    },
    "libero_10/task_6": {
      "success_percent": 60.0,
      "successes": 3,
      "total": 5
    },
    "libero_10/task_8": {
      "success_percent": 20.0,
      "successes": 1,
      "total": 5
    },
    "libero_goal/task_0": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_goal/task_2": {
      "success_percent": 60.0,
      "successes": 3,
      "total": 5
    },
    "libero_goal/task_4": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_goal/task_6": {
      "success_percent": 20.0,
      "successes": 1,
      "total": 5
    },
    "libero_goal/task_8": {
      "success_percent": 60.0,
      "successes": 3,
      "total": 5
    },
    "libero_object/task_0": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_object/task_2": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_object/task_4": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_object/task_6": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_object/task_8": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_spatial/task_0": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_spatial/task_2": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_spatial/task_4": {
      "success_percent": 20.0,
      "successes": 1,
      "total": 5
    },
    "libero_spatial/task_6": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_spatial/task_8": {
      "success_percent": 60.0,
      "successes": 3,
      "total": 5
    }
  },
  "rank4_lora_seed_33": {
    "libero_10/task_0": {
      "success_percent": 40.0,
      "successes": 2,
      "total": 5
    },
    "libero_10/task_2": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_10/task_4": {
      "success_percent": 0.0,
      "successes": 0,
      "total": 5
    },
    "libero_10/task_6": {
      "success_percent": 20.0,
      "successes": 1,
      "total": 5
    },
    "libero_10/task_8": {
      "success_percent": 20.0,
      "successes": 1,
      "total": 5
    },
    "libero_goal/task_0": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_goal/task_2": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_goal/task_4": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_goal/task_6": {
      "success_percent": 20.0,
      "successes": 1,
      "total": 5
    },
    "libero_goal/task_8": {
      "success_percent": 60.0,
      "successes": 3,
      "total": 5
    },
    "libero_object/task_0": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_object/task_2": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_object/task_4": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_object/task_6": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_object/task_8": {
      "success_percent": 80.0,
      "successes": 4,
      "total": 5
    },
    "libero_spatial/task_0": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_spatial/task_2": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_spatial/task_4": {
      "success_percent": 40.0,
      "successes": 2,
      "total": 5
    },
    "libero_spatial/task_6": {
      "success_percent": 100.0,
      "successes": 5,
      "total": 5
    },
    "libero_spatial/task_8": {
      "success_percent": 60.0,
      "successes": 3,
      "total": 5
    }
  }
}
```
