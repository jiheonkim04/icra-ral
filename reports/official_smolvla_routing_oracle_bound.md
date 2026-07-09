# Official SmolVLA Routing Oracle Bound

Date: 2026-07-09 KST

## Boundary

- official SmolVLA-LIBERO evidence only
- no method implementation
- standard rank-4 LoRA retrained only because saved adapter/per-frame rows were unavailable: `True`
- held-out frames: `200`
- task groups: `5`

## Aggregate Metrics

| variant | action L2 | eval loss | translation L2 | rotation L2 | gripper abs | gripper sign | selected base/LoRA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| frozen_base | 0.10651496 | 0.01197837 | 0.075728161 | 0.014086517 | 0.039751591 | 0.985 | None |
| rank4_lora | 0.118024259 | 0.01214829 | 0.076366197 | 0.015253659 | 0.050169439 | 0.98 | None |
| mean_action_prior | 1.144859722 | None | 0.515473497 | 0.082277143 | 0.988086157 | 0.62 | None |
| frame_oracle | 0.084582188 | 0.012969294 | 0.062440094 | 0.013420008 | 0.029592019 | 0.99 | {'rank4_lora': 98, 'frozen_base': 102} |
| task_oracle | 0.106079976 | 0.011028373 | 0.074262319 | 0.014470935 | 0.039627159 | 0.985 | {'frozen_base': 120, 'rank4_lora': 80} |
| instruction_task_id_oracle | 0.106079976 | 0.011028373 | 0.074262319 | 0.014470935 | 0.039627159 | 0.985 | {'frozen_base': 120, 'rank4_lora': 80} |
| eval_loss_oracle | 0.117064321 | 0.006147801 | 0.076791049 | 0.014781846 | 0.049766632 | 0.98 | {'frozen_base': 101, 'rank4_lora': 99} |
| action_dim_oracle | 0.075210683 | 0.006147801 | 0.053520119 | 0.00945968 | 0.027040264 | 0.99 | {'per_action_dimension_oracle': 200} |

## Headroom

- frame oracle improvement over frozen/base: `{'absolute': 0.021932772, 'relative': 0.205912597}`
- task oracle improvement over frozen/base: `{'absolute': 0.000434984, 'relative': 0.004083783}`
- eval-loss oracle improvement over frozen/base: `{'absolute': -0.010549361, 'relative': -0.099041121}`
- hard-gate threshold: at least `0.005` absolute and `5%` relative action-L2 improvement over frozen/base

## Per-Task Oracle Gains

| task | base action L2 | task-oracle action L2 | abs gain | rel gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.140706462 | 0.140706462 | 0.0 | 0.0 |
| 2 | 0.071951801 | 0.071951801 | 0.0 | 0.0 |
| 4 | 0.068254358 | 0.068254358 | 0.0 | 0.0 |
| 5 | 0.127170722 | 0.126285626 | 0.000885096 | 0.006959904 |
| 8 | 0.124491459 | 0.123201633 | 0.001289826 | 0.010360762 |

Routing upper-bound verdict: `frame_headroom_only`