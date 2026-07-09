# Official SmolVLA-LIBERO Failure Mining Result

- final decision: `GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING`
- status: `completed`
- experiments happened: `True`
- training happened: `True`
- downloads/OpenVLA-OFT/rollout: `False` / `False` / `False`
- official model/dataset used: `True`

## Metric Reconciliation

- primary metric recommendation: `postprocessed held-out action L2 with translation/rotation/gripper breakdown and mean-action prior comparison`
- secondary metric recommendation: `normalized chunk flow eval loss as a stability/retention warning metric`
- warning: `If action L2 improves while eval loss worsens, treat it as a control-stability/retention warning, not as a paper-grade success.`

## Aggregate Metrics

### frozen_base

- sample count: `200`
- eval loss mean: `0.01197837`
- action L2 mean: `0.10651496`
- translation L2 mean: `0.075728161`
- rotation L2 mean: `0.014086517`
- gripper abs mean/sign accuracy: `0.039751591` / `0.985`
- range violation rate: `0.54`

### rank4_lora

- sample count: `200`
- eval loss mean: `0.01214829`
- action L2 mean: `0.118024259`
- translation L2 mean: `0.076366197`
- rotation L2 mean: `0.015253659`
- gripper abs mean/sign accuracy: `0.050169439` / `0.98`
- range violation rate: `0.545`

### mean_action_prior

- sample count: `200`
- eval loss mean: `None`
- action L2 mean: `1.144859722`
- translation L2 mean: `0.515473497`
- rotation L2 mean: `0.082277143`
- gripper abs mean/sign accuracy: `0.988086157` / `0.62`
- range violation rate: `0.0`

## Comparison

- LoRA help/hurt count: `98` / `102`
- task mean help/hurt count: `2` / `3`
- LoRA eval loss worse count: `101`
- mean prior better than LoRA count: `4`

## Subgroup Failures

Per-task action L2:

| task | frozen/base | rank-4 LoRA | mean-action | LoRA gain | eval-loss delta | help fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.140706462 | 0.142809121 | 1.136228448 | -0.002102660 | 0.002842181 | 0.575 |
| 2 | 0.071951801 | 0.125088170 | 1.129058793 | -0.053136368 | -0.003744569 | 0.375 |
| 4 | 0.068254358 | 0.072736744 | 1.159404522 | -0.004482386 | 0.006501973 | 0.450 |
| 5 | 0.127170722 | 0.126285626 | 1.173568854 | 0.000885096 | -0.003822969 | 0.500 |
| 8 | 0.124491459 | 0.123201633 | 1.126037994 | 0.001289826 | -0.000927014 | 0.550 |

Per-phase action L2:

| phase | frozen/base | rank-4 LoRA | LoRA gain | eval-loss delta | help fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| early | 0.119669650 | 0.119615740 | 0.000053910 | -0.000395997 | 0.500 |
| mid | 0.109251183 | 0.144067369 | -0.034816186 | 0.002421720 | 0.516666667 |
| late | 0.091014937 | 0.094110112 | -0.003095174 | -0.001194275 | 0.457142857 |

Per-action-dimension absolute error:

- frozen/base: `[0.033745615, 0.037118165, 0.043534335, 0.005062995, 0.00837664, 0.0068652, 0.03975157]`
- rank-4 LoRA: `[0.035980945, 0.03589962, 0.041945745, 0.00482541, 0.009851505, 0.007215055, 0.050169465]`

Training/eval gap:

- rank-4 LoRA train-episode action L2 mean: `0.062521894`
- rank-4 LoRA held-out action L2 mean: `0.118024259`
- rank-4 LoRA train-episode eval loss mean: `0.004519369`
- rank-4 LoRA held-out eval loss mean: `0.01214829`

## Gap Analysis

- strongest method-worthy gap: `task_adapter_interference`
- estimated kill risk: `high`
- recommended method direction: `Task-Conditional Adapter Routing`

Important caveat: frozen/base beats rank-4 LoRA on aggregate action L2 and eval loss in this broader held-out subset. The method gap is not "LoRA is strong"; it is that low-data LoRA creates mixed task/frame-level interference that a future method would need to avoid while beating frozen/base.

Exact next prompt: Design a task-conditional adapter-routing plan only after explicitly comparing against MoIRA-style routing and standard LoRA anchors.
