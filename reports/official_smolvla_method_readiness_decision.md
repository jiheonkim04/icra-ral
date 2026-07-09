# Official SmolVLA Method-Readiness Decision

Date: 2026-07-09 KST

Branch: `codex/official-smolvla-libero-failure-mining`

Final decision: `GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING`

## Boundary

- experiments happened: yes
- training happened: yes, standard rank-4 LoRA only
- loss computed: yes
- GPU used: yes, `NVIDIA GeForce RTX 5080`
- downloads happened: no
- OpenVLA-OFT happened: no
- simulator rollout/full benchmark happened: no
- official dataset/model used: yes
- old custom `LIBERO_7D` route used: no
- method implemented: no
- paper claim made: no

## Evidence

Official held-out diagnostic subset:

- task groups: `5`
- episodes: `10`
- held-out frames: `200`
- train episode excluded: yes
- mean-action prior included: yes

Aggregate metrics:

| variant | eval loss | action L2 | translation L2 | rotation L2 | gripper abs | gripper sign |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen/base | 0.011978370 | 0.106514960 | 0.075728161 | 0.014086517 | 0.039751591 | 0.985 |
| rank-4 LoRA | 0.012148290 | 0.118024259 | 0.076366197 | 0.015253659 | 0.050169439 | 0.980 |
| mean-action prior | n/a | 1.144859722 | 0.515473497 | 0.082277143 | 0.988086157 | 0.620 |

Training:

- steps: `100`
- trainable params: `185,664`
- loss before/after: `0.008108919` / `0.011093494`
- loss decrease fraction: `-0.368060774`
- peak CUDA allocation: `1104.506 MB`
- total runtime: `217.031 sec`

The broader run contradicts the small previous mini-holdout in aggregate: rank-4 LoRA does not beat frozen/base overall. However, it is not explained by a trivial mean-action prior.

## Structured Failure

Task/frame-level adapter interference:

- LoRA helps `98` held-out frames and hurts `102`;
- task-mean help/hurt count is `2` / `3`;
- task 2 is strongly hurt: action L2 `0.071951801 -> 0.125088170`;
- task 5 and task 8 are slightly helped on average;
- mid-phase action L2 is hurt most: `0.109251183 -> 0.144067369`;
- mean-action prior beats LoRA on only `4` / `200` frames, so the failure is not explained by a trivial prior.

Rejected gaps:

- gripper/contact phase: rejected because gripper sign accuracy remains high (`0.980`) and gripper absolute error is small enough for this diagnostic scale;
- pure control-stability objective: rejected as primary because aggregate action L2 also worsened, not only eval loss;
- action range drift: range violation rate is high for both frozen/base and LoRA (`0.540` / `0.545`), so this is not clearly LoRA-specific.

## Latest-Paper Comparison

- SmolVLA itself is a compact official VLA intended for single-GPU fine-tuning and chunked action generation: https://arxiv.org/abs/2506.01844 and https://huggingface.co/blog/smolvla
- Adaptive/real-time action chunking papers already target chunk consistency/reactivity at inference time, so a generic chunk-consistency method is likely weak: https://arxiv.org/abs/2604.04161
- MoIRA is a close routing baseline because it uses modular routing with low-rank adapters and evaluates on LIBERO-like benchmarks: https://arxiv.org/abs/2507.01843

Consequence: task-adapter routing has a plausible gap but high kill risk. It must beat frozen/base, standard LoRA, mean-action prior, and a MoIRA-style routing comparison. A generic routing story is not novel enough.

## Method Shortlist

1. Task-Conditional Adapter Routing

- precise gap: rank-4 LoRA helps some task/frame groups while hurting others;
- latest-paper comparison: MoIRA is close and makes novelty high-risk;
- required baselines: frozen/base official SmolVLA, standard rank-4 LoRA, mean-action prior, MoIRA-style routing;
- expected improvement axis: reduce negative transfer while beating frozen/base aggregate action L2 and eval loss;
- expected kill risk: high;
- first experiment: planning-only method spec with fixed routing rule and the same failure-mining subset;
- RA-L stability estimate: low-medium until official rollout exists.

2. Control-Stable LoRA Adapter

- precise gap: adapter drift and chunk eval-loss retention;
- latest-paper comparison: adaptive/real-time chunking already attacks a nearby stability problem;
- required baselines: frozen/base, standard LoRA, inference-only chunking/retention ablation if available;
- expected improvement axis: keep action L2 gains without eval-loss degradation;
- expected kill risk: medium-high;
- first experiment: only after task-routing plan decides whether this is an auxiliary loss or a separate method;
- RA-L stability estimate: medium only with simulator rollout.

## Decision

Use `GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING`.

Reason: the strongest official-data failure is mixed task/frame-level LoRA interference, not gripper phase, not mean-action triviality, and not a clean action-L2/eval-loss metric conflict. The direction is allowed only as a design plan, not an implementation, and must explicitly confront MoIRA-style routing plus frozen/base aggregate strength.

Exact next prompt:

Design a task-conditional adapter-routing plan only after explicitly comparing against MoIRA-style routing and standard LoRA anchors. Do not implement it yet.
