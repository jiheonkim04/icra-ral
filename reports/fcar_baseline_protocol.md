# FCAR Baseline Protocol

Date: 2026-07-09 KST

All baselines use official SmolVLA-LIBERO assets only.

| baseline | implementation plan | allowed inputs | realistic/oracle | expected metric | kill role |
| --- | --- | --- | --- | --- | --- |
| Frozen/base SmolVLA | load `smolvla_libero`, run official processor and postprocessor | official observation, instruction | realistic | action L2 `0.106514960` anchor | FCAR must beat it |
| Standard rank-4 LoRA | reproduce bounded 100-step rank-4 LoRA baseline | official train frames only | realistic | current action L2 `0.118024259` | FCAR must beat it |
| Mean-action prior | predict official action stats mean | no observation | trivial realistic prior | action L2 `1.144859722` | rejects trivial explanations |
| Frame oracle | choose frozen/base or LoRA per frame by lower action L2 | ground-truth action | oracle only | action L2 `0.084582188` | upper bound and soft target |
| Task oracle | choose frozen/base or LoRA per task by task-mean action L2 | task labels plus ground-truth eval | oracle only | action L2 `0.106079976` | proves task routing headroom is tiny |
| MoIRA-style instruction/task router | route by task/instruction only, no frame observation | instruction/task text | realistic baseline | expected near task-oracle scale | kills text-only routing novelty |
| Adapter soup / static LoRA merge | static weighted blend or merged LoRA weights, no frame gate | no frame-dependent selection | realistic baseline | unknown; must be measured | kills FCAR if it matches |
| Rank-8 LoRA | same protocol as rank-4 with rank 8 | official train frames only | optional realistic | only if cheap | checks whether capacity fixes issue |
| Action-dim oracle | choose best dimension from base/LoRA per action dimension | ground-truth action | diagnostic oracle only | action L2 `0.075210683` | not valid method evidence |

## MoIRA-Style Router Protocol

The MoIRA-style comparison is an instruction/task router:

- no current-frame state input;
- no base-vs-LoRA disagreement input;
- no action uncertainty input;
- route once per task/instruction or per episode;
- choose among frozen/base and rank-4 LoRA experts.

If FCAR only matches this baseline, the novelty is killed.

## Adapter Soup / Static Merge Protocol

Use a small fixed grid of static mixture weights:

```text
a_static = w * a_lora + (1 - w) * a_base
w in {0.0, 0.25, 0.5, 0.75, 1.0}
```

If feasible later, also test actual LoRA weight merge/soup. The first gate may use prediction-level mixing because saved policy outputs are the artifact needed for FCAR anyway.
