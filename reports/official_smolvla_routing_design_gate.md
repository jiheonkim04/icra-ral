# Official SmolVLA Routing Design Gate

Date: 2026-07-09 KST

Final decision: `GO_DESIGN_FRAME_CONDITIONAL_ROUTING`

## Boundary

- experiments happened: `True`
- training happened: `True`
- loss computed: `True`
- GPU/download/OpenVLA-OFT: `True` / `False` / `False`
- official dataset/model used: `True`
- old custom route used: `False`
- method implemented: `False`

## Key Metrics

- frozen/base action L2: `0.10651496`
- rank-4 LoRA action L2: `0.118024259`
- mean-action prior action L2: `1.144859722`
- frame oracle action L2: `0.084582188`
- task oracle action L2: `0.106079976`
- action-dim oracle action L2: `0.075210683`

## Conclusion

Frame oracle clears the routing headroom gate, while task/instruction oracle headroom is tiny. A viable design therefore needs frame/state/action-disagreement signals and an explicit frozen/base fallback, not text-only task routing.

Exact next prompt: Design a Frame-Conditional Adapter Retention method plan for official SmolVLA-LIBERO. Do not implement it; predeclare frozen/base, rank-4 LoRA, mean-action prior, frame oracle, task oracle, and MoIRA-style instruction router baselines.