# Official SmolVLA Routing Next Decision

Date: 2026-07-09 KST

Final decision: `GO_DESIGN_FRAME_CONDITIONAL_ROUTING`

Reason: Frame oracle clears the routing headroom gate, while task/instruction oracle headroom is tiny. A viable design therefore needs frame/state/action-disagreement signals and an explicit frozen/base fallback, not text-only task routing.

Exact next prompt: Design a Frame-Conditional Adapter Retention method plan for official SmolVLA-LIBERO. Do not implement it; predeclare frozen/base, rank-4 LoRA, mean-action prior, frame oracle, task oracle, and MoIRA-style instruction router baselines.