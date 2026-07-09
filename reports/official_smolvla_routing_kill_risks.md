# Official SmolVLA Routing Kill Risks

Date: 2026-07-09 KST

Estimated kill risk: `high`

Risks:

- Frozen/base is already stronger than standard rank-4 LoRA on aggregate.
- Task oracle may be too weak even when frame oracle is useful.
- Instruction/task routing alone is killed by MoIRA-style routing.
- A learned gate may fail to approach frame-oracle headroom.
- Offline action L2 may not translate to simulator success without WSL/Linux LIBERO rollout.
- Adapter soup or weighted LoRA merge may match a proposed router.