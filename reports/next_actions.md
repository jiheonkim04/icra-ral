# Next Actions

Date: 2026-07-09 KST

Current decision: `GO_DESIGN_FRAME_CONDITIONAL_ROUTING`

## Immediate Next Action

Create the first Frame-Conditional Adapter Retention experiment plan.

Required boundary:

- official assets only: `C:\assets\checkpoints\smolvla_libero` and `C:\assets\datasets\lerobot_libero`;
- mandatory anchors: frozen/base official SmolVLA, standard rank-4 LoRA official baseline, mean-action prior, frame oracle, and task oracle;
- include MoIRA-style instruction routing, task-specific LoRA experts, simple instruction embedding routing, and adapter soup / weighted LoRA merge as required comparisons or kill checks;
- retain the routing-gate signal: frame oracle has meaningful headroom, but task/instruction oracle headroom is tiny;
- predeclare primary metrics, baselines, ablations, split/sample policy, tuning budget, and kill/pivot criteria before any method run;
- do not implement or train the method until that first-experiment plan is complete;
- no archived custom `LIBERO_7D` adapter route;
- no OpenVLA-OFT;
- no full benchmark or simulator rollout until WSL/Linux/MuJoCo readiness is handled separately.

Official simulator eval remains a separate WSL/Linux/MuJoCo readiness milestone. Do not substitute the archived custom replay bridge as official eval.
