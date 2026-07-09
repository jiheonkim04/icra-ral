# Next Actions

Date: 2026-07-09 KST

Current decision: `READY_FOR_METHOD_DESIGN_ON_OFFICIAL_SMOLVLA`

## Immediate Next Action

Create the first official-path method-design plan.

Required boundary:

- official assets only: `C:\assets\checkpoints\smolvla_libero` and `C:\assets\datasets\lerobot_libero`;
- mandatory anchors: frozen/base official SmolVLA and standard rank-4 LoRA official baseline;
- retain the mixed baseline signal: LoRA improved mini-holdout action L2 but worsened mini-holdout eval loss;
- predeclare primary metrics, baselines, ablations, split/sample policy, tuning budget, and kill/pivot criteria before any method run;
- no archived custom `LIBERO_7D` adapter route;
- no OpenVLA-OFT;
- no full benchmark or simulator rollout until WSL/Linux/MuJoCo readiness is handled separately.

Official simulator eval remains a separate WSL/Linux/MuJoCo readiness milestone. Do not substitute the archived custom replay bridge as official eval.
