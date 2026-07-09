# Next Actions

Date: 2026-07-09 KST

Current decision: `GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING`

## Immediate Next Action

Create a task-conditional adapter-routing design plan only.

Required boundary:

- official assets only: `C:\assets\checkpoints\smolvla_libero` and `C:\assets\datasets\lerobot_libero`;
- mandatory anchors: frozen/base official SmolVLA, standard rank-4 LoRA official baseline, and mean-action prior;
- include a MoIRA-style routing comparison because recent routing work is close;
- retain the failure-mining signal: broader rank-4 LoRA worsened aggregate action L2 and eval loss, but helped some task/frame groups and hurt others;
- predeclare primary metrics, baselines, ablations, split/sample policy, tuning budget, and kill/pivot criteria before any method run;
- do not implement the method in the next planning step;
- no archived custom `LIBERO_7D` adapter route;
- no OpenVLA-OFT;
- no full benchmark or simulator rollout until WSL/Linux/MuJoCo readiness is handled separately.

Official simulator eval remains a separate WSL/Linux/MuJoCo readiness milestone. Do not substitute the archived custom replay bridge as official eval.
