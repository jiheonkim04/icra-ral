# Next Actions

Date: 2026-07-09 KST

Current decision: `READY_TO_IMPLEMENT_FCAR_TINY_GATE`

## Immediate Next Action

Implement the FCAR tiny-gate experiment.

Required boundary:

- official assets only: `C:\assets\checkpoints\smolvla_libero` and `C:\assets\datasets\lerobot_libero`;
- follow `reports/fcar_implementation_todo.md` exactly;
- first regenerate and save compact official per-frame base/LoRA predictions because they are not yet saved;
- mandatory anchors: frozen/base official SmolVLA, standard rank-4 LoRA official baseline, mean-action prior, frame oracle, and task oracle;
- include MoIRA-style instruction routing, task-specific LoRA experts, simple instruction embedding routing, and adapter soup / weighted LoRA merge as required comparisons or kill checks;
- retain the routing-gate signal: frame oracle has meaningful headroom, but task/instruction oracle headroom is tiny;
- do not change baselines, metrics, split policy, tuning budget, or kill criteria after seeing FCAR results;
- no archived custom `LIBERO_7D` adapter route;
- no OpenVLA-OFT;
- no full benchmark or simulator rollout until WSL/Linux/MuJoCo readiness is handled separately.

Official simulator eval remains a separate WSL/Linux/MuJoCo readiness milestone. Do not substitute the archived custom replay bridge as official eval.
