# Next Actions

Date: 2026-07-09 KST

Current decision: `READY_FOR_OFFICIAL_BASELINE_SCALEUP`

## Immediate Next Action

Create a bounded official baseline scaleup script/run.

Required boundary:

- official assets only: `C:\assets\checkpoints\smolvla_libero` and `C:\assets\datasets\lerobot_libero`;
- standard LoRA only, rank `4`;
- batch size `1`;
- fixed small step count under the repo training budget;
- runtime under 30 minutes;
- log model/input devices, CUDA allocated/max memory, autocast/fp16/bf16 status, loss before/after, gradients, and output action validity;
- no RA-L method, no OpenVLA-OFT, no full benchmark, no simulator rollout.

Official simulator eval remains a separate WSL/Linux/MuJoCo readiness milestone. Do not substitute the archived custom replay bridge as official eval.
