# Next Actions

Date: 2026-07-10 KST

Current decision: `FCAR_KILLED_BY_STATIC_BASELINE`

## Immediate Next Action

Do not scale FCAR.

Result boundary:

- official assets only: `C:\assets\checkpoints\smolvla_libero` and `C:\assets\datasets\lerobot_libero`;
- compact official per-frame base/LoRA predictions were regenerated and saved at `reports/fcar_prediction_artifact.json`;
- FCAR tiny gate was implemented and trained only as a small CPU gate;
- no SmolVLA backbone training happened;
- fixed rank-4 LoRA was regenerated only as the required baseline artifact source;
- FCAR gate-test action L2 was `0.100144625`;
- frozen/base gate-test action L2 was `0.123998278`;
- rank-4 LoRA gate-test action L2 was `0.076191123`;
- val-selected static mixture `w=0.5` gate-test action L2 was `0.091179973`;
- final decision is `FCAR_KILLED_BY_STATIC_BASELINE`;
- no archived custom `LIBERO_7D` adapter route;
- no OpenVLA-OFT;
- no full benchmark or simulator rollout until WSL/Linux/MuJoCo readiness is handled separately.

Preserve the FCAR artifacts and reports for audit, but do not continue FCAR scaleup from this result. Official simulator eval remains a separate WSL/Linux/MuJoCo readiness milestone. Do not substitute the archived custom replay bridge as official eval.
