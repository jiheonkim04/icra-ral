
# OpenVLA-OFT Quantized Cross-Backbone Decision

Final decision: `FAILURE_NOT_REPRODUCED_IN_SECOND_ARCHITECTURE`

Why: quantized OpenVLA-OFT INT4 ran the frozen hard-slice manifest on the local RTX 5080 with no CPU/disk offload and succeeded on `20/20` episodes. The matched exact-init SmolVLA frozen-base rerun still failed on the hard slices (`libero_spatial/task_4 = 1/5`, `libero_10/task_4 = 1/5`), so the prior SmolVLA failure mechanisms were not reproduced in the second architecture.

- branch: `codex/rtx5080-int4-openvla-cross-backbone`
- checkpoint: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` @ `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- checkpoint size: `15939168050` bytes (`14.845` GiB)
- download happened: `true`
- training happened: `false`
- INT4 load status: `True`
- INT8 diagnostic status: `True`
- CPU/disk offload status: `NO_CPU_OR_DISK_OFFLOAD_DETECTED`
- peak OpenVLA VRAM/RAM: `5539.458` MiB CUDA allocated / `4637.605` MiB RSS
- episodes planned/completed: OpenVLA `20/20`, SmolVLA exact `20/20`, total `40/40`
- videos recorded: OpenVLA `20/20`, SmolVLA exact `20/20`
- quantization limitation: INT4 is quantized and not a full-precision OpenVLA-OFT claim
- failure generalizes across SmolVLA and quantized OpenVLA-OFT: `false`
- LIBERO-PRO justified now: `false`

Exact next step: stop method design and do not proceed to LIBERO-PRO from this evidence. A future run would need either full-precision second-backbone evidence on suitable hardware or a new predeclared hard slice where the second architecture also fails with the same visible physical phase.
