
# OpenVLA-OFT Quantized Hard-Slice Result

Final decision: `FAILURE_NOT_REPRODUCED_IN_SECOND_ARCHITECTURE`

- OpenVLA-OFT variant: `INT4 quantized`
- full precision claim: `false`
- training/fine-tuning happened: `false`
- full BF16 attempted: `false`
- CPU/disk offload: `false`
- exact manifest: `20260711..20260715 -> official init_state[0..4]` per task
- videos: OpenVLA `20/20`, SmolVLA exact `20/20`

| Task | OpenVLA-OFT INT4 | SmolVLA frozen-base exact-init |
| --- | ---: | ---: |
| `libero_10/task_2` | `5/5` | `4/5` |
| `libero_10/task_4` | `5/5` | `1/5` |
| `libero_spatial/task_2` | `5/5` | `5/5` |
| `libero_spatial/task_4` | `5/5` | `1/5` |

OpenVLA hard-slice peak CUDA allocated was `5539.458` MiB. SmolVLA exact peak CUDA allocated was `926.638` MiB. The prior SmolVLA failures were not reproduced in quantized OpenVLA-OFT INT4, so cross-backbone failure generality is blocked and LIBERO-PRO is not justified.
