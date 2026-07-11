
# OpenVLA-OFT INT4 Memory Preflight

- decision: `INT4_PREFLIGHT_OK`
- RTX 5080 VRAM total/free at preflight: `16302.562` / `14955.0` MiB
- WSL RAM total/available at preflight: `11.266` / `9.912` GiB
- full BF16 expected: `>14.8 plus overhead; forbidden`
- INT4 expected: `4.0-7.5 plus bf16 vision/projector/action-head overhead`
- INT8 expected: `8.0-11.5 plus bf16 vision/projector/action-head overhead`
- completed INT4 hard-slice peak CUDA allocated: `5539.458` MiB
- `/usr/bin/time` swaps during INT4 hard-slice: `0`
