
# OpenVLA-OFT INT4/INT8 Consistency Smoke

- INT8 run status: `True`
- diagnostic only: `true`
- full precision ablation: `false`
- INT4/INT8 action chunk L2: `1.1724120069754282`
- mean absolute difference: `0.05609847400364539`
- gripper agreement: `1.0`
- first-action translation cosine: `0.3274977792286874`
- first-action rotation cosine: `0.906135772799668`
- INT4 peak VRAM: `5539.458` MiB
- INT8 peak VRAM: `8391.52` MiB

Interpretation: INT8 was safe as a bounded diagnostic, but the continuous-action difference means the final claim remains explicitly about quantized INT4 OpenVLA-OFT, not full-precision OpenVLA-OFT.
