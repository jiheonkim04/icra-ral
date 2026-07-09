# SmolVLA LoRA Baseline STATE 1 Result

Bounded standard LoRA baseline reproduction on local LIBERO HDF5 data. This is not a new method, rollout, full benchmark, OpenVLA-OFT run, or paper claim.

- final decision: `KILL_MEAN_BASELINE_DOMINATED`
- model used: `C:\assets\checkpoints\smolvla`
- dataset: `C:\assets\data\libero\libero_10\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5`
- split: `deterministic_demo_holdout`
- train demos: `['demo_0', 'demo_1', 'demo_2']`
- eval demos: `['demo_3', 'demo_4']`
- train/eval counts: `9 / 6`
- training happened: `True`
- loss computed: `True`
- LoRA rank: `4`
- trainable params: `9984`
- VRAM peak MB: `1190.228`
- runtime sec: `43.765`
- loss start/end: `0.06359 / 0.008743`
- loss decreased meaningfully: `True`

## Eval Metrics

| variant | action L2 | first6 L2 | translation L2 | rotation L2 | gripper error | gripper accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_action | 0.486561 | 0.486561 | 0.474176 | 0.098626 | 0.0 | 1.0 |
| frozen_base_smolvla | 1.6029 | 1.6029 | 0.7074 | 1.429045 | 0.0 | 1.0 |
| standard_lora | 0.940196 | 0.940196 | 0.648108 | 0.638062 | 0.0 | 1.0 |

- LoRA beats mean-action baseline: `False`
- LoRA beats frozen/base baseline: `True`
- LoRA learns: `True`

Exact next step: Standard LoRA did not beat the mean-action baseline.
