# SmolVLA 7D Baseline Reproduction

Final decision: `READY_FOR_RA_L_METHOD_ON_SMOLVLA_7D`

This is a standard fixed-interface baseline reproduction, not a new method or paper claim.

## Summary

- model used: `C:\assets\checkpoints\smolvla`
- dataset/split used: `same_task_demo_holdout`
- LoRA ranks tested: `[4, 8]`
- target modules tested: `['libero_7d_adapter_head_only', 'frozen_state_proj_plus_7d_adapter', 'state_proj_lora_plus_7d_adapter']`
- experiments happened: `True`
- training happened: `True`
- loss computed: `True`
- GPU training happened: `False`
- downloads happened: `False`
- OpenVLA-OFT happened: `False`
- mean-action metric: `1.082453`
- ridge/MLP metric: `0.518738`
- frozen/base metric: `0.890604`
- best LoRA/adapter metric: `0.494959`
- best LoRA metric: `0.494959`
- LoRA beats mean-action: `True`
- LoRA beats MLP/ridge: `True`
- VRAM peak MB: `0.0`
- runtime sec: `9.438`
- trainable params: `{'small_mlp': 487, 'frozen_base_smolvla_7d_linear_adapter': 6734, 'smolvla_7d_adapter_no_lora': 124039, 'smolvla_state_proj_lora_rank4_7d_adapter': 128007, 'smolvla_state_proj_lora_rank8_7d_adapter': 131975}`
- exact next step: Future method planning may start only after preserving this fixed-interface baseline table and predeclaring simple baselines.
