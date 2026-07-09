# SmolVLA LoRA Capacity Audit

- split train/eval: `9 / 6`
- mean-action eval action L2: `0.486561`
- best LoRA: `current_projection_lora`
- best LoRA eval action L2: `0.912258`
- best LoRA eval per-dim MAE: `[0.205597, 0.46348, 0.183665, 0.194572, 0.603922, 0.162997, 0.0]`
- current LoRA train action L2: `0.856091`
- current LoRA eval action L2: `0.912258`
- best small MLP/ridge: `state_time_mlp`
- best small MLP/ridge eval action L2: `0.401848`
- best small MLP/ridge eval per-dim MAE: `[0.19739, 0.206432, 0.18908, 0.043752, 0.081978, 0.029857, 0.064771]`
- LoRA beats mean-action: `False`
- LoRA beats small MLP/ridge: `False`
- one-sample overfit passed: `False`
- one-demo overfit passed: `False`
- runtime sec: `56.109`
