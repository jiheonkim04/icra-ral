# RL4IL Action-Oracle Prior Module Training Result

- Execution classification: `PRIOR_MODULE_TRAINING`
- Implementation label: `MECHANISM_FAITHFUL_RL4IL_LOCAL_PORT`
- Success: `True`
- Trainable prior parameters: `13512777`
- Optimizer steps: `54`
- Finite nonzero gradients: `True`
- Weights changed: `True`
- Checkpoint reload OK: `True`
- CUDA PID: `295`
- Peak VRAM MiB: `655.63525390625`

This is external RL4IL retrieval/imputation prior-module training, not VLA training and not Ours.

## Component summary

| task | component | params | steps | first loss | final loss | grad nonzero | changed | checkpoint |
|---|---|---:|---:|---:|---:|---|---|---|
| `libero_goal_task0` | `clean_retrieval_policy` | 1117185 | 3 | 0.449195921421051 | 0.5920188426971436 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_goal_task0/clean_retrieval_policy.pt` |
| `libero_goal_task0` | `clean_action_fusion_head` | 541376 | 3 | 0.2314879298210144 | 0.21550878882408142 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_goal_task0/clean_action_fusion_head.pt` |
| `libero_goal_task0` | `imputation_policy_mod1` | 985857 | 3 | 0.42405691742897034 | 0.4928993880748749 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_goal_task0/imputation_policy_mod1.pt` |
| `libero_goal_task0` | `soft_imputation_head_mod1` | 201280 | 3 | 0.22847744822502136 | 0.22181229293346405 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_goal_task0/soft_imputation_head_mod1.pt` |
| `libero_goal_task0` | `mask1_retrieval_policy` | 1117185 | 3 | 0.6109839081764221 | 0.4728247821331024 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_goal_task0/mask1_retrieval_policy.pt` |
| `libero_goal_task0` | `mask1_action_fusion_head` | 541376 | 3 | 0.2352023720741272 | 0.2164900302886963 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_goal_task0/mask1_action_fusion_head.pt` |
| `libero_object_task0` | `clean_retrieval_policy` | 1117185 | 3 | 0.5056449770927429 | 0.6024919748306274 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_object_task0/clean_retrieval_policy.pt` |
| `libero_object_task0` | `clean_action_fusion_head` | 541376 | 3 | 0.22449839115142822 | 0.21528524160385132 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_object_task0/clean_action_fusion_head.pt` |
| `libero_object_task0` | `imputation_policy_mod1` | 985857 | 3 | 0.5357648134231567 | 0.5970283150672913 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_object_task0/imputation_policy_mod1.pt` |
| `libero_object_task0` | `soft_imputation_head_mod1` | 201280 | 3 | 0.2426597774028778 | 0.2335430085659027 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_object_task0/soft_imputation_head_mod1.pt` |
| `libero_object_task0` | `mask1_retrieval_policy` | 1117185 | 3 | 0.5459299087524414 | 0.3811456859111786 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_object_task0/mask1_retrieval_policy.pt` |
| `libero_object_task0` | `mask1_action_fusion_head` | 541376 | 3 | 0.2239678055047989 | 0.2129165530204773 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_object_task0/mask1_action_fusion_head.pt` |
| `libero_spatial_task5` | `clean_retrieval_policy` | 1117185 | 3 | 0.528636634349823 | 0.542818009853363 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_spatial_task5/clean_retrieval_policy.pt` |
| `libero_spatial_task5` | `clean_action_fusion_head` | 541376 | 3 | 0.25090494751930237 | 0.23668688535690308 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_spatial_task5/clean_action_fusion_head.pt` |
| `libero_spatial_task5` | `imputation_policy_mod1` | 985857 | 3 | 0.3825641870498657 | 0.5435097813606262 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_spatial_task5/imputation_policy_mod1.pt` |
| `libero_spatial_task5` | `soft_imputation_head_mod1` | 201280 | 3 | 0.26824116706848145 | 0.2606450915336609 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_spatial_task5/soft_imputation_head_mod1.pt` |
| `libero_spatial_task5` | `mask1_retrieval_policy` | 1117185 | 3 | 0.46231016516685486 | 0.3791094720363617 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_spatial_task5/mask1_retrieval_policy.pt` |
| `libero_spatial_task5` | `mask1_action_fusion_head` | 541376 | 3 | 0.24505513906478882 | 0.23081061244010925 | `True` | `True` | `/mnt/c/Users/jiheo/tca_map/runs/rl4il_prior/action_oracle_port_20260718T194750KST/prior_module_training/libero_spatial_task5/mask1_action_fusion_head.pt` |
