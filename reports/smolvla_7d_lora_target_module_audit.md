# SmolVLA 7D LoRA Target Module Audit

- current projection modules: `{'requested': ['state_proj', 'action_in_proj', 'action_out_proj'], 'executed_for_fixed_7d': ['state_proj'], 'not_executed': ['action_in_proj', 'action_out_proj'], 'reason': 'action_in_proj/action_out_proj are native flow-action modules tied to max_action_dim/SO100 6D action preparation.'}`
- action head / 7D adapter only: `{'executed': True, 'variant': 'small_state_time_mlp_7d_baseline'}`
- projection + 7D adapter: `{'executed': True, 'variants': ['smolvla_7d_adapter_no_lora', 'smolvla_state_proj_lora_rank4_7d_adapter', 'smolvla_state_proj_lora_rank8_7d_adapter']}`
- projection + action head if available: `{'executed': True, 'interpretation': 'state_proj LoRA plus learned LIBERO_7D adapter head'}`
- strict boundary: No target-module variant uses the old hard-coded gripper fill or SO100 action normalizer for LIBERO labels.
