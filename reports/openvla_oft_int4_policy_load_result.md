
# OpenVLA-OFT INT4 Policy Load Result

- INT4 load status: `True`
- quantized: `true`
- full precision claim: `false`
- model parameter: `{'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'name': 'vision_backbone.featurizer.cls_token'}`
- action head parameter: `{'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'name': 'model.layer_norm1.weight'}`
- proprio projector parameter: `{'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'name': 'fc1.weight'}`
- official input device constant during hard-slice: `cuda:0`
- action chunk shape: `[8, 7]`
- action range: `{'finite': True, 'max': 1.0, 'min': -0.039105047547121186}`
- unnormalization key in smoke: `libero_spatial_no_noops`
- offload status: `NO_CPU_OR_DISK_OFFLOAD_DETECTED`
- autocast: `{'cpu': False, 'cuda': False}`
- peak CUDA allocated in load smoke: `5539.458` MiB
- one-episode smoke: success `True`, steps `129`, video `./rollouts/2026_07_11/2026_07_11-13_19_57--openvla_oft--episode=9001--success=True--task=pick_up_the_black_bowl_in_the_top_drawer_of_the_wo.mp4`
- hard-slice INT4 rollout: `20/20` completed, `20/20` succeeded

Initial load with inherited accelerate `1.14.0` failed because `dispatch_model` called `.to()` on a bitsandbytes quantized model. Pinning `accelerate==0.25.0` fixed the official INT4 local-path load. No CPU fallback occurred after the fix.
