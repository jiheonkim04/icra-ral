# WSL Official Rollout Environment Lock

Date: 2026-07-10 KST

## Source And Package Lock

- Official runner package: `lerobot==0.4.4`
- Official CLI verified: `lerobot-eval --help`
- LIBERO package: `hf_libero==0.1.4`
- LIBERO copied source revision: `8f1084e`
- PyTorch: `torch==2.10.0+cu128`
- TorchVision: `torchvision==0.25.0+cu128`
- PEFT: `peft==0.19.1`
- Transformers: `transformers==4.57.6`
- Accelerate: `accelerate==1.14.0`
- MuJoCo: `mujoco==3.8.1`
- RoboSuite: `robosuite==1.4.0`
- safetensors: `safetensors==0.8.0`
- EGL probe: `egl_probe==1.0.2`, `hf_egl_probe==1.0.2`
- `pip check`: `No broken requirements found.`

Full package captures:

- `reports/wsl_official_rollout_pip_freeze.txt`
- `reports/wsl_official_rollout_conda_list.txt`

## Official LIBERO Contract Verified

- Control mode: official relative control.
- Observation state after official LIBERO processor: 8D.
- Images: official `observation.images.image` and `observation.images.image2`.
- Checkpoint camera mapping:
  - `observation.images.image` -> `observation.images.camera1`
  - `observation.images.image2` -> `observation.images.camera2`
- Missing checkpoint `camera3` is filled through official SmolVLA `empty_cameras=1`.
- Action chunk: `[1, 50, 7]`, continuous 7D actions.
- Official policy preprocessor, postprocessor, action queue, and `eval_policy_all` loop were used.

## CUDA Audit

All policy-load audits reported:

- model parameter device: `cuda:0`
- input tensor devices: `cuda:0`
- action chunk device: `cuda:0`
- frozen-base parameter dtype: `torch.bfloat16`
- action chunk dtype: `torch.float32`
- autocast CUDA active: `False`
- fp16/bf16 AMP active: `False`
- no `CPU_FALLBACK_BUG`

Peak CUDA allocation during pilot policy audits was about `928.365 MiB`; after rollout the peak remained under `929 MiB` for LoRA policies.

## Runtime Notes

The official rollout completed successfully. RoboSuite emitted non-fatal EGL context cleanup warnings after completion (`EGL_NOT_INITIALIZED` during destructor cleanup). These happened after metrics were written and did not stop rollout execution.
