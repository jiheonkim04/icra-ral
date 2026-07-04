# WSL SmolVLA Single-Action Smoke

This rung verifies that the WSL-only runtime can load the local SmolVLA policy and produce one synthetic action before any learned-policy LIBERO rollout is attempted.

Planning-only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\69_plan_wsl_smolvla_single_action_smoke.ps1
```

Bounded execution, only after the planner says `proceed`:

```powershell
$env:ALLOW_WSL_SMOLVLA_SINGLE_ACTION="1"
powershell -ExecutionPolicy Bypass -File scripts\70_bounded_wsl_smolvla_single_action_smoke.ps1
Remove-Item Env:\ALLOW_WSL_SMOLVLA_SINGLE_ACTION -ErrorAction SilentlyContinue
```

The bounded execution uses:

- `/home/jiheon/.venvs/tca_map_sim/bin/python`,
- local SmolVLA checkpoint files under `C:\assets\checkpoints\smolvla`,
- local Hugging Face cache under `C:\assets\hf_home`,
- `HF_HUB_OFFLINE=1`,
- `TRANSFORMERS_OFFLINE=1`,
- CPU device by default,
- one synthetic observation and one action selection.

It must not create simulator environments, rollout, train, use GPU, download, execute OpenVLA-OFT, access tokens, or make paper claims.

Passing this rung is model-load/action-interface evidence only. It is not rollout evidence, benchmark success, SOTA evidence, or paper-grade evidence.

## Current Local Result

The bounded WSL run has passed with task-local `ALLOW_WSL_SMOLVLA_SINGLE_ACTION=1`.

Observed smoke metrics:

```text
device=cpu
load_and_interface_elapsed_sec=19.233
single_sample_inference_elapsed_sec=1.665
action_shape=[1, 6]
action_finite=true
rss_before_mb=215.277
rss_after_mb=987.48
gpu_jobs_performed=false
training_performed=false
real_rollouts_performed=false
openvla_oft_executed=false
```

The WSL setup needed additional LeRobot import/runtime packages before this passed: `draccus`, `datasets`, `imageio[ffmpeg]`, `diffusers`, `pyserial`, `deepdiff`, `av`, and `einops`. They were installed only into `/home/jiheon/.venvs/tca_map_sim`.
