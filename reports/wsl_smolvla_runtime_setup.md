# WSL SmolVLA Runtime Setup

This rung prepares the WSL-only topology for the first tiny learned-policy LIBERO rollout.

The simulator is ready in WSL, while the original SmolVLA load/interface smoke passed in the Windows conda environment. Before any learned-policy rollout, the selected WSL venv must be able to see the lightweight SmolVLA runtime modules from local assets.

## Scripts

Planning-only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\67_plan_wsl_smolvla_runtime_setup.ps1
```

Bounded setup, only after the planner says `proceed`:

```powershell
$env:ALLOW_WSL_SMOLVLA_RUNTIME_SETUP="1"
powershell -ExecutionPolicy Bypass -File scripts\68_setup_wsl_smolvla_runtime_deps.ps1
Remove-Item Env:\ALLOW_WSL_SMOLVLA_RUNTIME_SETUP -ErrorAction SilentlyContinue
```

## Scope

The setup may install packages only into:

```text
/home/jiheon/.venvs/tca_map_sim
```

It must not create a repo-local `.venv`, use sudo, use apt, change CUDA or drivers, download OpenVLA-OFT, load SmolVLA, run inference, train, rollout, use GPU, access tokens, or make paper claims.

The package plan uses:

- PyTorch official CPU wheel index for `torch` and `torchvision`,
- PyPI for `transformers`, `safetensors`, `huggingface_hub`, `accelerate`, and `num2words`,
- PyPI for the LeRobot import/runtime packages required by the WSL single-action smoke: `draccus`, `datasets`, `imageio[ffmpeg]`, `diffusers`, `pyserial`, `deepdiff`, `av`, and `einops`,
- `lerobot==0.4.4` with `--no-deps` to avoid broad simulator venv dependency drift before a separate WSL load-only smoke.

The current local WSL venv has passed module-spec readiness and the bounded WSL SmolVLA single-action smoke after these runtime packages were added. This remains engineering readiness only.

## Evidence Label

Passing this setup is runtime readiness only. It is not model-load evidence, inference evidence, rollout evidence, benchmark success, or paper-grade evidence.
