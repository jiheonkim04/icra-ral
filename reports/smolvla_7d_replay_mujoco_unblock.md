# SmolVLA 7D Replay Bridge MuJoCo Unblock

Date: 2026-07-09 KST

Final decision: `READY_FOR_METHOD_AFTER_REPLAY_BRIDGE`

## Scope

This report records the bounded environment unblock for the existing SmolVLA 7D replay bridge. It does not introduce a new method, does not change the adapter method, does not run OpenVLA-OFT, does not run a full benchmark, and does not make a paper claim.

## Environment Diagnosis

- replay Python: `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe`
- Python version: `3.10.20`
- conda env by path: `tca_map`
- bridge path: Windows conda Python, not WSL
- WSL status: `/usr/bin/python3` available, but no `mujoco`, `robosuite`, or `libero`
- required headless setting on Windows: `MUJOCO_GL=glfw`

Plain `mujoco` import now works in the active env: version `2.3.7`.

`robosuite` and `libero` are not plain site-package imports in this setup; the replay bridge imports them through the configured source roots:

- `C:\assets\repos\robosuite`
- `C:\assets\repos\LIBERO`

With the bridge source-path/config setup, imports passed:

- `mujoco`: `2.3.7`
- `robosuite`: `1.4.0`
- `libero.libero.envs.OffScreenRenderEnv`: import OK

## Install and Local Environment Changes

Install happened: yes.

Bounded changes:

- installed `mujoco>=2.3.0,<4`, which initially resolved to `3.10.0`
- installed minimal missing RoboSuite/LIBERO Python dependencies needed for import: `numba`, `scipy`, `opencv-python`, `bddl`, `gym`, `matplotlib`, `easydict`, `future`
- copied the active MuJoCo DLL into local RoboSuite source at `C:\assets\repos\robosuite\robosuite\utils\mujoco.dll`
- set local RoboSuite `macros.py` to `MUJOCO_GPU_RENDERING = False` so Windows does not force unsupported `egl`
- downgraded MuJoCo to `mujoco==2.3.7` after RoboSuite 1.4 hit the newer `mj_fullM` signature in MuJoCo 3.10
- recopied the MuJoCo 2.3.7 DLL into the local RoboSuite source

No large assets, CUDA reinstall, OpenVLA-OFT, or full benchmark run happened.

## Replay Result

Task/demo:

`KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5::demo_30`

Replay/control happened: yes, bounded exact-init only.

| policy | reward_sum | success | first_done | progress_proxy | object_displacement |
| --- | ---: | --- | ---: | ---: | ---: |
| expert | `1.0` | `True` | `250` | `0.229161` | `0.200001` |
| mean_action | `0.0` | `False` |  | `0.106222` | `0.000004` |
| ridge | `0.0` | `False` |  | `0.167573` | `0.015402` |
| SmolVLA 7D adapter | `0.0` | `False` |  | `0.234297` | `0.03555` |

Offline action L2:

| policy | action_l2 | translation_l2 | rotation_l2 | gripper_error |
| --- | ---: | ---: | ---: | ---: |
| expert | `0.0` | `0.0` | `0.0` | `0.0` |
| mean_action | `1.104166` | `0.442839` | `0.10286` | `0.965032` |
| ridge | `0.893329` | `0.406915` | `0.094357` | `0.729167` |
| SmolVLA 7D adapter | `0.464353` | `0.252312` | `0.068959` | `0.33562` |

Adapter action validity:

- clip rate element: `0.061303`
- clip rate step: `0.429119`
- controller-valid proxy rate: `0.570881`

## Interpretation

Expert replay succeeds, so the replay bridge is no longer blocked by the environment. The adapter is executable, and its target-distance progress proxy beats mean and ridge (`0.234297` versus `0.106222` and `0.167573`), matching the direction of the offline L2 improvement.

The adapter still does not solve the task in this one-demo bounded replay, and its gripper range has nontrivial clipping. This is enough to unblock the real LoRA path, not enough to claim a new method.

## Exact Next Step

Reproduce a real SmolVLA LoRA baseline on an official or standard LIBERO task split using this environment. Do not start a new method until standard LoRA baseline behavior and replay transfer are understood.
