# Project State

## Repository State

Canonical repository root:

```text
C:\Users\jiheo\tca_map
```

GitHub repository:

```text
jiheonkim04/icra-ral
```

Canonical branch:

```text
main
```

Current main commit at this update:

```text
143e2e6 or newer
```

Use explicit Python for validation:

```text
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe
```

## Completed Major Features

- scaffold and dummy smoke,
- preflight and smoke reports,
- compute budget guard,
- no-large local OpenVLA-OFT policy,
- Distributional TCA-Select scaffold,
- LoRA/QLoRA config guards,
- LoRA/QLoRA required experiment-track policy,
- LoRA adapter construction planning scaffold,
- LoRA tiny-smoke scaffold,
- TCA-Map + LoRA comparison plan,
- QLoRA feasibility check,
- LoRA/QLoRA go/no-go status update,
- head-only ActionMap vs TCA-Map tiny comparison report,
- local paper-grade runner and planning scripts,
- LIBERO dataset risk planner,
- LIBERO/RoboSuite official source-resolution planner,
- simulator readiness risk planner,
- bounded local pilot budget alignment and extension runner,
- Cursor safe local runner,
- SmolVLA asset prep,
- SmolVLA readiness semantics split,
- SmolVLA download plan guard,
- Windows Bash shim handling for Bash-specific tests,
- manual SmolVLA acquisition checklist,
- Codex delegation manual and project state files,
- SmolVLA load-only adapter smoke planning guard,
- SmolVLA load-only execution scaffold,
- SmolVLA runtime dependency checker and install risk plan,
- feature-cache interface contract and dummy cache planner,
- eval-only cached-feature smoke for the head/metric interface,
- tiny head-only pilot risk planner,
- bounded tiny head-only smoke runner,
- risk-gate status summary,
- go/no-go status summary,
- bounded SmolVLA runtime package install,
- SmolVLA autonomous pilot risk policy,
- bounded SmolVLA load-only model construction smoke on CPU.

## Current Asset Status

Known current state:

```text
SMOLVLA_CKPT=C:\assets\checkpoints\smolvla
CHECKPOINT_ROOT=C:\assets\checkpoints
HF_HOME=C:\assets\hf_home
```

The SmolVLA checkpoint directory exists. The approved SmolVLA source has been acquired from `lerobot/smolvla_base`.

Approved SmolVLA checkpoint source for the acquisition task:

```text
lerobot/smolvla_base
```

Acquisition target:

```text
C:\assets\checkpoints\smolvla
```

Cache target:

```text
C:\assets\hf_home
```

The approved tokenizer/processor dependency source has also been acquired:

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

Dependency target:

```text
C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct
```

Only tokenizer/processor/config files are retained for this dependency. Full SmolVLM2 model weights were avoided.

The original acquisition decisions were limited to SmolVLA checkpoint acquisition from `lerobot/smolvla_base` and tokenizer/processor/config dependency acquisition from `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. They did not authorize GPU jobs, model inference, training, rollouts, OpenVLA-OFT execution/download, dataset downloads, token/secret access, or committing checkpoint/cache files. The current risk policy permits bounded SmolVLA load-only heavy import/model construction and tiny smoke steps only when the task risk assessment is green and inside the autonomous pilot budget.

Current checker output after acquiring `lerobot/smolvla_base` and its tokenizer/processor dependency:

```text
ready_for_smolvla_path_check=true
smolvla_checkpoint_files_present=true
ready_for_smolvla_adapter_smoke=true
ready_for_openvla_oft_smoke=false
ready_for_libero_rollout=false
```

Detected checkpoint files include `config.json`, `model.safetensors`, `policy_preprocessor.json`, `policy_postprocessor.json`, and processor safetensors.

`policy_preprocessor.json` references tokenizer/model source:

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

The external tokenizer/processor dependency is now detected under `C:\assets\hf_home`. The current gate is Case C from the self-check gate policy: readiness is true, the load-only adapter smoke plan is prepared, and the bounded execution scaffold exists. Runtime packages are now installed, and bounded load-only model construction is allowed only inside the SmolVLA risk-assessed autonomous pilot budget.

Current runtime dependency probe:

```text
torch=2.10.0+cu128
torchvision=0.25.0+cu128
transformers=4.57.6
lerobot=0.4.4
safetensors=0.8.0
huggingface_hub=0.35.3
accelerate=1.14.0
num2words=0.5.14
```

The runtime install used a bounded package-install decision. The risk-assessed autonomous pilot policy now authorizes bounded SmolVLA load-only heavy import/model construction, single-sample interface smoke, tiny feature-cache/interface validation, and tiny head-only training smoke when the risk assessment is green and inside budget. It still does not authorize rollouts, simulator execution, OpenVLA-OFT, token access, paper-grade claims, dataset downloads, major CUDA/PyTorch changes, unplanned large package installs, or jobs over the runtime/VRAM budget unless the relevant risk policy explicitly passes.

The bounded SmolVLA load-only smoke has now passed on CPU with `ALLOW_HEAVY_IMPORT=1` set only inside that task. It loaded the local SmolVLA policy from `C:\assets\checkpoints\smolvla`, used the local tokenizer/processor dependency under `C:\assets\hf_home`, kept `load_vlm_weights=false`, and did not run inference, training, rollouts, OpenVLA-OFT, downloads, or token access.

Observed load-only smoke metrics:

```text
device=cpu
load_elapsed_sec=22.531
total_elapsed_sec=28.188
parameter_count=450046176
trainable_parameter_count=99880992
cuda_max_allocated_mb=0.0
rss_before_mb=651.992
rss_after_mb=1323.711
downloads_performed=false
model_inference_performed=false
training_performed=false
real_rollouts_performed=false
openvla_oft_executed=false
```

The bounded tiny head-only smoke has passed on cached/dummy features. It trained tiny CPU NumPy ActionMap and TCA-Map heads for 16 steps, overrode the 300-step local pilot config value to the smoke cap, and did not import SmolVLA, load a model, run VLA inference, use GPU, rollout, execute OpenVLA-OFT, download assets, or make paper claims.

The head-only ActionMap vs TCA-Map tiny comparison report is generated by:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\36_compare_head_only_tiny_pilot.ps1
```

It reads the existing bounded tiny head-only smoke report and compares only offline proxy metrics. It does not train, run inference, import heavy VLA models, download assets, use GPU, rollout, execute simulators, execute OpenVLA-OFT, or make paper claims.

Observed tiny head-only smoke metrics:

```text
cache_valid=true
max_steps=16
max_steps_cap=100
elapsed_seconds=0.00644
actionmap_action_l1=0.078738
actionmap_action_mse=0.012221
tca_map_action_l1=0.071502
tca_map_action_mse=0.010418
tca_map_target_top1_accuracy=0.75
tca_map_wrong_target_proxy_rate=0.25
tca_map_offline_standard_proxy=0.696373
max_gpu_memory_mb=0.0
downloads_performed=false
gpu_jobs_performed=false
heavy_model_imports_performed=false
model_load_performed=false
model_inference_performed=false
rollouts_performed=false
openvla_oft_executed=false
paper_grade_claims_made=false
```

During the bounded load-only debugging path, the SmolVLM processor required the small Python package `num2words`. The environment now has `num2words==0.5.14`, and `requirements.txt` plus the runtime dependency checkers include it so future environments fail early before load-only construction.

The single-sample interface smoke is now scaffolded as:

```powershell
scripts\28_smolvla_single_sample_interface_smoke.ps1
```

It requires both `ALLOW_HEAVY_IMPORT=1` and `ALLOW_SINGLE_SAMPLE_INFERENCE=1` inside the bounded task. It uses one synthetic state/image/text input, one local-only CPU `select_action` call, and writes `reports\smolvla_single_sample_interface_report.json`.

The bounded single-sample interface smoke has now passed on CPU. It used one synthetic observation, one local tokenizer pass, and one `select_action` call. It did not train, rollout, evaluate datasets, execute OpenVLA-OFT, download assets, access tokens, or make paper claims.

Observed single-sample interface smoke metrics:

```text
device=cpu
num_steps=1
load_and_interface_elapsed_sec=29.75
single_sample_inference_elapsed_sec=1.657
action_shape=[1, 6]
action_finite=true
cuda_max_allocated_mb=0.0
rss_before_mb=478.16
rss_after_mb=1695.855
downloads_performed=false
training_performed=false
real_rollouts_performed=false
openvla_oft_executed=false
```

The dummy feature-cache/interface validation has also passed. The planner wrote a dummy cache under `runs\feature_cache\dummy_contract`, validated `manifest.json` plus `features.jsonl`, and the eval-only cached-feature smoke consumed 4 records through the head/metric path.

Observed feature-cache interface metrics:

```text
cache_valid=true
cache_record_count=4
hidden_dim=12
offline_standard_proxy=0.157986
target_top1_accuracy=0.25
wrong_target_proxy_rate=0.75
downloads_performed=false
heavy_model_imports_performed=false
model_inference_performed=false
training_performed=false
rollouts_performed=false
openvla_oft_executed=false
```

Runtime dependency checker:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\17_check_smolvla_runtime_deps.ps1
```

Runtime install risk planner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\18_plan_smolvla_runtime_install.ps1
```

This planner is check-only. It writes `reports\smolvla_runtime_install_plan_report.json`, refuses dangerous gates such as `ALLOW_DOWNLOADS=1` or `ALLOW_HEAVY_IMPORT=1`, and does not install packages.

Other missing readiness inputs currently expected:

- OpenVLA-OFT checkpoint,
- separate offline comparison runners that consume the acquired LIBERO HDF5 files,
- separate simulator import/render/rollout readiness gates before any rollout metric.

The LIBERO and RoboSuite source checkouts are now path-ready after bounded source repo setup:

```text
LIBERO_ROOT=C:\assets\repos\LIBERO
ROBOSUITE_ROOT=C:\assets\repos\robosuite
LIBERO_DATA_ROOT=C:\assets\data\libero
```

The LIBERO data root now contains the official `yifengzhu-hf/LIBERO-datasets` demonstrations acquired under the dedicated 180 GB LIBERO data gate. Dataset files remain outside git under `C:\assets\data\libero`. The lightweight `h5py>=3.11` reader dependency has been installed after a green dependency risk assessment, and the check-only offline interface gate can read local LIBERO HDF5 action fields. Rollout readiness remains false until a separate simulator risk gate passes.

## Current Gate

The bounded local SmolVLA smoke stack is complete through load-only construction, single-sample interface smoke, dummy feature-cache validation, tiny head-only smoke, tiny LoRA smoke, head-only/LoRA comparison summaries, and the bounded cached-feature local pilot extension.

```text
C:\assets\checkpoints\smolvla
```

Ready SmolVLA file groups:

- `config.json`,
- external tokenizer/processor/config dependency files,
- weights file.

The current executable local path has cleared the LIBERO HDF5 reader boundary:

- `LIBERO_ROOT` exists,
- `ROBOSUITE_ROOT` exists,
- `LIBERO_DATA_ROOT` exists,
- official LIBERO HDF5 demonstration files are present under `LIBERO_DATA_ROOT`,
- the official full LIBERO dataset has been acquired under `C:\assets\data\libero` from `yifengzhu-hf/LIBERO-datasets`; runtime acquisition reports are ignored and dataset files are not committed,
- `h5py` is installed as a reader-only dependency,
- `scripts\50_check_libero_hdf5_reader.ps1` reports `ready_for_libero_hdf5_interface_read=true`,
- `scripts\48_plan_libero_offline_interface_smoke.ps1` reports `ready_for_offline_interface_smoke=true`,
- rollout readiness is still false and requires a separate simulator gate.

Codex should keep running routine readiness and status checks without asking. The tiny real/offline HDF5 split, comparison, LoRA proxy, bounded pilot report path, WSL simulator dependency setup, bounded simulator import-only smoke, bounded render smoke, bounded reset/step smoke, bounded render/reset-step risk planner, toy MuJoCo diagnostic rollout, and bounded LIBERO/RoboSuite zero-action diagnostic rollout have passed. The next autonomous boundary is learned-policy LIBERO rollout readiness with `scripts\66_plan_libero_policy_rollout_readiness.ps1`. Codex must still stop before sudo password entry, token access, license/payment gates, CUDA/driver/graphics-stack changes, OpenVLA-OFT, unsupported paper-grade claims, jobs over 30 minutes, more than 14GB VRAM, multi-seed rollout before a separate budget, or rollout beyond the current risk-assessed policy.

Current planning commands:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\45_resolve_libero_robosuite_sources.ps1
powershell -ExecutionPolicy Bypass -File scripts\42_plan_libero_dataset_risk.ps1
powershell -ExecutionPolicy Bypass -File scripts\43_plan_simulator_readiness.ps1
powershell -ExecutionPolicy Bypass -File scripts\39_generate_local_pilot_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\31_generate_go_no_go_report.ps1
```

These write ignored runtime outputs such as:

```text
reports\libero_dataset_risk_report.json
reports\libero_robosuite_source_resolution_report.json
reports\libero_robosuite_setup_report.json
reports\simulator_readiness_plan_report.json
reports\local_pilot_status_report.json
reports\go_no_go_status_report.json
```

## Validation Commands

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\40_cursor_safe_local_check.ps1
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\13_check_smolvla_adapter_smoke.ps1
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest -q
```

Relevant dry-run planner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\14_plan_smolvla_download.ps1
powershell -ExecutionPolicy Bypass -File scripts\15_plan_smolvla_load_only_smoke.ps1
```

Codex should run these commands itself when routine state is needed. The user should only be asked when risk cannot be assessed, risk exceeds budget, or an external irreversible/OpenVLA/paper-claim gate is reached.

## Risk Gates

Risk assessment required; proceed if inside budget and stop if outside or ambiguous:

- LIBERO/LIBERO-CF dataset setup or download,
- bounded rollouts,
- simulator readiness/import-render smoke,
- real benchmark data handling,
- training more than 300 local pilot steps,
- any job expected to exceed 30 minutes,
- using more than 14GB VRAM,

External stop gates:

- OpenVLA-OFT download/import/load/execution until a separate risk budget exists,
- changing CUDA/PyTorch major versions,
- installing large unplanned packages,
- token or secret handling,
- multi-seed experiments,
- external submission/upload/publishing,
- paper-level empirical claims.

## Research Direction Summary

TCA-Map / Distributional TCA-Map / Distributional TCA-Select is a low-compute VLA action-decoding and counterfactual grounding project.

Main hypothesis:

A VLA should first ground an instruction to a target distribution, then decode a target-conditioned action heatmap. Counterfactual target changes should shift target/action distributions consistently. Nuisance or paraphrase changes that preserve the target should keep distributions stable.

Core method:

- target heatmap / target distribution,
- target-conditioned ActionMap head,
- Distributional TCA-Select,
- counterfactual target/action consistency,
- nuisance invariance,
- required LoRA/QLoRA experiment tracks as compute-saving adaptation arms.

LoRA and QLoRA are required experimental tracks after the head-only path is validated. They are not the main novelty. The main novelty remains target-conditioned action heatmaps, counterfactual target/action consistency, and Distributional TCA-Select.

## Immediate Next Step

Codex should self-check current state. Since config/tokenizer dependency/weights are present, adapter-smoke readiness is true, runtime dependencies are installed, bounded load-only smoke passed, single-sample interface smoke passed, dummy feature-cache/interface validation passed, bounded tiny head-only smoke passed, tiny LoRA smoke passed, the bounded cached-feature local pilot extension passed, and the consolidated go/no-go summaries include that extension, the next safe path is no longer another dummy smoke. The next boundary is real LIBERO/LIBERO-CF-style data and simulator readiness.

The current planner state is:

```text
scripts\45_resolve_libero_robosuite_sources.ps1 -> repo setup decision=proceed, full dataset decision=stop
scripts\46_prepare_libero_robosuite_sources.ps1 -> completed source repo setup only
scripts\42_plan_libero_dataset_risk.ps1 -> decision=stop for full dataset by size budget unless a tiny subset exists
scripts\47_build_libero_metadata_subset.ps1 -> metadata-only target/counterfactual manifest builder, no demos required
scripts\48_plan_libero_offline_interface_smoke.ps1 -> check-only gate for tiny local data files; current decision=proceed after h5py install and local HDF5 inspection
scripts\50_check_libero_hdf5_reader.ps1 -> check-only h5py reader dependency gate; current ready=true
scripts\43_plan_simulator_readiness.ps1 -> decision=proceed for a separate bounded simulator import-smoke plan
scripts\60_link_wsl_simulator_sources.ps1 -> local source-link passed for the selected WSL venv
scripts\55_bounded_simulator_import_smoke.ps1 -> import-only smoke passed
scripts\59_bounded_simulator_render_smoke.ps1 -> tiny MuJoCo OSMesa render smoke passed
scripts\58_plan_simulator_render_reset.ps1 -> reset/step smoke planning is green; rollout remains false
scripts\61_bounded_simulator_reset_step_smoke.ps1 -> tiny MuJoCo reset/step smoke passed; rollout remains false
scripts\62_plan_tiny_diagnostic_rollout.ps1 -> tiny diagnostic rollout planning envelope is bounded; execution is authorized only through task-local ALLOW_TINY_ROLLOUT=1
scripts\63_bounded_tiny_diagnostic_rollout.ps1 -> bounded toy MuJoCo diagnostic rollout passed; benchmark rollout remains false
```

Reasons:

- LIBERO/RoboSuite source paths are now present,
- official LIBERO HDF5 demonstration files exist under `LIBERO_DATA_ROOT`,
- `h5py` is installed and HDF5 offline interface inspection is ready.

Safe autonomous work can continue on checkers, docs, reports, tiny local HDF5 interface reads, counterfactual split construction, offline proxy comparison scaffolds, simulator readiness scaffolds, learned-policy rollout readiness, and tiny benchmark rollout scaffolds whose risk assessment is green. The WSL simulator dependency setup, WSL local source linking, import-only smoke, bounded render smoke, reset/step planning, bounded reset/step smoke, toy MuJoCo diagnostic rollout, and bounded LIBERO/RoboSuite zero-action diagnostic rollout have passed. Real benchmark rollouts require the next learned-policy rollout readiness planner and must not be treated as paper-grade evidence automatically.

The bounded tiny diagnostic rollout boundary is now cleared. A tiny 64x64 MuJoCo offscreen render with `MUJOCO_GL=osmesa` passes in WSL, a tiny in-memory MuJoCo reset plus 3-step physics smoke passes, and the bounded rollout script completed 5 toy MuJoCo diagnostic tasks with 1 episode and 5 steps per task. This is simulator plumbing evidence only. It is not LIBERO/RoboSuite benchmark rollout evidence, not standard success, and not paper-grade evidence.

The LIBERO offline LoRA comparison is implemented as a bounded local proxy diagnostic in `scripts\53_compare_libero_offline_lora.ps1`. It trains only tiny NumPy low-rank adapter matrices over local HDF5 action-prefix snippets, requires `ALLOW_TINY_TRAINING=1`, and does not use GPU, model loading, model inference, simulator execution, rollout, OpenVLA-OFT, token access, or paper-grade claims.

The LIBERO offline bounded pilot report is implemented in `scripts\54_generate_libero_offline_bounded_pilot_report.ps1`. It consolidates existing ignored runtime reports only, marks the result as offline proxy evidence, and keeps simulator execution, rollout, OpenVLA-OFT, token access, GPU jobs, model loading, and paper-grade claims blocked.

The simulator readiness planner remains planning-only and is now a first-class input to `scripts\39_generate_local_pilot_status.ps1` and `scripts\31_generate_go_no_go_report.ps1`. The summaries report path readiness, selected runtime platform, import-smoke readiness, render-smoke readiness, rollout readiness, warnings, and stop reasons without importing simulators, rendering, rolling out, training, using GPU, importing heavy VLA models, executing OpenVLA-OFT, accessing tokens, or making paper claims.

The bounded simulator import smoke scaffold is implemented in `scripts\55_bounded_simulator_import_smoke.ps1`. It requires task-local `ALLOW_SIMULATOR_IMPORT_SMOKE=1`, reruns the planning-only readiness gate, and may attempt only WSL-visible `robosuite` and `libero` Python package imports. It does not render, create or step simulator environments, rollout policies, train, use GPU, download, install packages, import heavy VLA models, execute OpenVLA-OFT, access tokens, or make paper claims.

Current local bounded simulator import smoke result: the script ran under the task-local gate, selected the WSL venv at `$HOME/.venvs/tca_map_sim`, and imported both `robosuite` and `libero`. No render, rollout, simulator environment step, GPU job, training, heavy VLA import, OpenVLA-OFT execution, token access, or paper claim occurred. This clears import-only readiness only; it is not render evidence, rollout evidence, or paper-grade evidence.

The WSL simulator source-link helper is implemented in `scripts\60_link_wsl_simulator_sources.ps1`. It reuses `/home/jiheon/.venvs/tca_map_sim`, performs local editable source linking with `--no-index --no-deps --no-build-isolation`, writes a `.pth` entry for LIBERO's nested source layout, and writes a noninteractive WSL `~/.libero/config.yaml` pointing at the local LIBERO source and data roots. It does not create a repo-local venv, download packages, render, reset/step, rollout, train, use GPU, import heavy VLA models, execute OpenVLA-OFT, access tokens, or make paper claims.

The WSL simulator dependency checker is implemented in `scripts\56_check_wsl_simulator_deps.ps1`. It is check-only and records whether WSL has `python3`, global `pip`/`ensurepip`/`numpy`, the selected venv Python, selected venv `pip`, selected venv `numpy`, and missing modules from the bounded simulator import-smoke report. Current local result: the selected venv is ready for import-only retry, while global WSL Python still lacks pip and numpy. Future dependency additions should stay in the venv and stop before sudo password, token/license/payment, CUDA/driver, graphics-stack, OpenVLA-OFT, render, rollout, or paper-claim gates.

The bounded simulator render/reset-step planner is implemented in `scripts\58_plan_simulator_render_reset.ps1`. Current local result: it reads the passed import-only and render-smoke reports and reports `decision=proceed`, `ready_for_bounded_render_smoke_plan=true`, `ready_for_bounded_reset_step_smoke_plan=true`, and `ready_for_rollout=false`. It performs no render, reset/step, rollout, install, download, GPU job, training, heavy VLA import, OpenVLA-OFT execution, token access, or paper claim.

The bounded simulator render-smoke script is implemented in `scripts\59_bounded_simulator_render_smoke.ps1`. Current local result: it performed one tiny MuJoCo 64x64 offscreen render with `MUJOCO_GL=osmesa`; the rendered image shape was `[64, 64, 3]` and `image_mean` was nonzero. It did not create/reset/step LIBERO or RoboSuite environments, rollout, train, use GPU jobs, install packages, download assets, import heavy VLA models, execute OpenVLA-OFT, access tokens, or make paper claims.

The bounded simulator reset/step smoke script is implemented in `scripts\61_bounded_simulator_reset_step_smoke.ps1`. Current local result: it performed `mj_resetData`, `mj_forward`, and 3 `mj_step` calls on a tiny in-memory MuJoCo XML model through the selected WSL venv. It did not create LIBERO or RoboSuite environments, run rollouts, run policy inference, train, use GPU jobs, install packages, download assets, import heavy VLA models, execute OpenVLA-OFT, access tokens, or make paper claims.

The tiny diagnostic rollout planner is implemented in `scripts\62_plan_tiny_diagnostic_rollout.ps1`. Current local result: it reads the passed reset/step smoke report and reports a bounded max-5-task, one-episode, max-5-step planning envelope. It authorizes execution only through a separate task-local `ALLOW_TINY_ROLLOUT=1` run and still keeps `ready_for_rollout=false` for benchmark rollouts.

The bounded tiny diagnostic rollout runner is implemented in `scripts\63_bounded_tiny_diagnostic_rollout.ps1`. Current local result: with task-local `ALLOW_TINY_ROLLOUT=1`, it ran 5 toy MuJoCo diagnostic tasks, 1 episode each, 5 steps each, for 25 total steps through `/home/jiheon/.venvs/tca_map_sim/bin/python`. It did not create LIBERO or RoboSuite benchmark environments, run learned policy inference, train, use GPU jobs, install packages, download assets, import heavy VLA models, execute OpenVLA-OFT, access tokens, run multi-seed evaluation, or make benchmark/SOTA/paper-grade claims. Runtime reports are ignored by git:

```text
reports\bounded_tiny_diagnostic_rollout_report.json
reports\bounded_tiny_diagnostic_rollout_report.md
```

The bounded LIBERO/RoboSuite diagnostic rollout planner and runner are implemented in:

```text
scripts\64_plan_libero_robosuite_diagnostic_rollout.ps1
scripts\65_bounded_libero_robosuite_diagnostic_rollout.ps1
```

Current local result: with task-local `ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT=1`, the runner created one real LIBERO/RoboSuite environment from `libero_10`, reset it, stepped it 3 times with a zero action, read a finite 64x64 `agentview_image`, and closed the environment. It used the existing WSL venv at `/home/jiheon/.venvs/tca_map_sim`, the local official LIBERO source/data roots, and the RoboSuite checkout aligned to the official `v1.4.0` tag required by LIBERO. This did not run learned policy inference, train, use GPU, download assets during the diagnostic, execute OpenVLA-OFT, run multi-seed evaluation, or make benchmark/SOTA/paper-grade claims.

The learned-policy LIBERO rollout readiness planner is implemented in:

```text
scripts\66_plan_libero_policy_rollout_readiness.ps1
```

It is planning-only. It checks whether the WSL-only topology can host both the LIBERO/RoboSuite simulator and SmolVLA policy runtime from local assets. It does not load SmolVLA, run policy inference, train, use GPU, create simulator environments, rollout, download, execute OpenVLA-OFT, or make paper claims. If it reports `proceed`, the next safe branch is a separately gated tiny learned-policy rollout runner. If it reports `reduce_scope`, the next safe branch is WSL SmolVLA runtime setup/readiness in `/home/jiheon/.venvs/tca_map_sim`.

The WSL SmolVLA runtime setup planner and guard are implemented in:

```text
scripts\67_plan_wsl_smolvla_runtime_setup.ps1
scripts\68_setup_wsl_smolvla_runtime_deps.ps1
```

Current local result: after a green risk assessment, the WSL venv `/home/jiheon/.venvs/tca_map_sim` now has the lightweight SmolVLA runtime modules needed for WSL-only readiness:

```text
torch==2.10.0+cpu
torchvision==0.25.0+cpu
transformers==4.57.6
lerobot==0.4.4
safetensors==0.8.0
huggingface-hub==0.35.3
accelerate==1.14.0
num2words==0.5.14
draccus==0.10.0
datasets==4.8.5
imageio==2.37.3
imageio-ffmpeg==0.6.0
diffusers==0.35.2
pyserial==3.5
deepdiff==8.6.2
av==15.1.0
einops==0.8.2
```

The first task-local setup run reached the 1800 second timeout after venv package downloads/install activity. No residual pip/install process remained, and a follow-up module-spec probe reported all required modules present. A second task-local setup guard run detected `setup_required=false` and reported `setup_passed=true` without further installs or downloads. The later WSL single-action smoke found additional LeRobot import/runtime dependencies (`draccus`, `datasets`, `imageio`, `diffusers`, `pyserial`, `deepdiff`, `av`, and `einops`), which were installed venv-local after green risk assessments. This is runtime readiness only: no training, no rollout, no GPU job, no OpenVLA-OFT execution, no token access, and no paper claim.

The bounded WSL SmolVLA single-action smoke has now passed. It loaded the local SmolVLA policy in WSL, used CPU, ran one synthetic `select_action` call, produced a finite action with shape `[1, 6]`, and completed in about 20.26 seconds. It did not create a simulator environment, rollout, train, use GPU jobs, download assets during the smoke, execute OpenVLA-OFT, access tokens, or make paper claims.

WSL simulator compatibility adjustments made after green risk assessments:

```text
C:\assets\repos\robosuite -> official v1.4.0 checkout
/home/jiheon/.venvs/tca_map_sim: bddl==1.0.1, future==0.18.2, easydict==1.9,
matplotlib==3.5.3, numpy==1.22.4, cloudpickle==2.1.0, gym==0.25.2,
mujoco==2.3.7
```

These are local environment readiness changes only. They are not committed as assets, not paper evidence, and not a benchmark result. Learned-policy rollout and tiny benchmark rollout now proceed only through a separate green readiness/risk planner. Multi-seed rollout, OpenVLA-OFT execution, full fine-tuning, external upload, and unsupported paper-level claims remain stop gates.

## WSL Simulator Dependency Ladder Standing Approval

Codex has standing approval to continue through bounded WSL simulator dependency setup and simulator readiness progression after a green risk assessment. This covers WSL status inspection, WSL `python3`/`pip`/`venv` checks, minimal WSL Python packaging tools, the preferred venv `~/.venvs/tca_map_sim`, minimal import-readiness Python dependencies, bounded import smoke, bounded render smoke, bounded reset/step smoke, and bounded tiny rollout diagnostic.

Allowed minimal apt packages are `python3-pip`, `python3-venv`, `python3-dev` if needed, `build-essential` only if required for a Python package build, `git` if missing, and `curl` or `wget` only if needed for official setup checks.

Hard stops remain: sudo password or credentials, token/secret/API key/login, paid service or license click-through, CUDA driver/toolkit install, major graphics-stack changes, Windows driver changes, OpenVLA-OFT download/import/load/execution, full fine-tuning, training over 30 minutes, VRAM over 14GB, downloads beyond the approved budget, rollout beyond the current risk-assessed limits, unsupported benchmark/paper-grade claims, multi-seed experiments before a separate risk budget, external upload/submission/publishing, and deleting user files outside repo/cache cleanup.

The offline interface smoke gate inspects the acquired LIBERO demonstrations without model loading, training, simulator execution, rollout, OpenVLA-OFT, or paper claims. Its HDF5 report now records bounded samples instead of dumping every dataset path.

The consolidated local pilot and go/no-go summaries now include these LIBERO data gates when their runtime reports are present.

The required LoRA/QLoRA experiment-track policy is documented in `reports\lora_required_experiment_plan.md`:

```powershell
Get-Content reports\lora_required_experiment_plan.md
```

The required LoRA adapter construction plan is documented in `reports\lora_adapter_construction_plan.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\32_plan_lora_adapter_construction.ps1
```

The planning-only LoRA adapter construction check has passed. It validated the required LoRA and QLoRA configs, confirmed the adapter module allowlist, and did not download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims.

The planning-only LoRA tiny-smoke scaffold is documented in `reports\lora_tiny_smoke_scaffold.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\33_plan_lora_tiny_smoke.ps1
```

The scaffold has passed as a dry-run guard. It defines the required LoRA tiny-smoke envelope but does not construct adapters, train, import heavy VLA models, load models, infer, run GPU jobs, download assets, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims. Actual LoRA tiny-smoke execution remains gated by a separate bounded runner.

The required TCA-Map + LoRA comparison plan is documented in `reports\lora_comparison_plan.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\34_plan_lora_comparison.ps1
```

The comparison plan has passed as a planning-only guard. It fixes the required ActionMap + LoRA, TCA-Map + LoRA, and TCA-Map + LoRA + Distributional TCA-Select comparisons while separating head architecture gain, LoRA adaptation gain, and inference-time selection gain. It does not train, construct adapters, import heavy VLA models, load models, infer, run GPU jobs, download assets, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims.

The QLoRA feasibility check is documented in `reports\qlora_feasibility_check.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\35_check_qlora_feasibility.ps1
```

The check has passed as a check-only gate. It records whether QLoRA tooling is present without installing packages or changing CUDA/PyTorch. It keeps `safe_to_run_qlora_now=false` and does not train, construct adapters, import heavy VLA models, load models, infer, run GPU jobs, download assets, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims.

The LoRA/QLoRA-aware go/no-go generator is documented in `reports\go_no_go_status.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\31_generate_go_no_go_report.ps1
```

The updated go/no-go status is ready for bounded local pilot work and remains no-go only for paper-grade claims or tasks outside risk budget. The next meaningful work may include risk-assessed dataset readiness, simulator readiness/import-render smoke, bounded local pilot extension, or larger compute handoff planning. QLoRA package/tooling changes, real dataset setup, simulator rollout, and bounded training require risk assessment; OpenVLA-OFT execution and paper-grade claims remain blocked unless a separate policy exists.

The completed install risk boundary is documented in `reports\smolvla_runtime_install_request.md`. Future package upgrades, CUDA toolkit changes, or PyTorch changes require package/runtime risk assessment and remain blocked if they require system-wide changes.

The feature-cache interface contract is documented in `reports\feature_cache_interface_plan.md` and can be checked without SmolVLA imports:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\19_plan_feature_cache.ps1
```

The eval-only cached-feature smoke is documented in `reports\feature_cache_eval_smoke_plan.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\25_eval_feature_cache_smoke.ps1 -PrepareDummyCache
```

The tiny head-only pilot risk boundary is documented in `reports\tiny_head_only_pilot_plan.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\26_plan_tiny_head_only_pilot.ps1
```

The bounded tiny head-only smoke runner is documented in `reports\tiny_head_only_smoke.md`:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\29_tiny_head_only_smoke.ps1 -PrepareDummyCache
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

The bounded tiny LoRA smoke runner is documented in `reports\tiny_lora_smoke.md`:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\37_tiny_lora_smoke.ps1 -PrepareDummyCache
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

The runner trains only tiny NumPy LoRA adapter matrices over cached/dummy features. It covers `actionmap_lora`, `tca_map_lora`, and `tca_map_lora_distributional_select`, reports offline proxy metrics only, and does not download assets, use GPU, import heavy VLA models, load models, run model inference, rollout, execute simulators, execute OpenVLA-OFT, or make paper claims.

The tiny LoRA comparison report is documented in `reports\tiny_lora_comparison.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\38_compare_tiny_lora_pilot.ps1
```

The comparison reads `reports\tiny_lora_smoke_report.json` and reports ActionMap+LoRA vs TCA-Map+LoRA and TCA-Map+LoRA vs TCA-Map+LoRA+Distributional TCA-Select deltas using offline proxy metrics only. It does not train, download, use GPU, import heavy VLA models, load models, run inference, rollout, execute simulators, execute OpenVLA-OFT, or make paper claims.

The consolidated local pilot status report is documented in `reports\local_pilot_status.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\39_generate_local_pilot_status.ps1
```

The status generator reads existing bounded local runtime reports and summarizes what has passed. It is summary-only and does not download, train, use GPU, import heavy VLA models, load models, run inference, rollout, execute simulators, execute OpenVLA-OFT, or make paper claims. The next execution step should be chosen by risk assessment rather than by asking for routine approval.

## Current Risk-Assessed Autonomy Policy

The latest source of truth is the `Risk-assessed autonomous execution policy` in `reports\codex_delegation_manual.md`.

The previous approval-based hard-stop model is replaced for routine research-engineering work. Codex should not ask for permission merely because a task involves downloads, GPU, training, datasets, simulator readiness, or bounded local rollout. Codex must first write or print a risk assessment covering source, size, target path, disk free before/after estimate, runtime, RAM/VRAM, budget, dependency, license/token/payment status, decision, and reason.

If the risk assessment says `proceed`, Codex should continue autonomously. If it says `stop`, Codex should report the exact blocker and recommended next action.

Current default budgets:

- downloads: official/documented unambiguous source, no token/login/payment/license click-through, <=80GB single-task soft limit, keep >=100GB free disk, approved roots such as `C:\assets`, no asset/cache/data commits. Official LIBERO data from `yifengzhu-hf/LIBERO-datasets` is the only current exception and may use <=180GB only if >=250GB remains after acquisition,
- GPU: SmolVLA/local-pilot only, expected VRAM <=14GB, runtime <=30 minutes, batch size 1, timeout/stop condition, memory/runtime logged when measurable,
- training: SmolVLA-only, frozen backbone or LoRA/QLoRA adapter only, no full fine-tuning, max 300 local pilot steps after smaller smoke is stable, runtime <=30 minutes, VRAM <=14GB, batch size 1, smoke/offline proxy/local pilot labels only,
- real datasets: official/documented unambiguous source, no token/login/payment/license click-through, inside download/disk budget, no automatic rollout, prefer metadata-only or tiny subset first,
- simulator readiness: WSL2/Linux preferred. Minimal WSL Python packaging setup is standing-approved after risk assessment; import/render smoke remains <=10 minutes, no policy rollout or benchmark evaluation,
- bounded rollout: simulator import/render/reset-step smoke passed, task count <=5 for the first local benchmark rung, runtime <=30 minutes, no OpenVLA-OFT, no external service/token, no unsupported paper claim.

Codex must still stop before token/secret/API key access, paid service, license click-through, external upload/submission/publishing, deleting user files outside approved cache/repo cleanup, system-wide CUDA/PyTorch/driver changes, credentialed/system-driver/license-gated system setup, OpenVLA-OFT execution, or unsupported paper-level empirical claims. Paper-grade candidate reports are allowed only from verified experiment outputs with explicit evidence labels. Minimal WSL Python packaging setup is standing-approved after the WSL simulator dependency ladder risk assessment; a sudo password prompt remains a hard stop.

Next autonomous direction: `scripts\66_plan_libero_policy_rollout_readiness.ps1` now reports the WSL-only simulator plus SmolVLA runtime topology as green, and `scripts\70_bounded_wsl_smolvla_single_action_smoke.ps1` has passed one CPU synthetic action. Create a separately gated tiny learned-policy LIBERO rollout runner. Stop before multi-seed rollout, OpenVLA-OFT, full fine-tuning, external upload, or unsupported paper-level claims.

The current bounded cached-feature local pilot extension is documented in `reports\bounded_local_pilot_extension.md` and runs through:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\44_bounded_local_pilot_extension.ps1 -PrepareDummyCache
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

This remains offline proxy/interface evidence only. It does not authorize real dataset training, simulator execution, rollouts, OpenVLA-OFT, or paper claims.

The consolidated local pilot status and go/no-go generators now include the bounded extension report when present. They remain summary-only and refuse execution gates.

The consolidated hard-stop status is documented in `reports\hard_stop_status.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\27_summarize_hard_stop_status.ps1
```

The go/no-go status summary is documented in `reports\go_no_go_status.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\31_generate_go_no_go_report.ps1
```

## Tiny Learned-Policy LIBERO Rollout Result

Current local bounded result: passed as diagnostic evidence only.

`scripts\71_plan_tiny_learned_policy_rollout.ps1` reported `decision=proceed`, and `scripts\72_bounded_tiny_learned_policy_rollout.ps1` passed with task-local `ALLOW_TINY_LEARNED_POLICY_ROLLOUT=1`.

Observed local result:

- suite: `libero_10`,
- task count: 1,
- max steps per task: 3,
- completed steps: 3,
- SmolVLA policy calls: 3,
- policy action shape: `[1, 6]`,
- LIBERO environment action dimension: 7,
- diagnostic success check: `false`,
- reward sum: `0.0`,
- inner runtime: about 30.6 seconds,
- no downloads, no training, no GPU job, no OpenVLA-OFT, no token access, no multi-seed evaluation, and no paper claim.

This clears the first learned-policy LIBERO diagnostic topology only. It is not standard success, not benchmark success, not counterfactual robustness evidence, and not paper-grade evidence.

Next autonomous direction: create a tiny benchmark-metric diagnostic report and, if green, progress to a bounded small learned-policy rollout matrix with explicit evidence labels and no multi-seed or paper-grade claim.

## Tiny Learned-Policy Metric Summary Result

Current local report-only result: passed.

`scripts\73_generate_tiny_learned_policy_metric_summary.ps1` summarized the existing tiny learned-policy rollout report without loading models, running inference, creating simulator environments, rolling out, training, using GPU jobs, downloading, executing OpenVLA-OFT, accessing tokens, or making paper claims.

Observed diagnostic metrics:

- source rollout passed: true,
- tasks completed: 1,
- total steps: 3,
- policy calls: 3,
- diagnostic success count: 0,
- diagnostic success rate: 0.0,
- reward sum: 0.0,
- mean policy latency from the recorded final step: about 0.157 seconds,
- policy action shape: `[1, 6]`,
- environment action dimension: 7,
- failure mode: `diagnostic_success_check_false`.

Interpretation: the learned-policy simulator-control topology is working, but the current one-task, three-step diagnostic did not solve the task. This remains diagnostic/local-pilot evidence only.

Next autonomous direction: create a bounded small learned-policy rollout matrix planner with explicit task count, step count, runtime, evidence-label, and no-paper-claim guards.

## Bounded Learned-Policy Rollout Matrix Planner Result

Current local planning result: `reduce_scope`.

`scripts\74_plan_bounded_learned_policy_rollout_matrix.ps1` read the tiny learned-policy metric summary and did not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

Decision details:

- source rollout passed: true,
- source total steps: 3,
- source policy calls: 3,
- diagnostic success rate: 0.0,
- reward sum: 0.0,
- ready for reduced-scope learned-policy runner: true,
- ready for bounded small multi-task matrix runner: false,
- recommended task count: 1,
- recommended steps per task: 10,
- evidence label: diagnostic/local pilot only.

Interpretation: a broader matrix is premature because the first learned-policy diagnostic did not achieve task success. The next safe rung is a separately gated one-task longer diagnostic runner.

## Bounded Reduced-Scope Learned-Policy Rollout Result

Current local bounded result: passed as diagnostic execution evidence only.

`scripts\75_bounded_reduced_scope_learned_policy_rollout.ps1` ran under task-local `ALLOW_BOUNDED_LEARNED_POLICY_MATRIX=1`, used the WSL simulator topology, loaded local SmolVLA on CPU, created one real `libero_10` RoboSuite/LIBERO environment, made 10 policy calls, and stepped the environment 10 times.

Observed local result:

- completed steps: 10,
- policy calls: 10,
- policy action shape: `[1, 6]`,
- environment action dimension: 7,
- diagnostic success check: `false`,
- reward sum: `0.0`,
- inner runtime: about 35.8 seconds,
- no downloads, no installs, no training, no GPU job, no OpenVLA-OFT, no multi-seed evaluation, no token access, and no paper claim.

Interpretation: the reduced-scope longer diagnostic is stable but still not successful on the selected task. The next safe rung is a report-only reduced-scope metric summary, followed by a research decision about whether to tune interface/action scaling, inspect action semantics, or continue with a tiny controlled baseline comparison.

## Reduced-Scope Rollout Metric Summary Result

Current local report-only result: passed.

`scripts\76_generate_reduced_scope_rollout_metric_summary.ps1` summarized the one-task, 10-step reduced-scope learned-policy rollout without loading models, running inference, creating simulator environments, rolling out, training, using GPU jobs, downloading, executing OpenVLA-OFT, accessing tokens, or making paper claims.

Observed diagnostic metrics:

- source runner passed: true,
- tasks completed: 1,
- total steps: 10,
- policy calls: 10,
- diagnostic success count: 0,
- diagnostic success rate: 0.0,
- reward sum: 0.0,
- mean policy latency from the recorded final step: about 0.147 seconds,
- action max absolute value: about 0.793,
- action L2 norm: about 1.222,
- gripper component: 0.0,
- failure mode: `diagnostic_success_check_false`.

Interpretation: the policy emits finite, nontrivial continuous actions, but the action/observation/control interface likely needs diagnosis before any larger rollout matrix is useful.

## Action-Interface Diagnostic Planner Result

Current local planning result: `proceed`.

`scripts\77_plan_action_interface_diagnostics.ps1` read the reduced-scope rollout metric summary and local SmolVLA `config.json` metadata without downloading, installing, loading models, running inference, creating simulator environments, rolling out, training, using GPU jobs, executing OpenVLA-OFT, accessing tokens, or making paper claims.

Observed signals:

- diagnostic success rate: 0.0,
- reward sum: 0.0,
- policy action dimension: 6,
- environment action dimension: 7,
- gripper component: 0.0,
- action max absolute value: about 0.793,
- action L2 norm: about 1.222.

Priority diagnosis:

- high: action dimension and gripper mapping,
- high: action normalization and scale,
- high: observation state mapping,
- medium: camera mapping,
- medium: language prompt mapping,
- medium: zero-action versus SmolVLA-action comparison.

Next autonomous direction: create a bounded action-interface audit that reads metadata and existing reports only, then a zero-action versus SmolVLA-action diagnostic comparison if the audit remains green.

## Action-Interface Metadata Audit Result

Current local audit result: `proceed`.

`scripts\78_audit_action_interface_metadata.ps1` read existing reports plus SmolVLA config/preprocessor/postprocessor metadata and the local rollout bridge source without downloading, installing, loading models, running inference, creating simulator environments, rolling out, training, using GPU jobs, executing OpenVLA-OFT, accessing tokens, or making paper claims.

High-priority findings:

- action dimension mismatch remains: policy action dim 6 versus env action dim 7, now through an explicit adapter,
- the current explicit gripper strategy is zero-hold and needs validation,
- nontrivial continuous actions still yield diagnostic success rate 0.0 and reward 0.0.

Medium-priority finding:

- camera feature naming mismatch between config and preprocessor metadata.

Next autonomous direction: run adapter-strategy/action-scale diagnostics before any rollout scaling.

## Zero-Action Versus SmolVLA-Action Diagnostic Comparison

Current planned branch: compare the existing zero-action LIBERO/RoboSuite diagnostic rollout against the existing reduced-scope SmolVLA learned-policy diagnostic rollout using reports only.

The comparison is summary-only. It must not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

Expected interpretation: if zero-action simulator plumbing passed and SmolVLA emits nontrivial actions but does not improve success or reward, the next autonomous step is an explicit adapter patch when adapter metadata is absent, or adapter-strategy/action-scale diagnosis when explicit adapter metadata is present.

Current local result: `scripts\79_compare_zero_action_policy_diagnostic.ps1` passed as summary-only. It compared the existing zero-action diagnostic against the latest adapter-wired SmolVLA learned-policy diagnostic on the same `libero_10` task. Both had diagnostic success `false` and reward sum `0.0`; SmolVLA actions were nontrivial with max absolute action about `0.793093`, and explicit adapter metadata was present. The next autonomous direction is adapter-strategy/action-scale diagnosis.

## Action/State Adapter Patch Plan

Current planned branch: create a planning-only adapter patch specification before modifying rollout behavior.

The plan must require explicit action, state, and camera adapters. It must keep rollout scaling blocked until pure adapter helpers and unit tests pass, followed by single-sample/interface checks.

Current local result: `scripts\80_plan_action_state_adapter_patch.ps1` passed as planning-only. It requires action, state, and camera alias adapters, sets `ready_for_pure_adapter_implementation=true`, and keeps `ready_for_rollout_scaling=false`.

## Pure Action/State/Image Adapter Helpers

Current planned branch: add pure helper functions for action, state, and image alias adaptation. This step must be unit-test-only and must not wire into rollout execution yet.

Current local result: pure adapter helper tests passed. Rollout wiring remains intentionally untouched.

## Single-Sample Adapter Metadata Wiring

Current planned branch: wire pure adapter helpers into the synthetic single-sample interface path so reports include state, image alias, and action adapter metadata without simulator execution or rollout.

Current local result: bounded synthetic single-sample smoke passed with adapter metadata recorded. The report includes explicit state adapter metadata, image alias source keys, and a 6D-to-7D action adapter using the named gripper-zero diagnostic strategy. Simulator execution and rollouts remain false.

## Rollout Bridge Adapter Wiring Plan

Current planned branch: add a planning-only gate for wiring pure adapters into the learned-policy rollout bridge. This planner must not run rollouts, simulators, model loading, inference, downloads, training, or OpenVLA-OFT.

Current local result: `scripts\81_plan_rollout_bridge_adapter_wiring.ps1` passed. It sets `ready_for_rollout_bridge_adapter_wiring=true` and `ready_for_rollout_execution=false`.

## Rollout Bridge Adapter Wiring

Current local result: pure action, state, and image adapters are wired into the learned-policy rollout bridge in code and covered by unit tests. The bridge now records adapter metadata in task summaries.

This remains interface-plumbing validation only. It is not benchmark rollout evidence, standard success, counterfactual robustness evidence, or paper-grade evidence. The next autonomous rung is a separate bounded diagnostic rollout gate that compares explicit-adapter behavior against prior zero-action and legacy learned-policy diagnostics.

## Adapter-Wired Learned-Policy Diagnostic

Current local result: the bounded one-task, 10-step learned-policy diagnostic reran with explicit adapter metadata and passed as an execution wrapper. It did not produce task success or reward.

Observed diagnostic metrics:

- diagnostic success rate: 0.0,
- reward sum: 0.0,
- policy calls: 10,
- last action max abs: about 0.793,
- gripper component: 0.0,
- action adapter strategy: `policy_6d_delta_pose_plus_gripper_zero_hold`,
- state adapter: `diagnostic_eef_pos_quat_xyz_6d_state_adapter`,
- image sources: `agentview_image` and `robot0_eye_in_hand_image`,
- implicit action padding/truncation: false,
- state padding/truncation: false,
- zero-image fallback: false.

Interpretation: explicit adapter wiring is working as metadata-visible plumbing, but the selected diagnostic task still fails. The next safe task is adapter-strategy/action-scale diagnosis before any rollout scaling.

## Adapter-Strategy/Action-Scale Diagnostics Plan

Current local planning target: `scripts\82_plan_adapter_strategy_action_scale_diagnostics.ps1`.

This planner reads the adapter-aware audit, reduced-scope metric summary, zero-action comparison, and rollout bridge source. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

The planned first diagnostic runner is one task, at most 10 steps per variant, and at most three gripper-strategy variants:

- `policy_6d_delta_pose_plus_gripper_zero_hold`,
- `policy_6d_delta_pose_plus_gripper_open`,
- `policy_6d_delta_pose_plus_gripper_close`.

Rollout scaling remains blocked until the strategy diagnostics explain or improve the zero-reward behavior.

## Adapter-Strategy Diagnostic Runner Result

Current local bounded result: passed as diagnostic execution evidence only.

`scripts\83_bounded_adapter_strategy_diagnostic.ps1` ran under task-local `ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC=1` and executed three one-task, 10-step gripper-strategy variants:

- `policy_6d_delta_pose_plus_gripper_zero_hold`,
- `policy_6d_delta_pose_plus_gripper_open`,
- `policy_6d_delta_pose_plus_gripper_close`.

Observed local result:

- variants completed: 3,
- wrapper/execution passed for all variants,
- diagnostic success rate: 0.0 for all variants,
- reward sum: 0.0 for all variants,
- observed gripper components: `0.0`, `1.0`, and `-1.0`,
- no downloads, no installs, no training, no GPU job, no OpenVLA-OFT, no multi-seed evaluation, no token access, and no paper claim.

Interpretation: explicit gripper-strategy selection is now execution-tested and metadata-visible, but gripper strategy alone does not explain the zero-reward behavior on the selected diagnostic task. Rollout scaling remains blocked. The next safe rung is a bounded action-scale, prompt-format, camera-source, or state-sufficiency diagnostic before any broader learned-policy rollout matrix.

## Action-Scale Diagnostic Runner Result

Current local bounded result: passed as diagnostic execution evidence only.

`scripts\84_plan_action_scale_diagnostic.ps1` reported `decision=proceed`, and `scripts\85_bounded_action_scale_diagnostic.ps1` ran under task-local `ALLOW_ACTION_SCALE_DIAGNOSTIC=1`.

Observed local result:

- action adapter strategy: `policy_6d_delta_pose_plus_gripper_zero_hold`,
- action scales tested: `0.25`, `0.5`, and `1.0`,
- variants completed: 3,
- wrapper/execution passed for all variants,
- diagnostic success rate: 0.0 for all variants,
- reward sum: 0.0 for all variants,
- last action max absolute values scaled as expected: about `0.198`, `0.397`, and `0.793`,
- no downloads, no installs, no training, no GPU job, no OpenVLA-OFT, no multi-seed evaluation, no token access, and no paper claim.

Interpretation: action-scale wiring is working, but action magnitude alone does not explain the zero-reward behavior on the selected diagnostic task. Rollout scaling remains blocked. The next safe rung is a bounded prompt-format, camera-source, or state-sufficiency diagnostic before any broader learned-policy rollout matrix.

## Prompt-Format Diagnostic Runner Result

Current local bounded result: passed as diagnostic execution evidence only.

`scripts\86_plan_prompt_format_diagnostic.ps1` reported `decision=proceed`, and `scripts\87_bounded_prompt_format_diagnostic.ps1` ran under task-local `ALLOW_PROMPT_FORMAT_DIAGNOSTIC=1`.

Observed local result:

- prompt strategies tested: `stem_spaces`, `bddl_language`, and `bddl_language_period`,
- observed BDDL language: `turn on the stove and put the moka pot on it`,
- variants completed: 3,
- wrapper/execution passed for all variants,
- diagnostic success rate: 0.0 for all variants,
- reward sum: 0.0 for all variants,
- prompt changes produced different continuous action previews,
- no downloads, no installs, no training, no GPU job, no OpenVLA-OFT, no multi-seed evaluation, no token access, and no paper claim.

Interpretation: BDDL language prompt parsing is wired and cleaner than stem-derived prompts, but prompt format alone does not explain the zero-reward behavior on the selected diagnostic task. Rollout scaling remains blocked. The next safe rung is a bounded camera-source or state-sufficiency diagnostic before any broader learned-policy rollout matrix.
