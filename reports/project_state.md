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
4fbbec1 or newer
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
- local paper-grade runner and planning scripts,
- Cursor safe local runner,
- SmolVLA asset prep,
- SmolVLA readiness semantics split,
- SmolVLA download plan guard,
- Windows Bash shim handling for Bash-specific tests,
- manual SmolVLA acquisition checklist,
- Codex delegation manual and project state files,
- SmolVLA load-only adapter smoke planning guard,
- SmolVLA load-only execution scaffold,
- SmolVLA runtime dependency checker and install approval plan,
- feature-cache interface contract and dummy cache planner,
- eval-only cached-feature smoke for the head/metric interface,
- tiny head-only pilot approval planner,
- bounded tiny head-only smoke runner,
- hard-stop approval status summary,
- go/no-go status summary,
- explicitly approved SmolVLA runtime package install,
- SmolVLA autonomous pilot standing approval policy,
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

The original acquisition approvals were limited to SmolVLA checkpoint acquisition from `lerobot/smolvla_base` and tokenizer/processor/config dependency acquisition from `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. They did not approve GPU jobs, model inference, training, rollouts, OpenVLA-OFT execution/download, dataset downloads, token/secret access, or committing checkpoint/cache files. A later standing approval now permits bounded SmolVLA load-only heavy import/model construction and tiny smoke steps inside the autonomous pilot budget only.

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

The external tokenizer/processor dependency is now detected under `C:\assets\hf_home`. The current gate is Case C from the self-check gate policy: readiness is true, the load-only adapter smoke plan is prepared, and the bounded execution scaffold exists. Runtime packages are now installed, and bounded load-only model construction is standing-approved inside the SmolVLA autonomous pilot budget.

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

The runtime install used the explicit package-install approval only. The standing-approved autonomous pilot policy now authorizes bounded SmolVLA load-only heavy import/model construction, single-sample interface smoke, tiny feature-cache/interface validation, and tiny head-only training smoke within budget. It still does not authorize rollouts, simulator execution, OpenVLA-OFT, token access, paper-grade claims, dataset downloads, major CUDA/PyTorch changes, unplanned large package installs, or jobs over the runtime/VRAM budget.

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

The bounded tiny head-only smoke has passed on cached/dummy features. It trained tiny CPU NumPy ActionMap and TCA-Map heads for 16 steps, overrode the 1000-step config value to the smoke cap, and did not import SmolVLA, load a model, run VLA inference, use GPU, rollout, execute OpenVLA-OFT, download assets, or make paper claims.

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

Runtime install approval planner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\18_plan_smolvla_runtime_install.ps1
```

This planner is check-only. It writes `reports\smolvla_runtime_install_plan_report.json`, refuses dangerous gates such as `ALLOW_DOWNLOADS=1` or `ALLOW_HEAVY_IMPORT=1`, and does not install packages.

Other missing assets currently expected:

- OpenVLA-OFT checkpoint,
- LIBERO source checkout,
- LIBERO data root,
- RoboSuite root.

## Current Gate

The next real-adapter step is the standing-approved single-sample SmolVLA interface smoke with synthetic or dummy inputs. It is not a paper result and must remain within the bounded autonomous pilot policy.

```text
C:\assets\checkpoints\smolvla
```

Ready file groups:

- `config.json`,
- external tokenizer/processor/config dependency files,
- weights file.

Codex should not ask whether these files were placed, whether runtime packages are installed, whether load-only smoke passed, or whether to run the next bounded interface smoke. It should run the readiness checkers and continue autonomously through the SmolVLA pilot path. It must still stop before inference beyond a single dummy/interface smoke, rollout, simulator execution, real benchmark evaluation, OpenVLA-OFT, token access, paper-grade claims, jobs over 30 minutes, more than 14GB VRAM, major CUDA/PyTorch changes, or unplanned large package installs.

Planning command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\15_plan_smolvla_load_only_smoke.ps1
```

Execution scaffold command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\16_smolvla_load_only_smoke.ps1
```

This writes ignored runtime output to:

```text
reports\smolvla_load_only_smoke_plan_report.json
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

Codex should run these commands itself when routine state is needed. The user should only be asked at dangerous gates.

## Safety Gates

Forbidden until explicitly approved as true hard-stop gates:

- OpenVLA-OFT download/import/load/execution,
- LIBERO/RoboSuite/RoboCasa/dataset download,
- rollouts,
- simulator execution,
- real benchmark evaluation,
- training longer than 15 minutes or more than 100 steps,
- any job expected to exceed 30 minutes,
- using more than 14GB VRAM,
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

Codex should self-check current state. Since config/tokenizer dependency/weights are present, adapter-smoke readiness is true, runtime dependencies are installed, bounded load-only smoke passed, single-sample interface smoke passed, dummy feature-cache/interface validation passed, bounded tiny head-only smoke passed, and the go/no-go summary exists, the next safe non-heavy task is LoRA/QLoRA required-track planning. No larger experimental stage is safe without explicit approval for exactly one true next gate. Do not cross rollout, simulator, real benchmark, token, OpenVLA-OFT, major CUDA/PyTorch, unplanned large package, >14GB VRAM, >30 minute, or paper-claim gates without explicit approval.

The required LoRA/QLoRA experiment-track policy is documented in `reports\lora_required_experiment_plan.md`:

```powershell
Get-Content reports\lora_required_experiment_plan.md
```

The completed install approval boundary is documented in `reports\smolvla_runtime_install_request.md`. Future package upgrades, CUDA toolkit changes, or PyTorch changes remain separate hard-stop gates.

The feature-cache interface contract is documented in `reports\feature_cache_interface_plan.md` and can be checked without SmolVLA imports:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\19_plan_feature_cache.ps1
```

The eval-only cached-feature smoke is documented in `reports\feature_cache_eval_smoke_plan.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\25_eval_feature_cache_smoke.ps1 -PrepareDummyCache
```

The tiny head-only pilot approval boundary is documented in `reports\tiny_head_only_pilot_plan.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\26_plan_tiny_head_only_pilot.ps1
```

The bounded tiny head-only smoke runner is documented in `reports\tiny_head_only_smoke.md`:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\29_tiny_head_only_smoke.ps1 -PrepareDummyCache
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

The consolidated hard-stop status is documented in `reports\hard_stop_status.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\27_summarize_hard_stop_status.ps1
```

The go/no-go status summary is documented in `reports\go_no_go_status.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\31_generate_go_no_go_report.ps1
```
