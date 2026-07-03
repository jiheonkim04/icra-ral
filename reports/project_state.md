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
- LoRA adapter construction planning scaffold,
- LoRA tiny-smoke scaffold,
- TCA-Map + LoRA comparison plan,
- QLoRA feasibility check,
- LoRA/QLoRA go/no-go status update,
- head-only ActionMap vs TCA-Map tiny comparison report,
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

The bounded tiny head-only smoke has passed on cached/dummy features. It trained tiny CPU NumPy ActionMap and TCA-Map heads for 16 steps, overrode the 1000-step config value to the smoke cap, and did not import SmolVLA, load a model, run VLA inference, use GPU, rollout, execute OpenVLA-OFT, download assets, or make paper claims.

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

Other missing assets currently expected:

- OpenVLA-OFT checkpoint,
- LIBERO source checkout,
- LIBERO data root,
- RoboSuite root.

## Current Gate

The next real-adapter step is the risk-assessed single-sample SmolVLA interface smoke with synthetic or dummy inputs. It is not a paper result and must remain within the bounded autonomous pilot policy.

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

Codex should self-check current state. Since config/tokenizer dependency/weights are present, adapter-smoke readiness is true, runtime dependencies are installed, bounded load-only smoke passed, single-sample interface smoke passed, dummy feature-cache/interface validation passed, bounded tiny head-only smoke passed, and the go/no-go summary exists, the next safe path is risk-assessed local SmolVLA work. Bounded local pilots may proceed autonomously if risk assessment stays within max 300 local pilot steps after smaller smoke is stable, max 200 samples, max 30 minutes, max 14GB VRAM, batch size 1, no OpenVLA-OFT, no full fine-tuning, and no paper claim. Downloads, datasets, simulator readiness, GPU work, and bounded rollout should be assessed rather than treated as approval-only gates.

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

- downloads: official/documented unambiguous source, no token/login/payment/license click-through, <=80GB single-task soft limit, keep >=100GB free disk, approved roots such as `C:\assets`, no asset/cache/data commits,
- GPU: SmolVLA/local-pilot only, expected VRAM <=14GB, runtime <=30 minutes, batch size 1, timeout/stop condition, memory/runtime logged when measurable,
- training: SmolVLA-only, frozen backbone or LoRA/QLoRA adapter only, no full fine-tuning, max 300 local pilot steps after smaller smoke is stable, runtime <=30 minutes, VRAM <=14GB, batch size 1, smoke/offline proxy/local pilot labels only,
- real datasets: official/documented unambiguous source, no token/login/payment/license click-through, inside download/disk budget, no automatic rollout, prefer metadata-only or tiny subset first,
- simulator readiness: already installed locally, no large install/download, runtime <=10 minutes, no policy rollout or benchmark evaluation,
- bounded rollout: simulator import/render smoke passed, task count <=5, runtime <=30 minutes, no OpenVLA-OFT, no external service/token, no paper claim.

Codex must still stop before token/secret/API key access, paid service, license click-through, external upload/submission/publishing, deleting user files outside approved cache/repo cleanup, system-wide CUDA/PyTorch/driver changes, admin/system-level installers, OpenVLA-OFT execution, or paper-level empirical claims.

Next autonomous direction: run risk assessment for the next concrete stage instead of asking for approval. The current next concrete stage is LIBERO/LIBERO-CF-style dataset readiness/tiny subset planning via `scripts\42_plan_libero_dataset_risk.ps1`; no download, rollout, simulator execution, training, or paper claim is authorized by that planner.

The consolidated hard-stop status is documented in `reports\hard_stop_status.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\27_summarize_hard_stop_status.ps1
```

The go/no-go status summary is documented in `reports\go_no_go_status.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\31_generate_go_no_go_report.ps1
```
