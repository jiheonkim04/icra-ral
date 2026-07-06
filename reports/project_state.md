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
f337304 or newer
```

Use explicit Python for validation:

```text
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe
```

## Current Autopilot Policy

Autopilot is bounded per execution. Codex must not run unbounded end-to-end research loops in a single execution.

Current rule:

- at most one major research milestone per execution,
- stop before commit if more than 50 files or more than 5,000 changed lines would be included,
- before every merge, report changed-file count, line diff count, whether training happened, whether rollout happened, whether loss was computed, whether the work is only planning/scaffolding, validation results, and merge justification,
- stop if a task runs longer than 2 hours without actual training or rollout progress,
- do not keep expanding planners indefinitely when no loss, metric, rollout result, or concrete validation result is being produced.

Canonical policy document:

```text
reports/autopilot_bounded_execution_policy.md
```

## Current Research Integrity Policy

Before any confirmatory ActionMap vs TCA-Map, TCA-Select, LoRA, or QLoRA
comparison, use:

```text
reports/research_integrity_evaluation_policy.md
```

This policy fixes the primary metrics, baseline list, ablation list,
split/sample policy, tuning budget, and kill/pivot criteria before results are
inspected. The objective is a trustworthy conclusion, not a positive result.

If ActionMap + LoRA or ActionMap + counterfactual augmentation matches TCA-Map,
the novelty is weak under that evidence level. If TCA-Select adds no measurable
gain, report it. If offline gains disappear in rollout, report it. If TCA-Map
fails, produce a kill/pivot report.

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

The first dependency acquisition retained tokenizer/processor/config files only. A later separately risk-assessed VLM required-file acquisition added the root `model.safetensors` under the same `HF_HOME` dependency directory for bounded `load_vlm_weights=true` diagnostics.

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

## Latest Diagnostic State-Sufficiency Result

The bounded learned-policy diagnostic ladder has now executed the state-sufficiency rung with `ALLOW_STATE_SUFFICIENCY_DIAGNOSTIC=1` set only for that task.

Summary:

- branch: `codex/state-sufficiency-diagnostic-runner`,
- scripts: `scripts\90_plan_state_sufficiency_diagnostic.ps1` and `scripts\91_bounded_state_sufficiency_diagnostic.ps1`,
- variants: `eef_pos_quat_first3`, `eef_pos_quat_last3`, and `eef_pos_zero_rot`,
- task scope: one `libero_10` task, at most 10 steps per variant,
- execution scope: WSL CPU learned-policy diagnostic only,
- downloads: false,
- training: false,
- GPU jobs: false,
- OpenVLA-OFT: false,
- benchmark/paper-grade claims: false.

Result:

- all three variants passed wrapper/execution,
- all three variants produced diagnostic success rate `0.0`,
- all three variants produced reward sum `0.0`,
- action previews changed across state strategies,
- rollout scaling remains blocked.

Interpretation: the local SmolVLA/LIBERO bridge can run bounded diagnostic rollouts and record explicit state metadata, but gripper strategy, action scale, prompt format, camera source, and state-vector variants have not produced a positive reward/success signal. The next safe research-engineering step is diagnostic synthesis and compatibility/no-go analysis, not broader rollout scaling.

## Latest Learned-Policy Diagnostic Synthesis

The report-only learned-policy diagnostic synthesis is now generated by `scripts\92_generate_learned_policy_diagnostic_synthesis.ps1`.

Current synthesis result:

- decision: `no_go_rollout_scaling`,
- diagnostic ladder complete: true,
- positive diagnostic signal found: false,
- rollout scaling ready: false,
- paper-grade claim ready: false.

The synthesis covers zero-action comparison, adapter strategy, action scale, prompt format, camera source, and state sufficiency. All wrapper/execution diagnostics passed where applicable, but every reward/success proxy stayed at `0.0` and every source report keeps `ready_for_rollout_scaling=false`.

Current autonomous next step: create a bounded environment-policy compatibility audit focused on task/checkpoint alignment, action convention, and observation convention. Do not scale learned-policy rollouts from the current evidence.

## Latest Environment-Policy Compatibility Audit

The report-only environment-policy compatibility audit is now generated by `scripts\93_audit_environment_policy_compatibility.ps1`.

Current audit result:

- decision: `no_go_rollout_scaling`,
- high-severity issues: 4,
- rollout scaling ready: false,
- paper-grade claim ready: false.

The audit flags task/checkpoint alignment, `load_vlm_weights=false` diagnostic policy loading, 6D policy action versus 7D environment action convention, and repeated zero-reward diagnostic results as high-severity blockers. Observation convention remains a medium-severity blocker.

Current autonomous next step: create a bounded offline LIBERO demonstration interface audit that reads one local HDF5 file and compares action dimensions/ranges, observation keys, camera shapes, and language/task alignment against the SmolVLA config. Do not load models, run simulator rollout, train, or scale learned-policy rollout for that audit.

## Latest LIBERO HDF5 Interface Audit

The report-only LIBERO HDF5 interface audit is now generated by `scripts\94_audit_libero_hdf5_interface.ps1`.

Current audit result:

- decision: `no_go_rollout_scaling`,
- high-severity issues: 2,
- rollout scaling ready: false,
- paper-grade claim ready: false.

Observed on the first local `libero_10` HDF5 demo:

- demonstration actions are 7D,
- SmolVLA policy action config is 6D,
- `obs/ee_states` is 6D and matches the policy state dimension,
- HDF5 exposes two RGB camera streams while the policy config expects three image inputs,
- HDF5 images are 128x128 while the policy config expects 256x256.

Current autonomous next step: create a report-only offline adapter reproduction check that builds SmolVLA-compatible state/image/action adapter inputs from the first HDF5 timestep and compares dimensions/ranges without model loading or rollout.

## Latest Offline Adapter Reproduction Check

The report-only offline adapter reproduction check is now generated by `scripts\95_check_offline_adapter_reproduction.ps1`.

Current check result:

- decision: `no_go_rollout_scaling`,
- best first-action gripper adapter strategy: `policy_6d_delta_pose_plus_gripper_close`,
- zero-hold first-action L1 mismatch: about `0.142857`,
- close first-action L1 mismatch: `0.0`,
- rollout scaling ready: false,
- paper-grade claim ready: false.

The first local LIBERO demonstration action has gripper value `-1.0`, so the current zero-hold gripper default is not demonstration-informed for that trajectory. HDF5 `obs/ee_pos + obs/ee_ori` exactly reproduces `obs/ee_states`, which supports using `ee_states` as the offline 6D state reference.

Current autonomous next step: plan a bounded one-task gripper-close compatibility diagnostic only if it remains inside the tiny diagnostic budget. Keep rollout scaling and paper-grade claims blocked.

## Gripper-Close Compatibility Diagnostic Plan

The planning-only gripper-close compatibility gate is now defined by `scripts\96_plan_gripper_close_compat_diagnostic.ps1`.

The planner reads the offline adapter reproduction report, a previous adapter-strategy diagnostic report if present, and the rollout bridge source. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

Expected interpretation:

- `decision=proceed`: offline HDF5 evidence supports a separately gated one-task gripper-close diagnostic and no equivalent zero-signal close diagnostic has already been found.
- `decision=reduce_scope`: gripper-close remains an offline compatibility clue, but an equivalent close-strategy rollout diagnostic already produced zero success and zero reward; do not rerun the same variant.
- `decision=stop`: required report/source prerequisites are missing or unsafe gates are set.

Rollout scaling, multi-seed evaluation, paper-grade claims, and OpenVLA-OFT remain blocked by this planning result.

## HDF5-to-Rollout Alignment Audit

The report-only HDF5-to-rollout alignment audit is defined by `scripts\97_audit_hdf5_rollout_alignment.ps1`.

It checks whether the offline LIBERO HDF5 demonstration and previous close-strategy rollout refer to the same task, whether the HDF5 demonstration exposes `init_state`/`states`, and whether the rollout bridge appears to set the HDF5 initial state or only calls `env.reset()`. It reads local reports, one local HDF5 file, and source text only.

Expected interpretation:

- `decision=reduce_scope`: the task appears aligned, but the HDF5 initial-state/replay convention is not established; plan a bounded HDF5 initial-state or first-action replay diagnostic before another learned-policy rollout.
- `decision=stop`: report inputs are missing or the HDF5 task and rollout task differ.
- `decision=proceed`: no report-only alignment blocker was found, but rollout scaling still needs a positive diagnostic signal.

This audit does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

## HDF5 Initial-State Replay Plan

The planning-only HDF5 initial-state replay gate is defined by `scripts\98_plan_hdf5_initial_state_replay.ps1`.

It checks whether the HDF5 alignment audit authorized a replay plan, whether the selected HDF5 demo contains `init_state`, `states`, 7D actions, and a model XML attribute, and whether local LIBERO/RoboSuite source code exposes `set_init_state` or flattened-state replay helpers.

If it reports `decision=proceed`, the next safe task is a separately gated one-demo replay runner. The first runner should set the HDF5 initial state if supported and replay only the first demonstration action. It must not load SmolVLA, perform learned-policy inference, train, use GPU jobs, download assets, execute OpenVLA-OFT, run multi-seed evaluation, or make paper claims.

## Bounded HDF5 Initial-State Replay Runner

The bounded one-demo replay runner is defined by `scripts\100_bounded_hdf5_initial_state_replay.ps1`.

It requires task-local `ALLOW_HDF5_REPLAY_DIAGNOSTIC=1`, reruns the HDF5 replay planner without the gate, maps the local HDF5 demo into WSL, creates a single LIBERO/RoboSuite environment, sets the HDF5 demonstration initial state if supported, and replays only the first demonstration action.

This runner is simulator/data compatibility evidence only. It performs no learned-policy loading or inference, no training, no GPU job, no download, no OpenVLA-OFT, no benchmark/multi-seed rollout, and no paper claim.

## Init-State Learned-Policy Recheck Plan

The next planning gate is defined by `scripts\101_plan_init_state_learned_policy_recheck.ps1`.

It reads the bounded HDF5 replay result, learned-policy rollout readiness report, WSL SmolVLA single-action smoke report, and prior reduced-scope learned-policy rollout report. It is planning-only: no model load, no inference, no simulator environment, no rollout, no training, no GPU job, no download, no OpenVLA-OFT, and no paper claim.

If the planner reports `decision=proceed`, a future runner may recheck the learned policy from the validated HDF5 initial-state convention using one task and at most five policy-controlled steps under task-local `ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK=1`. This remains diagnostic/local-pilot evidence only and does not unblock rollout scaling or paper-grade claims.

## Bounded Init-State Learned-Policy Recheck Result

The bounded recheck runner is defined by `scripts\102_bounded_init_state_learned_policy_recheck.ps1`.

Current local result:

- decision: `proceed`,
- wrapper/execution passed: true,
- HDF5 init state set in environment: true,
- task suite: `libero_10`,
- task count: 1,
- policy-controlled steps: 3,
- action adapter strategy: `policy_6d_delta_pose_plus_gripper_close`,
- diagnostic success: false,
- reward sum: `0.0`,
- downloads/installs/training/GPU/OpenVLA-OFT/multi-seed/paper claims: false.

Interpretation: the initial-state learned-policy topology now runs from the documented HDF5 convention, but it still produced no task success or reward. The next safe step is a report-only diagnostic metric summary versus previous reset-only learned-policy results, not rollout scaling or paper-grade claims.

## Init-State Recheck Metric Summary Result

The report-only metric summary is generated by `scripts\103_generate_init_state_recheck_metric_summary.ps1`.

Current local result:

- decision: `no_go_rollout_scaling`,
- summary passed: true,
- reset-only 3-step reward/success: `0.0` / false,
- reset-only 10-step reward/success: `0.0` / false,
- HDF5-init-state 3-step reward/success: `0.0` / false,
- HDF5 init state set in environment: true,
- init-state versus reset-only 3-step reward delta: `0.0`,
- paper-grade claim ready: false.

Interpretation: initial-state alignment, gripper-close adaptation, and bounded learned-policy execution are operational, but none produced a positive reward or diagnostic success signal. More rollout scaling is not justified from this evidence. The next safe research-engineering direction is checkpoint/task alignment, VLM loading policy, and offline demonstration-conditioned action decoding analysis.

## SmolVLA/LIBERO Checkpoint-Task Alignment Audit

The report-only checkpoint/task alignment audit is generated by `scripts\104_audit_smolvla_libero_checkpoint_task_alignment.ps1`.

Expected interpretation:

- `decision=no_go_rollout_scaling`: learned-policy rollout scaling remains blocked because the current local evidence still has zero reward/success, uncertain checkpoint-task provenance, VLM-loading-policy mismatch risk, and 6D policy action versus 7D LIBERO action convention risk.
- `ready_for_offline_demonstration_conditioned_action_decoding_plan=true`: the next safe research-engineering step is a planning-only gate for one-sample offline action decoding from a real LIBERO demonstration observation, not another rollout variant.

The audit is report-only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

## Offline Demonstration-Conditioned Action Decoding Plan

The planning-only offline action-decoding gate is generated by `scripts\105_plan_offline_demo_conditioned_action_decoding.ps1`.

Expected interpretation:

- `decision=proceed`: local report and file prerequisites are present for a future one-sample offline action-decoding runner.
- `ready_for_bounded_offline_demo_action_decoding_runner=true`: a later runner may be implemented under a separate task-local gate, read one HDF5 observation/action pair, load local SmolVLA on CPU, run exactly one `select_action` call, compare the decoded action to the expert action, and write diagnostic metrics.

The planner itself performs no model load, inference, simulator environment creation, rollout, training, GPU job, download, OpenVLA-OFT execution, token access, or paper claim. The future runner must still remain offline and non-rollout.

## Bounded Offline Demonstration Action Decoding

The one-sample offline action-decoding diagnostic is implemented by `scripts\106_bounded_offline_demo_action_decoding.ps1` and `tca_map.smolvla.offline_demo_action_decoding`.

Scope:

- reads one local LIBERO HDF5 observation/action pair,
- loads local SmolVLA on CPU,
- runs exactly one `select_action` call,
- adapts the decoded 6D action to the 7D expert-action convention,
- reports action L1/MSE to the expert action.

It does not create simulator environments, rollout, train, download, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims. It requires the task-local gate `ALLOW_OFFLINE_DEMO_ACTION_DECODING=1`.

## Offline Demonstration Action Decoding Summary

The report-only summary is generated by `scripts\107_summarize_offline_demo_action_decoding.ps1`.

Current expected interpretation after the one-sample diagnostic:

- action L1/MSE to expert are diagnostic-only offline metrics,
- weak alignment keeps rollout scaling blocked,
- the next safe direction is VLM loading policy, checkpoint provenance, and action normalization analysis before more learned-policy rollout variants.

The summary does not load models, infer, create simulator environments, rollout, train, download, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

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

## Camera-Source Diagnostic Runner Result

Current local bounded result: passed as diagnostic execution evidence only.

`scripts\88_plan_camera_source_diagnostic.ps1` reported `decision=proceed`, and `scripts\89_bounded_camera_source_diagnostic.ps1` ran under task-local `ALLOW_CAMERA_SOURCE_DIAGNOSTIC=1`.

Observed local result:

- camera alias strategies tested: `current_aliases`, `camera3_eye_in_hand`, and `all_agentview`,
- image sources changed as expected and were recorded in metadata,
- variants completed: 3,
- wrapper/execution passed for all variants,
- diagnostic success rate: 0.0 for all variants,
- reward sum: 0.0 for all variants,
- camera changes produced different continuous action previews,
- no downloads, no installs, no training, no GPU job, no OpenVLA-OFT, no multi-seed evaluation, no token access, and no paper claim.

Interpretation: camera source selection is now execution-tested as an interface axis, but it did not produce a positive diagnostic signal. Rollout scaling remains blocked. The next safe rung is a bounded state-sufficiency diagnostic before any broader learned-policy rollout matrix.

## VLM Loading Policy and Action-Normalization Audit

The report-only VLM/action audit is generated by `scripts\108_plan_vlm_loading_policy_action_normalization_audit.ps1`.

Current expected result:

- decision: `no_go_rollout_scaling`,
- rollout scaling ready: false,
- repeated offline decoding plan ready: true,
- paper-grade claim ready: false.

The audit records that the local checkpoint config requests `load_vlm_weights=true`, while the bounded offline diagnostic observed `load_vlm_weights=false`. It also records ACTION `MEAN_STD` normalization, 6D policy action versus 7D LIBERO expert action, gripper-close adaptation, and action clipping. This keeps learned-policy rollout scaling blocked and points to a tiny repeated offline HDF5 action-decoding diagnostic before any further rollout.

## Repeated Offline Demonstration Action-Decoding Plan

The planning-only repeated offline decoding gate is generated by `scripts\109_plan_repeated_offline_demo_action_decoding.ps1`.

Expected result:

- decision: `proceed`,
- runner ready: true,
- planned HDF5 timesteps: at most 3,
- rollout scaling ready: false,
- paper-grade claim ready: false.

This planner does not load SmolVLA or run inference. It inspects the selected local HDF5 file and prepares a bounded future runner that may decode at most three HDF5 timesteps on CPU under `ALLOW_REPEATED_OFFLINE_DEMO_DECODING=1`.

## Bounded Repeated Offline Demonstration Action Decoding

The bounded repeated offline decoder is implemented by `scripts\110_bounded_repeated_offline_demo_action_decoding.ps1` and `tca_map.smolvla.repeated_offline_demo_action_decoding`.

Scope:

- loads local SmolVLA on CPU only under `ALLOW_REPEATED_OFFLINE_DEMO_DECODING=1`,
- decodes at most three local LIBERO HDF5 timesteps,
- records action L1/MSE, clipping, gripper strategy, `load_vlm_weights`, and image aliases,
- does not create simulator environments, rollout, train, download, use GPU jobs, execute OpenVLA-OFT, or make paper claims.

Expected interpretation: if repeated offline alignment remains weak, rollout scaling stays blocked and the next safe research direction is VLM-enabled loading risk, checkpoint provenance, or action-normalization analysis.

Current local result:

- runner passed,
- decoded timesteps: `0`, `136`, and `271`,
- sample count: 3,
- mean action L1 to expert: `0.412322`,
- max action L1 to expert: `0.478394`,
- mean action MSE to expert: `0.286972`,
- mean policy-6D L1 to expert first 6 dimensions: `0.608221`,
- clipped action values total: 3,
- `load_vlm_weights=false`,
- offline alignment signal: `weak`,
- rollout scaling ready: false,
- paper-grade claim ready: false.

Interpretation: the bounded repeated offline diagnostic confirms that weak expert-action alignment is not just a single-timestep artifact. The next safe direction is a report-only VLM-enabled loading risk/provenance plan and action-normalization diagnosis before any further learned-policy rollout.

## VLM-Enabled Loading Risk Plan

The metadata-only VLM-enabled loading risk planner is implemented by `scripts\111_plan_vlm_enabled_loading_risk.ps1`.

Purpose:

- verify the official source `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`,
- estimate required config/tokenizer/root `model.safetensors` size,
- verify gated/private/license/token risk,
- check free-disk-after-acquisition budget,
- decide whether a later VLM weight acquisition plan is safe.

This planner does not download weights, load models, infer, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

Current local result:

- source: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`,
- source status: official/public/ungated,
- license: `apache-2.0`,
- root model weight: `model.safetensors`,
- estimated required file size: `1.895GB`,
- free disk after estimate: about `419GB`,
- token/login/license/payment required: false,
- decision: `proceed`,
- ready for VLM weight acquisition plan: true,
- ready for VLM-enabled load smoke: false.

Next safe direction: create a separately gated VLM weight acquisition plan/runner for required files only. Do not run VLM-enabled loading until acquisition and a bounded load-smoke plan pass.

## VLM Required Files Acquisition Result

The bounded required-file acquisition runner is implemented by `scripts\112_acquire_vlm_required_files.ps1` and `tca_map.smolvla.vlm_required_files_acquisition`.

Current local result:

- source: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`,
- target: `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`,
- cache: `C:\assets\hf_home`,
- acquired required files: root `model.safetensors` plus config/tokenizer/processor files,
- detected target size after acquisition: about `1.895GB`,
- decision: `acquisition_complete`,
- ready for bounded VLM-enabled load-smoke planning: true.

This stage performed a bounded official download only under task-local `ALLOW_DOWNLOADS=1`. It did not load models, run inference, run training, run rollouts, use GPU jobs, execute OpenVLA-OFT, access tokens, install packages, or make paper-grade claims.

The next safe research-engineering step is a separate bounded VLM-enabled load-smoke planner. Do not enable `load_vlm_weights=true` in execution until that planner passes and the load smoke remains inside the RAM/runtime budget.

## VLM-Enabled Load Smoke Plan

The planning-only VLM-enabled load-smoke gate is implemented by `scripts\113_plan_vlm_enabled_load_smoke.ps1`.

Purpose:

- verify that SmolVLA adapter readiness, runtime dependencies, VLM risk metadata, and VLM required-file acquisition are all present,
- estimate a CPU-first `load_vlm_weights=true` load-only task,
- keep model loading, inference, training, rollout, GPU jobs, OpenVLA-OFT, token access, and paper claims out of the planner.

Expected interpretation:

- `decision=proceed`: a future separately gated bounded runner may be created with `ALLOW_HEAVY_IMPORT=1` and `ALLOW_VLM_ENABLED_LOAD_SMOKE=1`.
- `decision=stop`: resolve the listed readiness/RAM/acquisition/gate blocker before attempting VLM-enabled loading.

Passing the plan still does not prove VLM-enabled loading works. It only authorizes implementation of a bounded load-only runner.

Current local result:

- decision: `proceed`,
- ready for bounded VLM-enabled load-smoke runner: true,
- expected future task: CPU-first `load_vlm_weights=true` load-only construction,
- expected runtime: 15 minutes,
- expected RAM: 16GB,
- observed total/free RAM at planning time: about `23.163GB` / `8.538GB`,
- required future gates: `ALLOW_HEAVY_IMPORT=1` and `ALLOW_VLM_ENABLED_LOAD_SMOKE=1`.

The plan performed no model load, inference, training, rollout, GPU job, download, install, OpenVLA-OFT execution, token access, or paper claim.

## Bounded VLM-Enabled Load Smoke

The bounded VLM-enabled load-only runner is implemented by `scripts\114_bounded_vlm_enabled_load_smoke.ps1` and `tca_map.smolvla.vlm_enabled_load_smoke`.

Scope:

- requires `ALLOW_HEAVY_IMPORT=1` and `ALLOW_VLM_ENABLED_LOAD_SMOKE=1`,
- CPU-first,
- sets `load_vlm_weights=true`,
- constructs and releases the local SmolVLA policy,
- does not call `select_action`,
- does not train, rollout, use GPU jobs, download, install, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

Passing this runner is engineering load-smoke evidence only. It does not prove offline action alignment, simulator success, benchmark success, counterfactual robustness, or paper-grade readiness.

Current local result:

- decision: `load_smoke_complete`,
- runner passed: true,
- `load_vlm_weights`: true,
- device: CPU,
- parameter count: `450046176`,
- trainable parameter count as loaded: `99880992`,
- load elapsed: about `8.484s`,
- total runner elapsed: about `14.5s`,
- RSS before/after load: about `646MB` / `1726MB`,
- CUDA max allocated: `0MB`,
- downloads/installs/inference/training/rollout/GPU jobs/OpenVLA-OFT/tokens/paper claims: false.

Interpretation: the local dependency files are sufficient for VLM-enabled SmolVLA construction on CPU within the bounded load-only budget. This still does not prove action alignment; the next safe step is a separately planned repeated offline action-decoding recheck with VLM enabled.

## VLM-Enabled Repeated Offline Decoding Plan

The planning-only VLM-enabled repeated offline recheck gate is implemented by `scripts\115_plan_vlm_enabled_repeated_offline_decoding.ps1`.

Purpose:

- compare the previous repeated offline diagnostic using `load_vlm_weights=false`,
- require the bounded VLM-enabled load-only smoke to have passed,
- reuse the existing selected LIBERO HDF5 timesteps,
- authorize only a future CPU offline diagnostic with at most three policy calls.

This planner does not load models, infer, train, rollout, download, install, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

Current local result:

- decision: `proceed`,
- ready for bounded VLM-enabled repeated offline decoding runner: true,
- baseline to compare: previous `load_vlm_weights=false` repeated offline diagnostic,
- previous mean action L1/MSE: `0.412322` / `0.286972`,
- previous alignment signal: `weak`,
- selected HDF5 timesteps: `0`, `136`, and `271`,
- future runner expected runtime: 20 minutes,
- future runner expected RAM: 18GB,
- expected VRAM: 0GB.

The future runner remains offline diagnostic evidence only. It must not create simulator environments, rollout, train, download, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

## Bounded VLM-Enabled Repeated Offline Decoding

The bounded VLM-enabled repeated offline decoder is implemented by `scripts\116_bounded_vlm_enabled_repeated_offline_decoding.ps1` and `tca_map.smolvla.vlm_enabled_repeated_offline_decoding`.

Scope:

- requires `ALLOW_HEAVY_IMPORT=1` and `ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING=1`,
- CPU-only,
- loads local SmolVLA with `load_vlm_weights=true`,
- decodes at most three local LIBERO HDF5 timesteps,
- reports action L1/MSE deltas versus the previous no-VLM repeated offline diagnostic,
- does not create simulator environments, rollout, train, download, install, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

Passing this runner is still offline diagnostic evidence only.

Current local result:

- decision: `diagnostic_complete`,
- runner passed: true,
- decoded timesteps: `0`, `136`, and `271`,
- `load_vlm_weights=true`,
- mean action L1 to expert: `0.301665`,
- mean action MSE to expert: `0.216188`,
- mean action L1 delta versus previous no-VLM repeated offline diagnostic: `-0.110657`,
- mean action MSE delta versus previous no-VLM repeated offline diagnostic: `-0.070784`,
- clipped action values total: `3`,
- CUDA max allocated: `0MB`,
- total policy inference elapsed: about `1.344s`,
- downloads/installs/training/rollout/GPU jobs/OpenVLA-OFT/token access/paper claims: false.

Interpretation: enabling VLM weights improves the tiny offline action-distance metrics versus the previous no-VLM repeated diagnostic, but the alignment signal is still `weak` and still not rollout, benchmark, standard-success, counterfactual-robustness, or paper-grade evidence. The next safe step is a report-only VLM-enabled versus no-VLM offline decoding summary and action-normalization/provenance analysis before any rollout scaling.

## VLM-Enabled Offline Decoding Summary

The report-only VLM-on/off comparison is implemented by `scripts\117_summarize_vlm_enabled_offline_decoding.ps1` and `tca_map.smolvla.vlm_enabled_offline_decoding_summary`.

Scope:

- reads the existing no-VLM and VLM-enabled repeated offline decoding runtime reports,
- reads local SmolVLA config/preprocessor/postprocessor JSON files,
- computes action-distance deltas and remaining blockers,
- does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

Current local result:

- summary passed,
- no-VLM mean action L1/MSE: `0.412322` / `0.286972`,
- VLM-enabled mean action L1/MSE: `0.301665` / `0.216188`,
- L1/MSE reduction: `26.838%` / `24.666%`,
- both no-VLM and VLM-enabled alignment signals remain `weak`,
- both runs have clipped values total `3`,
- config action normalization is ACTION `MEAN_STD`,
- policy action shape is `6`, while local LIBERO expert actions are adapted as `7D`,
- rollout scaling, benchmark claims, and paper claims remain blocked.

Interpretation: VLM-enabled loading is behaviorally relevant, but the current blocker has shifted to action-normalization/provenance and 6D-to-7D adapter interpretation. The next safe step is a report-only action-normalization/provenance audit before any learned-policy rollout scaling.

## Action Normalization Provenance Audit

The report-only action normalization/provenance audit is implemented by `scripts\118_audit_action_normalization_provenance.ps1` and `tca_map.smolvla.action_normalization_provenance_audit`.

Scope:

- reads local SmolVLA config/preprocessor/postprocessor JSON,
- reads processor safetensors with `safetensors.safe_open`,
- reads existing VLM-enabled offline diagnostic reports,
- compares checkpoint action-stat prefixes and scale against local LIBERO action previews,
- does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

Current local result:

- audit passed,
- decision: `no_go_rollout_scaling`,
- action stat prefixes: `so100`, `so100-blue`, `so100-red`,
- action mean max abs: `125.720543`,
- action std max: `59.359951`,
- local LIBERO expert action preview max abs: `1.0`,
- policy action shape: `[6]`,
- adapted decoded actions still clipped `3` values,
- rollout scaling, benchmark claims, and paper claims remain blocked.

Interpretation: the current learned-policy rollout blocker is a strong action-stat/checkpoint-provenance mismatch risk. The next safe step is a planning-only action-stat mapping or checkpoint/task-provenance correction plan before any learned-policy rollout scaling.

## Action-Stat Provenance Correction Plan

The planning-only correction gate is implemented by `scripts\119_plan_action_stat_provenance_correction.ps1` and `tca_map.smolvla.action_stat_provenance_correction_plan`.

Scope:

- reads the action normalization provenance audit report,
- selects the next safe correction/audit step,
- does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, alter policy behavior, access tokens, or make paper claims.

Expected interpretation: if the provenance mismatch is present, reduce scope to a report-only LIBERO action-stat subset audit before any normalized-action probe, postprocessor bypass, checkpoint download, or learned-policy rollout.

Current local result:

- plan passed,
- decision: `reduce_scope`,
- selected next step: `libero_action_stat_subset_audit`,
- ready for LIBERO action-stat audit: true,
- rollout scaling, benchmark claims, and paper claims remain blocked.

Interpretation: the next safe task is to compute bounded LIBERO HDF5 action statistics and compare them against checkpoint processor stats before any policy behavior change or rollout.

## LIBERO Action-Stat Subset Audit

The report-only LIBERO action-stat subset audit is implemented by `scripts\120_audit_libero_action_stats.ps1` and `tca_map.smolvla.libero_action_stat_subset_audit`.

Scope:

- reads bounded local LIBERO HDF5 `actions` arrays,
- samples at most 5 files and 500 actions per file by default,
- compares local action scale/dimensions against checkpoint processor action stats,
- does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, alter policy behavior, access tokens, or make paper claims.

Expected interpretation: if local LIBERO actions are unit-scale 7D while checkpoint processor stats are SO100/large-scale 6D, learned-policy rollout scaling remains blocked and the next safe step is a normalized-action-space probe or checkpoint/task provenance resolution plan.

Current local result:

- audit passed,
- decision: `no_go_rollout_scaling`,
- sampled files/actions: `5` / `2500`,
- LIBERO action dim: `7`,
- LIBERO action max abs: `1.0`,
- checkpoint action stat prefixes: `so100`, `so100-blue`, `so100-red`,
- checkpoint action mean max abs: `125.720543`,
- checkpoint action std max: `59.359951`,
- scale mismatch confirmed: true,
- dimension mismatch confirmed: true.

Interpretation: learned-policy rollout scaling remains blocked because the current checkpoint action statistics and local LIBERO action convention are mismatched. The next safe task is a normalized-action-space probe or checkpoint/task provenance resolution plan.

## Normalized Action-Space Probe Plan

The planning-only normalized-action/provenance gate is implemented by `scripts\121_plan_normalized_action_space_probe.ps1` and `tca_map.smolvla.normalized_action_space_probe_plan`.

Scope:

- reads the LIBERO action-stat subset audit runtime report,
- optionally reads the VLM-enabled offline decoding summary runtime report,
- chooses between checkpoint/task provenance resolution, a future bounded normalized-action-space probe, and an offline head/TCA-Map pivot,
- does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, change policy behavior, access tokens, or make paper claims.

Expected interpretation:

- if SO100-prefixed checkpoint stats and 7D unit-scale LIBERO actions are both confirmed, the selected next step is a report-only checkpoint/task provenance resolution audit,
- a normalized-action-space probe remains deferred until provenance is resolved and a separate task-local gate exists,
- learned-policy rollout scaling remains blocked.

Current local result:

- plan passed,
- decision: `reduce_scope`,
- selected next step: `checkpoint_task_provenance_resolution`,
- ready for checkpoint/task provenance resolution: true,
- ready for bounded normalized-action-space probe runner: false,
- ready for rollout scaling, benchmark claims, and paper claims: false.

Interpretation: the current checkpoint/action mismatch is strong enough that provenance must be resolved before any postprocessor bypass, normalized-action runner, learned-policy rollout scaling, or paper claim.

## Checkpoint / Task Provenance Resolution

The report-only provenance resolver is implemented by `scripts\122_resolve_checkpoint_task_provenance.ps1` and `tca_map.smolvla.checkpoint_task_provenance_resolution`.

Scope:

- reads the local SmolVLA checkpoint config, policy preprocessor, policy postprocessor, and README,
- reads the normalized-action plan and LIBERO action-stat subset audit runtime reports,
- decides whether the current checkpoint can support learned-policy LIBERO rollout evidence,
- does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, change policy behavior, access tokens, or make paper claims.

Expected interpretation:

- if checkpoint metadata stays 6D/SO100-like while LIBERO HDF5 action stats stay 7D/unit-scale, learned-policy rollout scaling with this checkpoint remains no-go,
- the next safe path is offline/head TCA-Map plus required LoRA evidence, or a separate source-resolution plan for a LIBERO-action-aligned SmolVLA checkpoint,
- normalized-action probes remain deferred until provenance is resolved and a separate gate exists.

Current local result:

- audit passed,
- decision: `no_go_learned_policy_rollout_scaling`,
- current checkpoint valid for LIBERO learned-policy rollout evidence: false,
- selected next step: `pivot_to_offline_head_tca_map_and_lora_or_find_libero_aligned_checkpoint`,
- ready for offline/head TCA-Map pivot: true,
- ready for LIBERO-aligned checkpoint source plan: true,
- ready for normalized-action-space runner: false,
- ready for rollout scaling, benchmark claims, and paper claims: false.

Interpretation: the current `lerobot/smolvla_base` checkpoint should be treated as a base/SO100-like SmolVLA asset for local interface/offline diagnostics, not as a paper-relevant LIBERO learned-policy rollout baseline. The next safe paper path is to continue real-LIBERO offline/head TCA-Map and required LoRA evidence, or separately resolve a LIBERO-action-aligned checkpoint source before any further learned-policy rollout evidence.

## Offline TCA-Map / LoRA Pivot Plan

The report-only pivot gate is implemented by `scripts\123_plan_offline_tca_map_lora_pivot.ps1` and `tca_map.smolvla.offline_tca_map_lora_pivot_plan`.

Scope:

- reads the checkpoint/task provenance report,
- reads existing LIBERO offline ActionMap/TCA-Map and LoRA comparison reports,
- decides whether to consolidate an offline evidence table and gap report,
- does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, change policy behavior, access tokens, or make paper claims.

Expected interpretation:

- if provenance blocks current-checkpoint learned-policy rollout scaling and offline reports are present, pivot to a report-only offline evidence table and gap report,
- keep learned-policy rollout scaling blocked until a LIBERO-action-aligned checkpoint source or bounded adaptation path is validated,
- keep standard success, benchmark success, SOTA, and paper-grade claims blocked.

Current local result:

- plan passed,
- decision: `pivot_offline_evidence_ladder`,
- selected next step: `consolidate_offline_tca_lora_evidence_table_and_gap_report`,
- ready for offline evidence table: true,
- ready for LoRA scale-up planning: true,
- ready for learned-policy rollout scaling: false,
- ready for benchmark or paper claim: false.

Interpretation: the safe low-compute paper path is now to summarize the real-LIBERO offline ActionMap/TCA-Map/TCA-Select/LoRA evidence and its gaps, while keeping current-checkpoint learned-policy rollout evidence blocked.

## Offline Evidence Gap Report

The report-only offline evidence table is implemented by `scripts\124_generate_offline_evidence_gap_report.ps1` and `tca_map.smolvla.offline_evidence_gap_report`.

Scope:

- reads the offline pivot plan,
- reads real-LIBERO offline head-only, TCA-Select, LoRA, provenance, and bounded-pilot reports,
- produces a consolidated evidence table and gap list,
- does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, change policy behavior, access tokens, or make paper claims.

Expected interpretation:

- useful offline proxy evidence exists for target-conditioned decoding and required LoRA tracks,
- current-checkpoint learned-policy rollout scaling remains blocked,
- the next safe task is a bounded LoRA/offline-proxy scale-up plan, not a paper claim.

Current local result:

- report passed,
- decision: `offline_evidence_table_ready`,
- evidence arms consolidated: 6,
- ready for LoRA scale-up planning: true,
- ready for offline proxy extension planning: true,
- ready for current-checkpoint learned-policy rollout scaling: false,
- ready for benchmark or paper claim: false.

Interpretation: the next safe work item is a planning-only bounded LoRA/offline-proxy scale-up on real LIBERO HDF5 subsets, while preserving the no-rollout/no-paper-claim boundary.

## Bounded LoRA / Offline Proxy Scale-Up Plan

The planning-only LoRA/offline scale-up gate is implemented by `scripts\125_plan_bounded_lora_offline_scaleup.ps1` and `tca_map.smolvla.bounded_lora_offline_scaleup_plan`.

Scope:

- reads the offline evidence gap runtime report,
- defines a bounded future CPU-only LoRA/offline proxy runner budget,
- caps the future runner to 16 pairs, 64 samples, 64 steps, 20 minutes, LoRA rank 4, frozen base weights, and no full fine-tuning,
- does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, change policy behavior, access tokens, or make paper claims.

Expected interpretation:

- if the plan passes, implement a separately gated offline LoRA scale-up runner under `ALLOW_TINY_TRAINING=1`,
- still label the future result as offline proxy only,
- current-checkpoint learned-policy rollout scaling remains blocked.

Current local result:

- plan passed,
- decision: `proceed_bounded_offline_lora_scaleup_runner`,
- ready for bounded offline LoRA scale-up runner: true,
- limits: 16 pairs, 64 samples, 64 steps, LoRA rank 4, CPU-only, frozen base, no full fine-tuning,
- required future gate: `ALLOW_TINY_TRAINING=1`,
- ready for learned-policy rollout scaling, benchmark claims, and paper claims: false.

Interpretation: the next safe task is implementation of the separately gated CPU-only offline LoRA scale-up runner.

## Bounded LIBERO Offline LoRA Scale-Up Runner

The bounded CPU-only runner is implemented by `scripts\126_bounded_lora_offline_scaleup.ps1` and `tca_map.datasets.libero_offline_lora_scaleup`.

Scope:

- requires task-local `ALLOW_TINY_TRAINING=1`,
- trains only tiny NumPy LoRA matrices over local LIBERO HDF5 action snippets,
- caps execution to 16 pairs, 64 records, 64 update steps, LoRA rank 4, and an enforced 900-second runtime cap,
- freezes the base representation and forbids full fine-tuning,
- does not download, install, import heavy VLA models, load SmolVLA, infer, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims.

Expected interpretation:

- if the runner passes, its result is useful offline proxy evidence for the required LoRA track,
- it is not standard success, not rollout success, not a SmolVLA model-load result, and not paper-grade evidence,
- the next safe task is to refresh the offline evidence table/gap report with the bounded scale-up result while keeping current-checkpoint learned-policy rollout scaling blocked.

Current local result:

- runner passed,
- `bounded_lora_offline_scaleup_passed=true`,
- record count: 16,
- max steps: 64,
- LoRA rank: 4,
- TCA-Map + LoRA vs ActionMap + LoRA action L1 delta: -0.004018,
- TCA-Map + LoRA vs ActionMap + LoRA wrong-target proxy delta: -0.4375,
- Distributional TCA-Select did not change the current tiny LoRA proxy metrics in this runner,
- GPU jobs, downloads, heavy imports, model loading, inference, rollouts, simulator execution, OpenVLA-OFT, token access, full fine-tuning, and paper claims remained false.

Interpretation: the bounded LoRA scale-up is now ready to be folded into the offline evidence table. It remains offline proxy evidence only.

## Scale-Up-Aware Offline Evidence Gap Refresh

`scripts\124_generate_offline_evidence_gap_report.ps1` now accepts `reports\bounded_lora_offline_scaleup_report.json` through `--bounded-lora-scaleup-report`.

Expected interpretation:

- if the bounded scale-up report exists and passed, the evidence table includes three additional bounded offline LoRA proxy rows,
- the refreshed report remains report-only and offline proxy only,
- learned-policy rollout scaling and paper claims remain blocked for the current checkpoint.

Current local result:

- refresh passed,
- `bounded_lora_scaleup_included=true`,
- bounded scale-up record count: 16,
- evidence rows: 9,
- bounded TCA-Map + LoRA vs ActionMap + LoRA action L1 delta: -0.004018,
- bounded TCA-Map + LoRA vs ActionMap + LoRA wrong-target proxy delta: -0.4375,
- bounded TCA-Select + LoRA vs TCA-Map + LoRA deltas: 0.0 in the current proxy,
- learned-policy rollout scaling, benchmark claims, and paper claims remain false.

Interpretation: the next safe task is a report-only attribution-gap synthesis. The evidence table supports continued low-compute method debugging, not paper-grade claims.

## Scale-Up Attribution Gap Synthesis

The report-only synthesis is implemented by `scripts\127_synthesize_scaleup_attribution_gaps.ps1` and `tca_map.smolvla.scaleup_attribution_gap_synthesis`.

Scope:

- reads the scale-up-aware offline evidence gap runtime report,
- explains target-conditioning, LoRA, and Distributional TCA-Select attribution gaps,
- keeps all outputs offline proxy only,
- does not train, download, install, import heavy VLA models, load models, infer, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims.

Expected interpretation:

- if the synthesis passes, the next safe step is a report-only TCA-Select ambiguity/stress-test plan,
- Distributional TCA-Select needs a stronger candidate-diversity proxy before claiming selection-specific gain,
- learned-policy rollout scaling and paper claims remain blocked.

Current local result:

- synthesis passed,
- decision: `scaleup_attribution_gaps_ready`,
- bounded LoRA scale-up included: true,
- bounded TCA-Map + LoRA vs ActionMap + LoRA wrong-target proxy delta: -0.4375,
- bounded TCA-Select + LoRA vs TCA-Map + LoRA deltas: 0.0,
- ready for learned-policy rollout scaling and paper claims: false.

Interpretation: the next safe task is a report-only TCA-Select candidate-ambiguity stress-test plan.

## TCA-Select Ambiguity Stress-Test Plan

The planning-only stress-test gate is implemented by `scripts\128_plan_tca_select_ambiguity_stress_test.ps1` and `tca_map.smolvla.tca_select_ambiguity_stress_plan`.

Scope:

- reads the scale-up attribution-gap synthesis report,
- defines ambiguous candidate generation, target/action consistency scoring, condition-sensitivity scoring, and offline proxy pass/fail criteria,
- forbids privileged simulator state, external verifiers, rollout outcomes, and paper-grade success labels,
- does not train, download, install, import heavy VLA models, load models, infer, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims.

Expected interpretation:

- if the plan passes, implement a CPU-only offline TCA-Select ambiguity stress-test runner,
- keep results offline proxy only,
- use this only to debug selection-specific gain before any paper claim.

Current local result:

- plan passed,
- decision: `proceed_offline_tca_select_ambiguity_stress_runner`,
- ready for offline runner: true,
- candidate count: 8,
- max records: 64,
- device: CPU,
- ready for rollout, benchmark claims, and paper claims: false.

Interpretation: the next safe task is implementation of the CPU-only offline ambiguity stress-test runner.

## Offline TCA-Select Ambiguity Stress-Test Runner

The CPU-only offline runner is implemented by `scripts\129_run_tca_select_ambiguity_stress_test.ps1` and `tca_map.smolvla.tca_select_ambiguity_stress`.

Scope:

- reads the existing local LIBERO offline counterfactual split report,
- reads bounded HDF5 action snippets,
- creates ambiguous target/action candidate heatmaps without loading SmolVLA,
- compares Distributional TCA-Select against a top-heatmap baseline,
- reports wrong-target proxy, action L1, condition sensitivity, target consistency, candidate diversity, latency, and GPU memory,
- does not train, download, install, import heavy VLA models, load models, infer with SmolVLA, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims.

Expected interpretation:

- if the runner passes, it provides selection-specific offline proxy evidence,
- the result is not standard success, not rollout success, and not paper-grade evidence,
- the next safe task is an attribution synthesis refresh that includes the stress-test result.

Current local result:

- runner passed,
- `tca_select_ambiguity_stress_passed=true`,
- record count: 16,
- selected wrong-target proxy rate: 0.0,
- top-heatmap wrong-target proxy rate: 1.0,
- wrong-target proxy delta vs top heatmap: -1.0,
- selected action L1: 0.0,
- top-heatmap action L1: 0.164299,
- action L1 delta vs top heatmap: -0.164299,
- latency: 0.428231 ms,
- max GPU memory: 0.0 MB,
- model loading, training, rollout, GPU jobs, simulator execution, OpenVLA-OFT, and paper claims remained false.

Interpretation: Distributional TCA-Select now has selection-specific offline proxy evidence in an ambiguity stress setting. This still does not unblock standard success, learned-policy rollout claims, or paper-grade claims. The next safe task is to refresh the attribution synthesis/evidence table with the stress-test result.

## Stress-Aware Attribution Synthesis

`scripts\127_synthesize_scaleup_attribution_gaps.ps1` now accepts `reports\tca_select_ambiguity_stress_report.json` as an optional input.

Expected interpretation:

- LoRA scale-up attribution and TCA-Select stress-test attribution are kept separate,
- zero TCA-Select delta in the bounded LoRA runner is preserved as a gap,
- positive TCA-Select ambiguity-stress proxy evidence is recorded separately,
- paper readiness, benchmark readiness, rollout readiness, and learned-policy rollout scaling remain false.

## Stress-Aware Offline Evidence Table

`scripts\124_generate_offline_evidence_gap_report.ps1` now accepts `reports\tca_select_ambiguity_stress_report.json` as an optional input.

Expected interpretation:

- if the stress report is present and passed, the evidence table gains a dedicated `Distributional TCA-Select ambiguity stress` row,
- selected action L1 and selected wrong-target proxy rate are table columns,
- top-heatmap comparison deltas remain in `deltas.tca_select_ambiguity_stress_vs_top_heatmap`,
- this remains offline proxy evidence only and does not unlock rollout or paper claims.

Current local result:

- evidence table refreshed successfully,
- evidence row count: 10,
- `tca_select_ambiguity_stress_included=true`,
- stress wrong-target delta vs top heatmap: -1.0,
- stress action L1 delta vs top heatmap: -0.164299,
- stress-aware attribution synthesis now routes to report-only learned-policy candidate-generation readiness planning,
- model inference, heavy imports, training, rollout, GPU jobs, OpenVLA-OFT, and paper claims remain false.

## Candidate-Generation Readiness Planning

`scripts\130_plan_candidate_generation_readiness.ps1` is a report-only gate for the next learned-policy interface step.

Scope:

- reads the stress-aware attribution synthesis and prior smoke reports,
- defines the future candidate action heatmap contract,
- keeps real model inference and model loading blocked,
- routes the next safe task to a synthetic-tensor contract checker.

Current local result:

- candidate-generation readiness plan passed,
- prior SmolVLA load-only, single-sample interface, and feature-cache reports were green,
- contract checker readiness is true,
- real candidate-generation smoke execution remains false,
- next safe task is synthetic-tensor candidate-generation contract checking.

## Candidate-Generation Contract Check

`scripts\131_check_candidate_generation_contract.ps1` validates the synthetic candidate heatmap contract before any real model inference.

Scope:

- synthetic tensors only,
- no heavy import, model load, model inference, training, rollout, GPU job, simulator execution, OpenVLA-OFT, or paper claim,
- validates candidate list, low-resolution heatmap, masked heatmap, target heatmap, metadata, and TCA-Select compatibility.

Current local result:

- contract checker passed,
- candidate count: 4,
- heatmap grid: 8,
- selected candidate index: 0,
- max GPU memory: 0.0 MB,
- real candidate-generation smoke execution remains false,
- next safe task is a planning-only real candidate-generation smoke risk gate.

## Real Candidate-Generation Smoke Plan

`scripts\132_plan_real_candidate_generation_smoke.ps1` is a planning-only risk gate for a future bounded real candidate-generation smoke.

Scope:

- reads the synthetic contract check, runtime dependencies, load-only smoke, and single-sample interface smoke reports,
- does not run heavy import or inference,
- if green, allows only future implementation planning with task-local gates,
- keeps smoke execution, rollout, training, OpenVLA-OFT, and paper claims false.

Current local result:

- plan passed,
- decision: `proceed_bounded_real_candidate_generation_smoke_implementation`,
- implementation readiness: true,
- execution readiness: false,
- no blockers,
- required future gates: `ALLOW_REAL_CANDIDATE_GENERATION_SMOKE=1`, `ALLOW_HEAVY_IMPORT=1`, `ALLOW_SINGLE_SAMPLE_INFERENCE=1`,
- no model load, inference, training, rollout, GPU job, simulator execution, OpenVLA-OFT, token access, or paper claim occurred.

## Bounded Real Candidate-Generation Smoke Scaffold

`scripts\133_bounded_real_candidate_generation_smoke.ps1` and `tca_map.smolvla.real_candidate_generation_smoke` implement the separately gated smoke selected by the planning step.

Default behavior:

- refuses execution unless `ALLOW_REAL_CANDIDATE_GENERATION_SMOKE=1`, `ALLOW_HEAVY_IMPORT=1`, and `ALLOW_SINGLE_SAMPLE_INFERENCE=1` are all set task-locally,
- writes ignored runtime reports under `reports\real_candidate_generation_smoke_report.*`,
- performs no downloads, installs, training, rollouts, simulator environment creation, OpenVLA-OFT execution, external verifier use, privileged inference, token access, or paper claims.

If the gates are set after a green risk assessment, the bounded runner may load local SmolVLA on CPU, run one synthetic `select_action` call, build at most four low-resolution target-conditioned candidates, and run Distributional TCA-Select over the resulting heatmaps. This remains engineering smoke evidence only, not standard success, rollout success, or paper-grade evidence.

## Bounded Real Candidate-Generation Smoke Result

Current local result:

- script: `scripts\133_bounded_real_candidate_generation_smoke.ps1`,
- task-local gates used only for the task: `ALLOW_REAL_CANDIDATE_GENERATION_SMOKE=1`, `ALLOW_HEAVY_IMPORT=1`, `ALLOW_SINGLE_SAMPLE_INFERENCE=1`,
- result: passed,
- elapsed time: about 38.3 seconds,
- device: CPU,
- CUDA max allocated: `0.0 MB`,
- candidate count: 4,
- heatmap grid: 8,
- selected candidate index: 0,
- selected target index: 0,
- wrong-target proxy: false,
- downloads, installs, training, rollouts, simulator environment creation, GPU jobs, OpenVLA-OFT execution, external verifier use, privileged state, token access, and paper claims: false.

Interpretation: this is engineering interface evidence that local SmolVLA can seed a tiny low-resolution candidate heatmap and Distributional TCA-Select can select a candidate without privileged inference. It is not standard success, not rollout success, and not paper-grade evidence.

The report-only synthesis step is `scripts\134_summarize_real_candidate_generation_smoke.ps1`.

## Bounded Learned-Policy Rollout Diagnostic Result

Execution-first rollout diagnostics have now run beyond the initial learned-policy rollout:

- adapter/gripper variants,
- action-scale variants,
- prompt-format variants,
- camera-source variants,
- state-sufficiency variants,
- HDF5 demo init-state learned-policy recheck.

All bounded variants executed within the diagnostic policy and produced zero reward and zero diagnostic success on the selected LIBERO task. The init-state recheck passed execution with local SmolVLA inference on CPU and the HDF5 demo initial state set in the environment, but still produced `success=false` and `reward_sum=0.0` over 5 steps.

Safety status for this milestone:

- downloads: false,
- training: false,
- LoRA training: false,
- GPU job: false,
- OpenVLA-OFT execution: false,
- paper-grade claim: false,
- rollout: true, bounded diagnostic only,
- loss: none; no loss because this was not a training task.

Current interpretation: the rollout bridge is not blocked by a simple action-shape mismatch. The remaining blocker is policy/checkpoint/task/action-stat provenance or policy capability under the current LIBERO setup. Do not scale learned-policy rollouts from this evidence.

Next execution-first milestone: run the fixed-integrity ActionMap vs TCA-Map tiny offline training/evaluation comparison over real LIBERO HDF5 snippets, producing loss curves and offline proxy metrics before any further learned-policy rollout scaling.

## Tiny Offline ActionMap vs TCA-Map Training/Eval Result

The execution-first ActionMap vs TCA-Map milestone has run on local LIBERO HDF5 snippets.

Scope and safety:

- script: `scripts\52_compare_libero_offline_actionmap_tca.ps1`,
- task-local gate: `ALLOW_TINY_TRAINING=1`,
- data source: `reports\libero_offline_counterfactual_split_report.json`,
- samples: 8 records, deterministic manifest order,
- split: 6 train records / 2 eval records,
- steps: 64,
- batch size: 1,
- device: CPU NumPy,
- training: true,
- LoRA training: false,
- rollout: false,
- downloads/GPU jobs/heavy imports/OpenVLA-OFT/paper claims: false.

Loss and metric result:

- ActionMap head-only loss: `0.162408 -> 0.010239`, decreased true.
- TCA-Map head-only loss: `0.855555 -> 0.126224`, decreased true.
- ActionMap eval: standard proxy `0.434797`, action L1 `0.130406`, target top1 `0.5`, wrong-target proxy `0.5`, counterfactual margin `0.0`.
- TCA-Map eval: standard proxy `0.0`, action L1 `0.111259`, target top1 `0.0`, wrong-target proxy `1.0`, counterfactual margin `0.02313`.
- TCA-Map + Distributional TCA-Select eval matched TCA-Map in this run and added no measured gain.

Conclusion: this tiny exploratory offline proxy weakens TCA-Map under the fixed integrity policy. The action regression part improved action L1 slightly, but the target head failed on the holdout pair and worsened wrong-target proxy and standard proxy. Do not present this as support for TCA-Map or as paper-grade evidence.

## Tiny Offline LoRA Attribution Comparison Result

The required tiny LoRA attribution milestone has run on the same deterministic tiny split as the head-only ActionMap vs TCA-Map diagnostic.

Scope and safety:

- script: `scripts\53_compare_libero_offline_lora.ps1`,
- task-local gate: `ALLOW_TINY_TRAINING=1`,
- data source: `reports\libero_offline_counterfactual_split_report.json`,
- same split as head-only: true,
- samples: 8 records, 6 train / 2 eval,
- steps: 64,
- batch size: 1,
- device: CPU NumPy,
- training: true,
- LoRA training: true,
- rollout: false,
- downloads/GPU jobs/heavy imports/OpenVLA-OFT/paper claims: false.

Sanity checks passed: target labels aligned, wrong-target proxy range valid, target-conditioning input non-constant, TCA-Select candidate scores non-degenerate, no external verifier, and no privileged inference.

Loss and metric result:

- ActionMap + LoRA loss: `0.034182 -> 0.034104`, decreased true, LoRA params `84`.
- TCA-Map + LoRA loss: `0.716626 -> 0.656217`, decreased true, LoRA params `168`.
- ActionMap + LoRA eval: standard proxy `0.454351`, action L1 `0.091298`, target top1 `0.5`, wrong-target proxy `0.5`, counterfactual margin `0.012553`.
- TCA-Map + LoRA eval: standard proxy `0.0`, action L1 `0.082307`, target top1 `0.0`, wrong-target proxy `1.0`, counterfactual margin `0.019284`.
- TCA-Map + LoRA + Distributional TCA-Select matched TCA-Map + LoRA and added no measured gain.

Conclusion: `lora_weakens_tca_map`. LoRA did not rescue the weak TCA-Map head-only result on this tiny diagnostic. The current evidence points toward debugging target labeling/conditioning or revising the TCA-Map formulation before scaling.

## TCA Label/Conditioning Debug Audit Result

The execution-first TCA label/conditioning milestone has run on the same deterministic 8-sample split as the weak head-only and LoRA diagnostics.

Scope and safety:

- script: `scripts\55_debug_tca_label_conditioning.ps1`,
- task-local gate: `ALLOW_TINY_TRAINING=1`,
- audit artifact: `reports\libero_tca_label_conditioning_audit_table.json`,
- samples: 8 records, 6 train / 2 eval,
- steps: 64,
- batch size: 1,
- device: CPU NumPy,
- training: true, diagnostic tiny heads only,
- LoRA training: false,
- rollout: false,
- downloads/GPU jobs/heavy imports/OpenVLA-OFT/paper claims: false.

Invariant and metric audit result:

- target labels changed correctly for counterfactual target changes,
- expert action labels were aligned with intended target candidates,
- target-conditioning input was non-constant across counterfactual samples,
- one-hot target conditioning changed with target label,
- no all-zero, NaN, or identical target distributions were found,
- no silent shape broadcast or off-by-one label index was found,
- no train/eval candidate mismatch was found,
- wrong-target proxy direction was verified as lower-is-better,
- `wrong_target_proxy_rate=1.0` for TCA really meant both eval target predictions were wrong,
- recomputed ActionMap/TCA metrics matched the existing head-only report.

Sanity check result:

- one-sample TCA overfit passed: loss `0.849565 -> 0.024294`, standard proxy `0.999963`,
- TCA-Select scores were finite and non-degenerate, with no external verifier or privileged inference,
- target-shuffle negative control was inconclusive because baseline TCA eval target accuracy was already `0.0`,
- constant-target baseline matched or beat current TCA standard proxy: `0.442708` vs `0.0`,
- oracle-target TCA evaluation improved standard proxy from `0.0` to `0.86561` and wrong-target proxy from `1.0` to `0.0`.

Diagnosis: no concrete label-construction, metric-direction, off-by-one, broadcast, candidate-space, or TCA-Select scoring bug was found. The current failure is the TCA target classifier/generalization or target prior design on the tiny holdout pair: it predicts the wrong target for every eval sample, while the action branch can recover when given the oracle target.

Conclusion: `verified_no_label_or_metric_bug_but_target_classifier_failure`. Do not scale training, LoRA, or rollout to search for a positive result. The next execution-first milestone should revise/debug the TCA target-conditioning design on the same split before any broader experiment.

## TCA Target-Prior Rescue Diagnostic Result

The target-prior rescue milestone has run on the same deterministic 8-sample split.

Scope and safety:

- script: `scripts\56_debug_tca_target_prior_rescue.ps1`,
- task-local gate: `ALLOW_TINY_TRAINING=1`,
- samples: 8 records, 6 train / 2 eval,
- steps: 64,
- batch size: 1,
- device: CPU NumPy,
- training: true, diagnostic tiny heads only,
- LoRA training: false,
- rollout: false,
- downloads/GPU jobs/heavy imports/OpenVLA-OFT/paper claims: false.

Target-head sanity:

- target CE loss: `0.693147 -> 0.121261`, decreased true,
- train target top1 accuracy: `1.0`,
- eval target top1 accuracy: `0.0`,
- eval target top-k accuracy with `k=2`: `1.0`,
- eval confusion: `true_0__pred_1=1`, `true_1__pred_0=1`.

Variant results:

- learned hard target TCA: standard proxy `0.0`, wrong-target `1.0`, action-target consistency `0.0`.
- oracle-target TCA upper bound: standard proxy `0.86561`, wrong-target `0.0`, action-target consistency `0.86561`.
- soft target marginalization: correct target appeared in top-2, but standard proxy stayed `0.0`, wrong-target stayed `1.0`, and action L1 slightly worsened.
- soft target distributional selection: no improvement over hard target prediction.
- instruction-text prior TCA: standard proxy `0.86561`, wrong-target `0.0`, action-target consistency `0.86561`, matching the oracle target upper-bound on this tiny eval pair.
- constant target baseline: standard proxy `0.444499`, wrong-target `0.5`.

Diagnosis: target top-k contains the correct target, but hard top-1 target prediction fails. Soft marginalization over the current learned probabilities does not rescue TCA because the wrong target has too much probability mass. A simple instruction/candidate-text prior rescues the proxy to the oracle upper-bound on this tiny split, so the target-conditioned action mechanism remains viable as a diagnostic. The immediate blocker is the learned target prior/classifier, not the action-conditioning branch.

Conclusion: `target_topk_contains_correct_but_top1_prior_fails`. Next milestone should improve the target prior/classifier and rerun the same head-only ActionMap vs TCA-Map comparison before any scaling, LoRA, or rollout.

## Target-Prior-Fixed Head Comparison

Status: complete as an exploratory tiny offline proxy diagnostic on the same fixed 8-sample split.

Implementation:
- script: `scripts\57_compare_libero_target_prior_fixed_tca.ps1`
- module: `tca_map.datasets.libero_target_prior_fixed_head_comparison`
- report: `reports\libero_target_prior_fixed_head_comparison_report.json` and `.md` are runtime outputs and ignored by git.
- task-local gate: `ALLOW_TINY_TRAINING=1`

Execution boundaries:
- training happened: yes, tiny CPU NumPy head-only diagnostic training.
- loss was computed: yes.
- LoRA training happened: no.
- rollout happened: no.
- GPU jobs, downloads, heavy VLA imports, simulator execution, OpenVLA-OFT execution, token access, and paper claims did not occur.

Result on the fixed 8-sample split:
- ActionMap head-only loss: `0.162408 -> 0.010239`
- TCA head-only loss: `0.855555 -> 0.126224`
- ActionMap standard proxy: `0.434797`, wrong-target proxy: `0.5`
- hard learned-target TCA standard proxy: `0.0`, wrong-target proxy: `1.0`
- instruction-text-prior TCA standard proxy: `0.86561`, wrong-target proxy: `0.0`
- learned+text fusion standard proxy: `0.0`
- top-k uniform marginalization standard proxy: `0.476471`
- oracle-target TCA standard proxy: `0.86561`
- Distributional TCA-Select over the best non-oracle prior standard proxy: `0.86561`

Interpretation: the smallest target-prior fix recovers TCA-Map over ActionMap on standard proxy and wrong-target proxy, but only when using the instruction-text prior. The learned target head remains the bottleneck. Distributional TCA-Select adds no measurable gain over the best non-oracle prior in this diagnostic, so it should be revised before scaling or LoRA attribution is rerun.

## TCA-Select Target-Uncertainty Audit

Status: complete as an exploratory tiny offline proxy diagnostic on the same fixed 8-sample split.

Implementation:
- script: `scripts\58_audit_tca_select_target_uncertainty.ps1`
- module: `tca_map.datasets.libero_tca_select_uncertainty_audit`
- report: `reports\libero_tca_select_uncertainty_audit_report.json` and `.md` are runtime outputs and ignored by git.
- task-local gate: `ALLOW_TINY_TRAINING=1`

Execution boundaries:
- training happened: yes, same tiny CPU NumPy ActionMap/TCA heads as the prior diagnostics.
- loss was computed: yes.
- LoRA training happened: no.
- rollout happened: no.
- GPU jobs, downloads, heavy VLA imports, simulator execution, OpenVLA-OFT execution, token access, and paper claims did not occur.

Fusion audit result:
- no normalization, NaN, all-zero, constant-vector, class-id, or target-index-space bug was found.
- equal learned+text fusion failed because the learned target prior was confidently wrong and overwrote the correct instruction-text prior on both eval samples.
- fixed fusion used temperature-calibrated learned logits with a lower learned weight under learned/text conflict.
- fixed learned+text fusion recovered to standard proxy `0.86561`, wrong-target proxy `0.0`, matching instruction-text prior and oracle-target TCA on this tiny split.

TCA-Select uncertainty result:
- existing TCA-Select baseline with learned prior: standard proxy `0.0`, wrong-target proxy `1.0`.
- TCA-Select with learned prior: standard proxy `0.0`, wrong-target proxy `1.0`.
- TCA-Select with temperature-calibrated learned prior: standard proxy `0.0`, wrong-target proxy `1.0`.
- TCA-Select with top-k uniform prior: standard proxy `0.444499`, wrong-target proxy `0.5`; delta over non-select top-k uniform was only `+0.005128` standard proxy with no wrong-target improvement.
- TCA-Select with instruction-text prior: standard proxy `0.86561`, wrong-target proxy `0.0`; delta over non-select instruction-text prior was `0.0`.
- TCA-Select with fixed learned+text fusion: standard proxy `0.86561`, wrong-target proxy `0.0`; delta over non-select fixed fusion was `0.0`.

Interpretation: TCA-Select was revised to marginalize over target uncertainty, but it did not produce a meaningful gain. The only positive selector delta was a weak top-k uniform proxy bump without wrong-target improvement. The next execution milestone should rerun LoRA attribution with the fixed target prior, while de-emphasizing TCA-Select unless a future selector shows a real contribution beyond target-prior correctness.

## Fixed-Prior LoRA Attribution Result

Status: complete as an exploratory tiny offline proxy diagnostic on the same fixed 8-sample split.

Implementation:
- script: `scripts\130_compare_libero_fixed_prior_lora.ps1`
- module: `tca_map.datasets.libero_fixed_prior_lora_attribution`
- report: `reports\libero_fixed_prior_lora_attribution_report.json` and `.md` are runtime outputs and ignored by git.
- task-local gate: `ALLOW_TINY_TRAINING=1`

Execution boundaries:
- training happened: yes, tiny CPU NumPy LoRA attribution training.
- LoRA training happened: yes.
- loss was computed: yes.
- rollout happened: no.
- GPU jobs, downloads, heavy VLA imports, model loading, model inference, simulator execution, OpenVLA-OFT execution, token access, and paper claims did not occur.

Result on the fixed 8-sample split:
- ActionMap + LoRA loss: `0.034182 -> 0.034104`; trainable LoRA params: `84`; standard proxy: `0.454351`; wrong-target proxy: `0.5`; action-target consistency: `0.451948`; counterfactual margin: `0.012553`.
- TCA-Map + LoRA hard learned target loss: `0.716626 -> 0.656217`; trainable LoRA params: `168`; standard proxy: `0.0`; wrong-target proxy: `1.0`.
- TCA-Map + LoRA instruction-text prior loss: `0.716626 -> 0.656217`; standard proxy: `0.910293`; wrong-target proxy: `0.0`; action-target consistency: `0.910293`; counterfactual margin: `0.009565`.
- TCA-Map + LoRA fixed learned+text fusion loss: `0.716626 -> 0.656217`; standard proxy: `0.910293`; wrong-target proxy: `0.0`; gap to oracle: `0.0`.
- Oracle-target TCA + LoRA upper bound: standard proxy `0.910293`; wrong-target proxy `0.0`.
- TCA-Select fixed-fusion ablation: standard proxy `0.910293`; wrong-target proxy `0.0`; delta over fixed-fusion non-select TCA: `0.0`.

Interpretation: fixed-prior TCA + LoRA beats ActionMap + LoRA by `+0.455942` standard proxy and `-0.5` wrong-target proxy on this tiny exploratory split. This keeps TCA-Map viable when the target prior is corrected, and improves over the previous hard-learned LoRA result by `+0.910293` standard proxy. The learned target head remains the bottleneck. TCA-Select still has no measurable gain and should be de-emphasized unless a future selector ablation shows nontrivial benefit. This is not paper-grade evidence.

## Scaled Fixed-Prior Offline Comparison

Status: complete as an exploratory offline proxy scale-up from 8 to 16 samples.

Implementation:
- script: `scripts\134_compare_libero_fixed_prior_offline_scale.ps1`
- module: `tca_map.datasets.libero_fixed_prior_offline_scale_comparison`
- report: `reports\libero_fixed_prior_offline_scale_comparison_report.json` and `.md` are runtime outputs and ignored by git.
- task-local gate: `ALLOW_TINY_TRAINING=1`

Execution boundaries:
- training happened: yes, CPU NumPy head-only and LoRA diagnostics.
- LoRA training happened: yes.
- loss was computed: yes.
- rollout happened: no.
- GPU jobs, downloads, heavy VLA imports, model loading, model inference, simulator execution, OpenVLA-OFT execution, token access, and paper claims did not occur.

Expanded split:
- records: `16`
- train/eval records: `12 / 4`
- task count: `4`
- target class count: `2`
- target balance: `{0: 8, 1: 8}`
- split: deterministic manifest-order pair holdout, exploratory.

Key results:
- ActionMap head-only: loss `0.031336 -> 0.001623`, standard proxy `0.484449`, wrong-target proxy `0.5`.
- fixed-prior TCA head-only: loss `0.724483 -> 0.459717`, standard proxy `0.944121`, wrong-target proxy `0.0`.
- hard learned-target TCA head-only: standard proxy `0.0`, wrong-target proxy `1.0`.
- ActionMap + LoRA: loss `0.034701 -> 0.034603`, standard proxy `0.427546`, wrong-target proxy `0.5`.
- fixed-prior TCA + LoRA: loss `0.727824 -> 0.726609`, standard proxy `0.854`, wrong-target proxy `0.0`.
- hard learned-target TCA + LoRA: standard proxy `0.642331`, wrong-target proxy `0.25`.
- oracle-target TCA + LoRA: standard proxy `0.854`, wrong-target proxy `0.0`.
- TCA-Select fixed-fusion ablation: standard proxy `0.854`, wrong-target proxy `0.0`, delta over fixed-prior TCA + LoRA `0.0`.

Interpretation: the fixed-prior TCA advantage over ActionMap survives this smallest larger split, including under the required LoRA attribution arm. This remains exploratory offline proxy evidence only. The learned target head remains a bottleneck, and TCA-Select should be de-emphasized as a core contribution unless a future scaled selector diagnostic shows nontrivial gain.

## Multi-Seed Fixed-Prior Offline Validation

Status: complete as exploratory offline proxy validation on the same fixed 16-sample split.

Implementation:
- script: `scripts\135_validate_libero_fixed_prior_multiseed.ps1`
- module: `tca_map.datasets.libero_fixed_prior_multiseed_validation`
- report: `reports\libero_fixed_prior_multiseed_validation_report.json` and `.md` are runtime outputs and ignored by git.
- task-local gate: `ALLOW_TINY_TRAINING=1`

Execution boundaries:
- training happened: yes, CPU NumPy head-only and LoRA diagnostics.
- LoRA training happened: yes.
- loss was computed: yes.
- rollout happened: no.
- GPU jobs, downloads, heavy VLA imports, model loading, model inference, simulator execution, OpenVLA-OFT execution, token access, and paper claims did not occur.

Seed policy:
- seeds: `11, 23, 37, 53, 71`
- split: exactly the same 16 records, `12 / 4` train/eval, unchanged across seeds.
- seed effect: CPU head-only SGD order and LoRA low-rank initialization only.

Aggregate results:
- ActionMap head-only standard proxy mean/std: `0.484044 / 0.000571`; wrong-target mean/std: `0.5 / 0.0`; loss mean: `0.031336 -> 0.001617`.
- fixed-prior TCA head-only standard proxy mean/std: `0.944914 / 0.000893`; wrong-target mean/std: `0.0 / 0.0`; loss mean: `0.724483 -> 0.460224`.
- ActionMap + LoRA standard proxy mean/std: `0.427817 / 0.001523`; wrong-target mean/std: `0.5 / 0.0`; loss mean: `0.031888 -> 0.031824`.
- fixed-prior TCA + LoRA standard proxy mean/std: `0.854615 / 0.004512`; wrong-target mean/std: `0.0 / 0.0`; loss mean: `0.726098 -> 0.725057`.
- hard learned-target TCA + LoRA standard proxy mean/std: `0.470181 / 0.084772`; wrong-target mean/std: `0.45 / 0.1`.
- fixed-prior TCA + LoRA beat ActionMap + LoRA in `5 / 5` seeds.
- fixed-prior TCA + LoRA improved wrong-target proxy in `5 / 5` seeds.
- TCA-Select showed nontrivial gain in `0 / 5` seeds.
- LoRA hurt fixed-prior TCA relative to fixed-prior head-only in `5 / 5` seeds, with mean standard-proxy delta `-0.090299`.

Interpretation: fixed-prior TCA advantage over ActionMap is stable across these five bounded seeds on the 16-sample split. LoRA should be treated as a robustness/attribution ablation, not a performance claim, because it consistently underperforms fixed-prior head-only TCA here. Hard learned-target TCA remains unstable/weaker. TCA-Select should be de-emphasized or killed as a core contribution unless future evidence changes.

## 32-Record Multi-Seed Fixed-Prior Offline Validation

Status: complete as exploratory offline proxy validation on a larger deterministic local LIBERO split.

Implementation:
- split config: `configs\libero_offline_counterfactual_split_scaled.yaml`
- split builder: `scripts\51_build_libero_offline_counterfactual_split.ps1`
- validation script: `scripts\135_validate_libero_fixed_prior_multiseed.ps1`
- module: `tca_map.datasets.libero_fixed_prior_multiseed_validation`
- runtime reports are ignored by git: `reports\libero_offline_counterfactual_split_scaled_report.json`, `.md`, `reports\libero_fixed_prior_multiseed_validation_report.json`, and `.md`.
- task-local gate: `ALLOW_TINY_TRAINING=1`

Execution boundaries:
- training happened: yes, CPU NumPy head-only and LoRA diagnostics.
- LoRA training happened: yes.
- loss was computed: yes.
- rollout happened: no.
- GPU jobs, downloads, heavy VLA imports, model loading, model inference, simulator execution, OpenVLA-OFT execution, token access, and paper claims did not occur.

Scaled split:
- source: existing local `libero_10` HDF5/counterfactual/offline data only.
- full scaled manifest capacity: `32` pairs / `64` records.
- executed split: `32` records, `24 / 8` train/eval.
- task count: `10`.
- target class count: `2`.
- target balance: `{0: 16, 1: 16}`.
- seed policy: same 32-record split for all seeds; seeds vary only CPU SGD order and LoRA low-rank initialization.
- seeds: `11, 23, 37, 53, 71`.

Aggregate results:
- ActionMap head-only: loss mean `0.035916 -> 0.01322`; standard proxy mean/std `0.456326 / 0.00059`; wrong-target mean/std `0.5 / 0.0`.
- fixed-prior TCA head-only: loss mean `0.729063 -> 0.400294`; standard proxy mean/std `0.925186 / 0.00243`; wrong-target mean/std `0.0 / 0.0`.
- hard learned-target TCA head-only: standard proxy mean/std `0.901054 / 0.048982`; wrong-target mean/std `0.025 / 0.05`.
- ActionMap + LoRA: loss mean `0.036285 -> 0.036235`; standard proxy mean/std `0.429068 / 0.001081`; wrong-target mean/std `0.5 / 0.0`.
- fixed-prior TCA + LoRA: loss mean `0.729591 -> 0.727671`; standard proxy mean/std `0.858448 / 0.004741`; wrong-target mean/std `0.0 / 0.0`.
- hard learned-target TCA + LoRA: standard proxy mean/std `0.429496 / 0.180547`; wrong-target mean/std `0.5 / 0.209165`.
- oracle-target TCA + LoRA: standard proxy mean/std `0.858448 / 0.004741`; wrong-target mean/std `0.0 / 0.0`.
- TCA-Select fixed-fusion LoRA ablation: standard proxy mean/std `0.858448 / 0.004741`; wrong-target mean/std `0.0 / 0.0`.

Aggregate interpretation:
- fixed-prior TCA head-only advantage over ActionMap head-only mean/std: `0.46886 / 0.002045`.
- fixed-prior TCA + LoRA advantage over ActionMap + LoRA mean/std: `0.429379 / 0.003737`.
- fixed-prior TCA + LoRA beat ActionMap + LoRA in `5 / 5` seeds.
- fixed-prior TCA + LoRA improved wrong-target proxy in `5 / 5` seeds.
- LoRA hurt fixed-prior TCA relative to fixed-prior head-only in `5 / 5` seeds, mean delta `-0.066738`.
- TCA-Select showed nontrivial gain in `0 / 5` seeds.

Conclusion: fixed-prior TCA advantage survives cautious scaling from 16 to 32 records on this exploratory offline proxy split. This is still not standard success, not rollout evidence, and not paper-grade evidence. The target-prior mechanism remains central; LoRA remains a required attribution/fairness ablation rather than a performance claim; TCA-Select should stay de-emphasized or be killed as a central contribution unless future evidence changes.

## 64-Record Multi-Seed Fixed-Prior Offline Validation

Status: complete as exploratory offline proxy validation on the full deterministic scaled local LIBERO split.

Implementation:
- split config: `configs\libero_offline_counterfactual_split_scaled.yaml`
- split builder: `scripts\51_build_libero_offline_counterfactual_split.ps1`
- validation script: `scripts\135_validate_libero_fixed_prior_multiseed.ps1`
- module: `tca_map.datasets.libero_fixed_prior_multiseed_validation`
- runtime reports are ignored by git: `reports\libero_offline_counterfactual_split_scaled_report.json`, `.md`, `reports\libero_fixed_prior_multiseed_validation_report.json`, and `.md`.
- task-local gate: `ALLOW_TINY_TRAINING=1`

Execution boundaries:
- training happened: yes, CPU NumPy head-only and LoRA diagnostics.
- LoRA training happened: yes.
- loss was computed: yes.
- rollout happened: no.
- GPU jobs, downloads, heavy VLA imports, model loading, model inference, simulator execution, OpenVLA-OFT execution, token access, and paper claims did not occur.

Scaled split:
- source: existing local `libero_10` HDF5/counterfactual/offline data only.
- executed split: `64` records, `48 / 16` train/eval.
- task count: `10`.
- target class count: `2`.
- target balance: `{0: 32, 1: 32}`.
- seed policy: same 64-record split for all seeds; seeds vary only CPU SGD order and LoRA low-rank initialization.
- seeds: `11, 23, 37`.

Aggregate results:
- ActionMap head-only: loss mean `0.042488 -> 0.015347`; standard proxy mean/std `0.460175 / 0.000317`; wrong-target mean/std `0.5 / 0.0`.
- fixed-prior TCA head-only: loss mean `0.735635 -> 0.482645`; standard proxy mean/std `0.921973 / 0.000983`; wrong-target mean/std `0.0 / 0.0`.
- hard learned-target TCA head-only: standard proxy mean/std `0.460371 / 0.000271`; wrong-target mean/std `0.5 / 0.0`.
- ActionMap + LoRA: loss mean `0.041479 -> 0.041452`; standard proxy mean/std `0.429293 / 0.001702`; wrong-target mean/std `0.5 / 0.0`.
- fixed-prior TCA + LoRA: loss mean `0.735888 -> 0.735426`; standard proxy mean/std `0.856646 / 0.002967`; wrong-target mean/std `0.0 / 0.0`.
- hard learned-target TCA + LoRA: standard proxy mean/std `0.374657 / 0.230835`; wrong-target mean/std `0.5625 / 0.270031`.
- oracle-target TCA + LoRA: standard proxy mean/std `0.856646 / 0.002967`; wrong-target mean/std `0.0 / 0.0`.
- TCA-Select fixed-fusion LoRA ablation: standard proxy mean/std `0.856646 / 0.002967`; wrong-target mean/std `0.0 / 0.0`.

Aggregate interpretation:
- fixed-prior TCA head-only advantage over ActionMap head-only mean/std: `0.461798 / 0.000798`.
- fixed-prior TCA + LoRA advantage over ActionMap + LoRA mean/std: `0.427353 / 0.002126`.
- fixed-prior TCA + LoRA beat ActionMap + LoRA in `3 / 3` seeds.
- fixed-prior TCA + LoRA improved wrong-target proxy in `3 / 3` seeds.
- LoRA hurt fixed-prior TCA relative to fixed-prior head-only in `3 / 3` seeds, mean delta `-0.065327`.
- TCA-Select showed nontrivial gain in `0 / 3` seeds.

Prior-source audit:
- fixed learned+text fusion is available at test time under the current offline proxy interface, does not use BDDL metadata at inference, and does not use eval labels. Its learned component is trained with train-split target labels, while inference uses instruction-text prior plus learned target logits.
- hard learned-target TCA uses learned target logits from instruction-derived features and train-split target labels for supervision; it does not use eval labels or BDDL metadata at inference.
- oracle-target TCA uses ground-truth target labels, is not available at test time, and remains an upper bound only.

Conclusion: fixed-prior TCA advantage survives the 64-record exploratory offline proxy split across three bounded seeds. This remains offline proxy evidence only, not standard success, rollout evidence, or paper-grade evidence. The target-prior mechanism remains central; LoRA remains a required attribution/fairness ablation and still underperforms fixed-prior head-only TCA; TCA-Select should be killed as a core contribution unless a future explicitly targeted selector stress test produces a nontrivial gain.

## Publishability Gate Audit After 64-Record Validation

Status: complete as a bounded exploratory offline proxy audit before learned target-head redesign.

Implementation:
- script: `scripts\136_audit_publishability_gate.ps1`
- module: `tca_map.datasets.libero_publishability_gate_audit`
- runtime reports are ignored by git: `reports\libero_publishability_gate_audit_report.json` and `.md`
- task-local gate: `ALLOW_TINY_TRAINING=1`

Execution boundaries:
- training happened: yes, CPU NumPy offline audit reconstruction.
- LoRA training happened: yes.
- loss was computed: yes.
- rollout happened: no.
- GPU jobs, downloads, heavy VLA imports, model loading, model inference, simulator execution, OpenVLA-OFT execution, token access, and paper claims did not occur.

Prior-source audit:
- instruction-text prior classification: `A_valid_test_time_semantic_prior`.
- fixed learned+text fusion classification: `A_valid_test_time_semantic_prior`.
- inference uses BDDL metadata: false.
- inference uses dataset target labels: false.
- inference uses eval labels: false.
- inference uses task id, filename, or manifest target field as target proxy: false.
- inference uses information unavailable at test time: false, under the explicit assumption that candidate/task natural-language text is available at test time.
- fixed fusion does use train-split target labels to train the learned target head, but not eval labels or dataset target labels at inference.
- oracle target remains `C_oracle_like_upper_bound`.

Per-task/per-target audit:
- evaluated split: `64` records, `48 / 16` train/eval, `10` total tasks, `9` eval task groups, target balance `{0: 32, 1: 32}`.
- fixed-prior TCA + LoRA beat ActionMap + LoRA across all seeds on `8 / 9` eval task groups.
- the task group where fixed-prior TCA did not beat ActionMap across all seeds was `put both moka pots on the stove`, with mean standard-proxy delta `-0.005339` and no wrong-target change.
- target `0`: fixed-prior TCA + LoRA standard proxy `0.839149` vs ActionMap + LoRA `0.839247`, mean delta `-0.000098`, wrong-target delta `0.0`.
- target `1`: fixed-prior TCA + LoRA standard proxy `0.878226` vs ActionMap + LoRA `0.0`, mean delta `+0.878226`, wrong-target delta `-1.0`.

Selector headroom audit:
- current TCA-Select selection turnover rate: `0.0`.
- oracle selector upper-bound standard proxy: `0.858687 / 0.003971` mean/std.
- oracle selector delta over non-select fixed-prior TCA: `0.0`.
- mean unique candidate count: `2.0`.
- candidate collapse rate: `0.0`.
- score diversity was non-degenerate: score range mean `0.488254`, score variance mean `0.063864`.
- correct candidates had slightly higher separation than wrong candidates: mean margin `0.001837`.

Conclusion: the fixed prior is not downgraded as leakage under the explicit test-time candidate-text assumption, but the gain is not broad across every target/task group. The improvement is dominated by target `1` and wrong-target correction. TCA-Select has no oracle-selector headroom in this audit and should be killed as a core contribution unless future targeted selector evidence changes this. The next milestone should be learned target-head or target-prior robustness redesign before limited rollout.

## Representation Sensitivity and Target-Reinjection Audit

Status: complete as a bounded exploratory offline proxy audit on the existing 64-record split.

Implementation:
- script: `scripts\137_audit_representation_sensitivity.ps1`
- module: `tca_map.datasets.libero_representation_sensitivity_audit`
- runtime reports are ignored by git: `reports\libero_representation_sensitivity_audit_report.json` and `.md`
- task-local gate: `ALLOW_TINY_TRAINING=1`

Execution boundaries:
- training happened: yes, CPU NumPy offline audit reconstruction.
- LoRA training happened: yes.
- loss was computed: yes.
- representation extraction happened: proxy only; full VLA hidden-state extraction did not happen.
- rollout happened: no.
- GPU jobs, downloads, heavy VLA imports, model loading, model inference, simulator execution, OpenVLA-OFT execution, token access, and paper claims did not occur.

Representation sensitivity result:
- evaluated split: `64` records, `48 / 16` train/eval, seeds `11, 23, 37`.
- full hidden extraction: false; blocked intentionally by the no-heavy-import/model-load policy.
- proxy representation source: cached `hidden_tokens`, not final SmolVLA/OpenVLA hidden states.
- target-swapped eval pair count: `8`.
- target-swap proxy cosine mean/std: `-0.094343 / 0.288315`.
- target-swap proxy L2 mean/std: `3.071515 / 0.491991`.
- target-preserving paraphrase pairs were unavailable, so no paraphrase-normalized representation sensitivity ratio is claimed.

Action pathway and target-prior reinjection result:
- ActionMap + LoRA standard proxy mean: `0.429275`; wrong-target proxy mean: `0.5`.
- fixed-prior TCA + LoRA standard proxy mean: `0.856612`; wrong-target proxy mean: `0.0`.
- fixed-prior TCA + LoRA advantage over ActionMap + LoRA: `+0.427337` standard proxy and `-0.5` wrong-target proxy.
- hard learned-target TCA + LoRA standard proxy mean: `0.464418`; wrong-target proxy mean: `0.458333`, with high seed variance.
- TCA-Select delta over fixed-prior TCA + LoRA: `0.0`.
- target `0` diagnosis: `near_saturation_or_metric_noise_not_a_material_failure`; fixed-prior TCA + LoRA delta `-0.003352` mean with wrong-target delta `0.0`.
- target `1` remains the main wrong-target correction gain.

Conclusion: this audit does not support a representation-collapse claim because full hidden states were not extracted. The supported claim is narrower and cleaner: fixed semantic target-prior reinjection improves wrong-target correction/action-pathway grounding on the offline proxy split without proving hidden collapse. Target-Prior TCA-Map remains the main method under the explicit non-leaking candidate-text assumption. TCA-Select should be killed or de-emphasized as a core contribution unless future evidence changes this. The next milestone may be a limited fixed-prior rollout diagnostic under a separate green rollout risk gate.

## Fixed-Prior Rollout Readiness Gate

Status: complete as a bounded readiness gate after the representation-sensitivity audit.

Implementation:
- script: `scripts\138_gate_fixed_prior_rollout_readiness.ps1`
- module: `tca_map.datasets.libero_fixed_prior_rollout_readiness`
- runtime reports are ignored by git: `reports\libero_fixed_prior_rollout_readiness_gate_report.json` and `.md`

Execution boundaries:
- rollout happened: no; the gate was red.
- training happened: no.
- LoRA training happened: no.
- loss was computed: no.
- GPU jobs, downloads, heavy VLA imports, model loading, model inference, simulator environment creation, OpenVLA-OFT execution, token access, and paper claims did not occur.

Gate result:
- risk gate status: `red`.
- rollout diagnostic authorized: `false`.
- simulator plumbing status: green; existing import, render, reset/step, and zero-action LIBERO/RoboSuite diagnostic reports are present/readable and passed.
- target-prior source status: green under the existing assumption; fixed learned+text fusion remains `A_valid_test_time_semantic_prior`, candidate/task natural-language text is available, and inference uses no BDDL metadata, eval labels, dataset target labels, task IDs, filenames, or manifest target fields as target proxies.
- action bridge status: red; current offline proxy records use `4D` actions from `ACTION_PREFIX_DIM=4`, while LIBERO env actions are `7D`.
- existing validated adapter error: `unsupported action dimension mapping: policy_dim=4, env_action_dim=7`.
- gripper mapping: unresolved.
- rotation/coordinate mapping: unresolved.
- camera/state mapping: not a live visual/state policy; the current fixed-prior proxy uses text-derived cached features and mean HDF5 action snippets.

Conclusion: do not run the limited rollout diagnostic yet. The smallest direct unblocker is a 7D rollout-action bridge for fixed-prior proxy outputs, preferably by rebuilding the offline ActionMap/TCA rollout records to preserve all seven LIBERO action dimensions and validating gripper, rotation, action scale, clipping, and coordinate conventions against local HDF5 before simulator stepping.

## Zero-Reward Rollout Diagnosis And Expert Replay Sanity

Status: complete as a bounded rollout diagnostic after the fixed-prior 7D action bridge milestone.

Implementation:
- script: `scripts\140_zero_reward_rollout_diagnosis.ps1`
- module: `tca_map.datasets.libero_zero_reward_rollout_diagnosis`
- targeted tests: `tests\test_libero_zero_reward_rollout_diagnosis.py`
- runtime reports are ignored by git: `reports\zero_reward_rollout_diagnosis_report.json` and `.md`
- task-local gate: `ALLOW_ZERO_REWARD_ROLLOUT_DIAGNOSIS=1`

Execution boundaries:
- rollout happened: yes, bounded diagnostic only.
- training happened: no.
- LoRA training happened: no.
- loss was computed: no.
- GPU jobs, downloads, heavy VLA imports, model loading, model inference, OpenVLA-OFT execution, token access, and paper claims did not occur.

Diagnostic result:
- task count: `1`.
- task: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`.
- horizons: `10`, `25`, `50`.
- variants: zero action, ActionMap-style target-agnostic mean, HDF5 expert replay, fixed semantic target-prior TCA proxy.
- total simulator steps: `340`.
- reward/success: all variants had reward `0.0` and success `false` through 50 steps.
- HDF5 expert metadata: first positive reward / done index is `271`, so zero reward through 50 steps is expected for this demonstration.
- action bridge: `7D` env actions, finite, no clipping, gripper command preserved from the HDF5 actions.
- fixed-prior proxy actions are identical to HDF5 expert replay actions for this diagnostic.
- intended object metric: available through the matched key `moka_pot_1_pos`.
- wrong-target object metric: unavailable because the counterfactual black-bowl target was not present in the current environment observation object-position keys.
- target movement at 50 steps: zero action moved slightly farther from the matched object; ActionMap-style mean moved closer by `0.140269`; HDF5 expert/fixed-prior moved closer by `0.009014`.

Conclusion: the current blocker is classified as `sparse_reward_or_short_horizon`, with target-directed movement still ambiguous under the naive object-distance metric. This does not support a rollout success claim, and it does not yet show fixed-prior target-directed movement advantage. The next execution-first milestone should be a separately bounded full-demo expert replay sanity check up to the first positive reward/done index for one task. If expert replay succeeds at the expected index, then rerun the fixed-prior diagnostic with a justified horizon. If expert replay fails, investigate init-state, action convention, gripper, rotation, and coordinate mapping before any learned-policy rollout scaling.

## Full-Demo Expert Replay Sanity

Status: complete as a bounded rollout diagnostic after zero-reward diagnosis classified the earlier 50-step result as sparse reward / short horizon.

Implementation:
- script: `scripts\141_full_demo_expert_replay_sanity.ps1`
- module: `tca_map.datasets.libero_full_demo_expert_replay_sanity`
- targeted tests: `tests\test_libero_full_demo_expert_replay_sanity.py`
- runtime reports are ignored by git: `reports\full_demo_expert_replay_sanity_report.json` and `.md`
- task-local gate: `ALLOW_FULL_DEMO_EXPERT_REPLAY=1`

Execution boundaries:
- rollout happened: yes, bounded diagnostic only.
- full-demo expert replay happened: yes.
- training happened: no.
- LoRA training happened: no.
- loss was computed: no.
- GPU jobs, downloads, heavy VLA imports, model loading, model inference, OpenVLA-OFT execution, token access, and paper claims did not occur.

Diagnostic result:
- task count: `1`.
- task: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`.
- variants: exact-init zero action, exact-init HDF5 expert replay, default-reset HDF5 expert replay.
- total simulator steps: `805`.
- HDF5 first positive reward/done index: `271`.
- observed exact-init expert first reward/done/success index: `260`.
- exact-init expert replay reward/success: reward sum `1.0`, final success `true`.
- exact-init zero action reward/success: reward sum `0.0`, final success `false`.
- default-reset expert replay reward/success: reward sum `0.0`, final success `false`.
- exact init-state application: `env.set_init_state(init_state)` with simulator state L2 to HDF5 init `0.0`.
- action convention: raw `7D` HDF5 actions reached reward/done/success. Actions were finite, in expected range, and clipping rate was `0.0`.
- controller evidence: controller type reported as `OSC_POSE`; MuJoCo timestep reported as `0.002`.
- gripper evidence: gripper command range `[-1.0, 1.0]`, mean `-0.220588`, first positive index `62`.
- target progress: exact-init expert replay reduced EEF distance to `moka_pot_1_pos` by `0.267427` and moved the matched object by `0.217868`.
- object-pose comparison to HDF5 obs: unavailable because the HDF5 `obs` group has no object-position keys, only camera/proprio-style keys.

Conclusion: the action convention/init-state bridge is green for longer-horizon method rollout only under matched HDF5 init-state diagnostic conditions. Default reset is not sufficient for this replay. The next execution-first milestone may be a bounded longer-horizon fixed-prior method rollout using matched init states and a validated horizon around the expert success window. This remains diagnostic rollout evidence only, not standard success or paper-grade evidence.

## Action-Source Audit And Matched-Init Candidate-Replay Diagnostic

Status: complete as a bounded action-source legitimacy audit after full-demo expert replay sanity made the matched-init bridge green.

Implementation:
- script: `scripts\142_action_source_audit_matched_init_diagnostic.ps1`
- module: `tca_map.datasets.libero_action_source_audit_matched_init_diagnostic`
- targeted tests: `tests\test_libero_action_source_audit_matched_init_diagnostic.py`
- runtime reports are ignored by git: `reports\action_source_audit_matched_init_diagnostic_report.json` and `.md`
- task-local gate: `ALLOW_ACTION_SOURCE_AUDIT_ROLLOUT=1`

Execution boundaries:
- rollout happened: yes, bounded diagnostic only.
- action-source audit happened: yes.
- training happened: no.
- LoRA training happened: no.
- loss was computed: no.
- GPU jobs, downloads, heavy VLA imports, model loading, model inference, OpenVLA-OFT execution, token access, and paper claims did not occur.

Action-source audit:
- zero action: programmatic diagnostic control; no future HDF5 action use; mean L2 to expert `1.128492712`.
- HDF5 expert replay: copied from HDF5 expert action at the same timestep; future expert action use; near-match rate to expert `1.0`; mean L2 to expert `0.0`.
- ActionMap-style rollout action: mean aggregate of positive and counterfactual HDF5 future action sequences; future HDF5 action use; near-match rate to expert `0.0`; mean L2 to expert `0.716695147`.
- fixed-prior TCA rollout action: copied from HDF5 expert action at the same timestep; future HDF5 expert action use; near-match rate to expert `1.0`; mean L2 to expert `0.0`.
- hard learned-target TCA proxy: selected from the offline counterfactual HDF5 candidate set; future HDF5 action use; near-match rate to expert `0.0`; mean L2 to expert `1.433390294`.

Matched-init diagnostic result:
- task: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`.
- horizon: requested up to the expert success window, with early stop at reward/done/success.
- total simulator steps: `1305`.
- zero action: reward sum `0.0`, success `false`, target-distance change `+0.000429`.
- HDF5 expert replay: reward sum `1.0`, success `true`, first reward/done index `260`, target-distance change `-0.267427`.
- ActionMap-style mean: reward sum `0.0`, success `false`, target-distance change `-0.010547`.
- fixed-prior TCA candidate replay: reward sum `1.0`, success `true`, first reward/done index `260`, target-distance change `-0.267427`.
- hard learned-target proxy: reward sum `0.0`, success `false`, target-distance change `-0.005729`.
- wrong-target object movement: unavailable because the counterfactual object is not present in the environment object-position keys.

Conclusion: fixed-prior TCA has no valid closed-loop rollout-level support from this diagnostic because the successful fixed-prior actions are exactly the future HDF5 expert actions. The result is useful as candidate-replay / action-bridge evidence only. A valid rollout method claim now requires an online action-generation bridge or a non-leaking action decoder that does not copy future HDF5 expert actions. Otherwise, paper materials must label rollout evidence with this caveat.

## Online Action-Generation Bridge Diagnostic

Status: completed as a bounded diagnostic.

Key result: source inventory found no deployable online 7D ActionMap/TCA action source. Native SmolVLA can produce an online 6D action and map it explicitly to 7D, but that is a native baseline rather than TCA-Map rollout support.

Bounded rollout result: exact-init native SmolVLA ran for 25 steps on one validated LIBERO/RoboSuite task. Reward and success stayed `0.0 / false`; the action sequence did not match future HDF5 expert actions, so it is valid online baseline evidence but not method success. HDF5 expert replay also stayed at `0.0 / false` at this short horizon.

Current blocker: `no_nonleaking_online_actionmap_tca_7d_head`.

## Online 7D Diagnostic Head Result

Status: complete as a bounded execution-first diagnostic after the online action-generation bridge blocker.

Implementation:
- script: `scripts\144_online_7d_diagnostic_head.ps1`
- module: `tca_map.smolvla.online_7d_diagnostic_head`
- targeted tests: `tests\test_online_7d_diagnostic_head.py`
- runtime reports are ignored by git: `reports\online_7d_diagnostic_head_report.json` and `.md`
- task-local gate for rollout: `ALLOW_ONLINE_7D_DIAGNOSTIC_HEAD_ROLLOUT=1`

Execution boundaries:
- training happened: yes, CPU ridge/linear diagnostic heads only.
- LoRA training happened: no.
- loss was computed: yes.
- rollout happened: yes, bounded matched-init diagnostic only.
- heavy model import/model load/model inference happened: yes, only for the native SmolVLA baseline inside the gated rollout.
- GPU jobs, downloads, OpenVLA-OFT execution, full fine-tuning, benchmark rollouts, token access, and paper claims did not occur.

Offline head result:
- training data source: local LIBERO HDF5 training demos only; the rollout demo path was filtered out of training labels.
- train/eval: `512 / 25` timestep samples.
- ActionMap-7D loss: `0.179330387 -> 0.02388843`; offline 7D L2 `1.000304358`; standard offline proxy `0.499923922`.
- fixed-prior TCA-7D loss: `0.179330387 -> 0.0238874`; offline 7D L2 `0.995906943`; standard offline proxy `0.501025363`.
- hard learned-target TCA-7D loss: `0.179330387 -> 0.023892769`; offline 7D L2 `1.016720219`; standard offline proxy `0.495854601`.

Bounded rollout result:
- task: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`.
- horizon: `25` exact-init steps per variant.
- variants: zero action, HDF5 expert upper bound, native SmolVLA online policy, ActionMap-7D, fixed-prior TCA-7D, hard learned-target TCA-7D.
- reward/success: all variants stayed at `0.0 / false`.
- method action provenance: ActionMap-7D, fixed-prior TCA-7D, and hard learned-target TCA-7D generated online 7D actions from current observation/instruction and did not use same/future HDF5 actions at inference.
- fixed-prior TCA valid rollout support: `false`.
- fixed-prior TCA partial target-movement support: `true`.
- blocker classification: `online_7d_head_partial_target_movement_no_success`.

Conclusion: the non-leaking online 7D ActionMap/TCA diagnostic head path is now implemented and runnable, but it does not support a rollout success claim. The fixed-prior TCA head shows only partial target-movement evidence and no reward/success improvement. The next execution-first milestone should diagnose online action quality/head training before scaling rollout.

## Online 7D Action-Quality Diagnosis

Status: complete as a bounded execution-first diagnostic after the minimal online 7D head milestone.

Implementation:
- script: `scripts\145_online_7d_action_quality_diagnosis.ps1`
- module: `tca_map.smolvla.online_7d_action_quality_diagnosis`
- targeted tests: `tests\test_online_7d_action_quality_diagnosis.py`
- runtime reports are ignored by git: `reports\online_7d_action_quality_diagnosis_report.json` and `.md`

Execution boundaries:
- training happened: yes, CPU ridge/linear diagnostic heads only.
- LoRA training happened: no.
- loss was computed: yes.
- rollout happened: yes, only because the prior bounded 25-step online 7D matched-init report was regenerated for closed-loop failure diagnosis.
- heavy model import/model load/model inference happened: yes, only inside that regenerated native SmolVLA baseline report.
- GPU jobs, downloads, OpenVLA-OFT execution, full fine-tuning, benchmark rollouts, token access, and paper claims did not occur.

Key diagnosis:
- ActionMap-7D vs fixed-prior TCA-7D mean action L2: `0.00712081`.
- fixed-prior TCA actions meaningfully different from ActionMap: `false`.
- supervised 25-step 7D L2: ActionMap `0.992624014`, fixed-prior TCA `0.988163728`, hard learned-target TCA `1.007243003`.
- mean-action baseline 7D L2: `0.57299313`, better than all current learned 7D heads.
- teacher-forced full-demo fixed-prior TCA delta vs ActionMap: `-0.001041313`, a tiny improvement only.
- gripper open/close timing is poor over the full demo: fixed-prior TCA predicted first open at step `100` while expert first open is step `62`.
- closed-loop reward/success remains `0.0 / false`; fixed-prior TCA valid rollout-level support remains `false`.
- dominant bottleneck: `translation`, with gripper timing also problematic over the full demonstration.

Conclusion: the current 7D head does not yet justify rollout scaling. The fixed-prior target signal barely changes the generated 7D actions, and the learned heads underperform a simple mean-action baseline on the held-out 25-step supervised diagnostic. The next milestone should redesign target-prior conditioning and/or head features before another method rollout.

## Bounded 7D Action-Head Redesign Gate

Status: complete as a bounded execution-first diagnostic after the online 7D action-quality diagnosis.

Implementation:
- script: `scripts\146_online_7d_head_redesign_gate.ps1`
- module: `tca_map.smolvla.online_7d_head_redesign_gate`
- targeted tests: `tests\test_online_7d_head_redesign_gate.py`
- runtime reports are ignored by git: `reports\online_7d_head_redesign_gate_report.json` and `.md`

Execution boundaries:
- training happened: yes, bounded CPU diagnostic heads only.
- LoRA training happened: no.
- loss was computed: yes.
- rollout happened: no; the offline rollout gate was red.
- GPU jobs, downloads, heavy VLA imports, model loading/inference, OpenVLA-OFT execution, full fine-tuning, benchmark rollouts, token access, and paper claims did not occur.

Key diagnosis:
- train/eval/teacher-forced samples: `512 / 25 / 272`.
- mean-action baseline eval 7D L2: `0.57299313`.
- best redesigned non-mean head: `small_cpu_mlp_fixed_prior_tca_7d`.
- best redesigned eval 7D L2: `0.669078005`, so it still does not beat the mean-action baseline.
- best fixed-prior TCA head beats best ActionMap head on eval 7D L2 (`0.669078005` vs `0.992624014`), and the best TCA/ActionMap action difference is meaningful, but the result is not rollout-ready because the mean baseline remains stronger.
- teacher-forced best non-mean 7D L2: `1.114676933`, still worse than mean-action baseline `1.091252901`.
- dominant bottleneck remains `translation`; gripper classification/calibration is also fragile for some variants.

Conclusion: the bounded redesign improved fixed-prior TCA over ActionMap in offline 7D metrics, but it failed the required rollout gate because no non-mean head beat the train-split mean-action baseline by the documented threshold. Do not run another method rollout from this head. The next milestone should redesign the target-prior conditioning/action features or move to a paper-readiness package that honestly states the rollout caveat.

## CSS-Shield Project Initialization

Status: initialized as the new Option 1 research direction.

The previous low-compute Target-Prior TCA-Map route remains killed for RA-L-stable submission. CSS-Shield is a new rollout-first project focused on runtime semantic/safety intervention for VLA manipulation.

Core claim target: a lightweight counterfactual semantic safety shield can reduce wrong-target and unsafe VLA actions while preserving useful task behavior.

Next state: minimal rollout-first safety diagnostic. It must produce simulator/rollout safety metrics or a concrete blocker, not another offline-only proxy package.

## CSS-Shield State 1 Minimal Rollout Diagnostic

Status: complete as a bounded rollout-first diagnostic.

Execution boundary:
- rollout happened: yes.
- proposal source: native SmolVLA.
- training happened: no.
- LoRA training happened: no.
- loss computed: no, because this was not a training task.
- downloads/GPU/OpenVLA-OFT/benchmark rollout/paper claim: no.

Key result:
- full CSS-Shield unsafe-rate reduction versus no shield: `0.8`.
- full CSS-Shield unsafe-rate reduction versus clipping-only: `0.8`.
- full CSS-Shield unsafe-rate reduction versus safety-only: `0.0`.
- wrong-target reduction: `0.0`, because the counterfactual object was not present as an observation object key.
- reward/success: `0.0 / false` for all variants.

Conclusion: CSS-Shield is not killed at the no-rollout-metric gate, but the State 1 evidence is safety-damping evidence rather than semantic shielding evidence. The next milestone should be a narrow semantic-coverage diagnostic where both intended and counterfactual targets are observable and safety-only is an insufficient baseline.
