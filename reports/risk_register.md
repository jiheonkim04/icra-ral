# Risk Register

## Missing SmolVLA Checkpoint Files

Risk: `C:\assets\checkpoints\smolvla` exists but lacks config/tokenizer/weights files.

Impact: Blocks adapter-smoke readiness.

Mitigation: Follow `reports/smolvla_manual_acquisition_checklist.md`, place real checkpoint files manually, or use the explicitly approved SmolVLA-only acquisition source `lerobot/smolvla_base`, then rerun readiness checks.

## Wrong Checkpoint Source

Risk: Accidentally downloading from an unapproved mirror, private fork, OpenVLA-OFT repository, dataset repository, or unrelated Hugging Face model.

Impact: Invalid readiness state, unexpected file formats, accidental large downloads, or policy drift into unapproved assets.

Mitigation: The approved SmolVLA source is only `lerobot/smolvla_base`. Stop if login/token access is required, if the source redirects to an ambiguous asset, or if the command attempts OpenVLA-OFT, LIBERO, RoboSuite, RoboCasa, dataset, GPU, training, rollout, heavy import, or inference behavior.

## External Tokenizer Dependency

Risk: The acquired `lerobot/smolvla_base` files reference `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` for tokenizer/model support.

Impact: Config and weights can be present while readiness remains false if the external dependency is missing or is not recognized by the checker.

Mitigation: The dependency acquisition was separately approved for tokenizer/processor/config files only. The checker now recognizes the dependency under `HF_HOME`. Do not silently download full SmolVLM2 model weights or execute inference. Heavy import/model construction is allowed only inside the bounded SmolVLA autonomous pilot load-only task.

## Hidden Processor Python Dependencies

Risk: SmolVLA can pass file readiness checks but fail during local policy construction because the referenced SmolVLM processor imports small Python utilities not listed in the original runtime gate.

Impact: Load-only smoke fails late with an import error after assets and core runtime packages appear ready.

Mitigation: The bounded load-only debug path identified `num2words==0.5.14` as required. It is now listed in `requirements.txt`, `scripts\17_check_smolvla_runtime_deps.ps1`, `scripts\18_plan_smolvla_runtime_install.ps1`, and `scripts\27_summarize_hard_stop_status.ps1`.

## Manual Confirmation Drift

Risk: Codex may waste turns asking the user to confirm routine state that can be checked automatically.

Impact: Slower progress, stale context, and avoidable human coordination overhead.

Mitigation: Follow the self-check gate policy in `reports/codex_delegation_manual.md`. Codex should run existing checkers, branch behavior on checker output, update state docs when useful, and ask only at dangerous gates.

## Windows / CUDA / PyTorch Compatibility

Risk: Local Windows environment may have CUDA or PyTorch compatibility issues, especially with newer RTX 5080 hardware.

Impact: Heavy imports or future GPU inference may fail even if file checks pass.

Mitigation: Keep current checks lightweight. Require risk assessment before heavy import or GPU inference, and proceed only if source/setup, runtime, RAM/VRAM, and policy budget are green. Prefer WSL2/Linux for simulator or heavier training work.

## RTX 5080 16GB VRAM

Risk: 16GB VRAM may be insufficient for larger VLA models, large heatmaps, full-resolution voxel heads, or non-quantized baselines.

Impact: OOM during future load-only smoke, feature caching, or pilots.

Mitigation: SmolVLA-first, frozen/head-only defaults, low-resolution heatmaps, batch size 1, memory estimates, required LoRA track only under the tiny-smoke budget, and QLoRA feasibility only when tooling and memory allow.

## LoRA Memory Risk

Risk: Required LoRA arms may exceed the RTX 5080 16GB budget once adapters, optimizer state, activations, and tokenizer/VLM dependencies are included.

Impact: OOM, unstable Windows/CUDA behavior, or pressure to loosen the compute policy.

Mitigation: Require memory estimates before LoRA smoke, batch size 1, max 100 smoke steps, max 200 samples, max 30 minutes, max 14GB VRAM target, and stop if QLoRA needs package/CUDA/PyTorch changes or cloud resources.

## LoRA API Mismatch Risk

Risk: LeRobot, Transformers, PEFT-style APIs, or local SmolVLA modules may not expose stable target module names for adapter insertion.

Impact: Adapter construction fails or silently attaches to the wrong modules.

Mitigation: Add a planning-only adapter construction checker before training. Require explicit module allowlists such as target fusion layers, action head projection, and small adapter layers.

## PEFT / Transformers / LeRobot Compatibility Risk

Risk: PEFT, Transformers, LeRobot, and local PyTorch/CUDA versions may be incompatible for LoRA/QLoRA.

Impact: Import errors, quantization failures, incorrect parameter freezing, or environment drift.

Mitigation: Treat package/CUDA/PyTorch changes as risk-assessed gates, with system-wide changes and large unplanned installs still external stop gates. Prefer check-only guards and tiny construction smoke before any LoRA training.

## LoRA Attribution Risk

Risk: LoRA gains may dominate TCA-Map gains, making the target-conditioned heatmap and Distributional TCA-Select contribution unclear.

Impact: The paper claim weakens or becomes a generic PEFT result.

Mitigation: Always compare ActionMap + LoRA vs TCA-Map + LoRA, and TCA-Map + LoRA vs TCA-Map + LoRA + Distributional TCA-Select. Report LoRA gain, target-conditioning gain, and selection gain separately.

## Unbounded Heavy Import

Risk: A risk-assessed load-only smoke task may accidentally become model inference, training, rollout, or GPU-heavy execution outside the bounded scope.

Impact: CUDA/Windows instability, OOM, hidden inference, or invalid claim that readiness is a result.

Mitigation: Set `ALLOW_HEAVY_IMPORT=1` only inside the bounded SmolVLA load-only task. Enforce no inference, no training, no rollout, no dataset evaluation, no simulator import, no OpenVLA-OFT, max 10 minutes for load-only, and max 14GB VRAM.

## Single-Sample Interface Scope Creep

Risk: A synthetic interface smoke may be mistaken for real evaluation or expanded into repeated inference.

Impact: Invalid claims, unapproved workload, or drift toward rollout/dataset evaluation.

Mitigation: Require `ALLOW_SINGLE_SAMPLE_INFERENCE=1` only inside the bounded task. Run one synthetic sample only, CPU by default, no dataset, no simulator, no rollout, no training, no OpenVLA-OFT, max 10 minutes, and max 14GB VRAM.

## SmolVLA Runtime Dependency Drift

Risk: Local files are ready and runtime packages are installed now, but later package upgrades or CUDA/PyTorch changes could break the SmolVLA load-only path.

Impact: Load-only execution may fail, or CUDA behavior may change unexpectedly on Windows.

Mitigation: Keep package versions recorded in `reports\project_state.md` and re-run `scripts\17_check_smolvla_runtime_deps.ps1`. Any future package upgrade, CUDA toolkit change, or PyTorch change requires risk assessment; system-wide or very large unplanned changes remain external stop gates.

## Unpinned Runtime Upgrade

Risk: Upgrading PyTorch, LeRobot, Transformers, or Safetensors without pinned versions can break the current environment or mismatch the RTX 5080/CUDA stack.

Impact: Failed model load, CUDA errors, dependency conflicts, or a hard-to-reproduce local setup.

Mitigation: Use `reports\smolvla_runtime_dependency_plan.md` and `scripts\17_check_smolvla_runtime_deps.ps1`. Run risk assessment before changing packages, capture environment state before and after, and validate with the safe runner. Stop for system-wide or very large unplanned package changes.

## Accidental Runtime Install During Planning

Risk: A planning task for SmolVLA runtime packages may accidentally become a package installation or CUDA/PyTorch change.

Impact: Broken local environment, CUDA mismatch, unexpected downloads, or blocked load-only validation.

Mitigation: Use `scripts\18_plan_smolvla_runtime_install.ps1` and `reports\smolvla_runtime_install_request.md`. The planner refuses dangerous gates and records that installs, downloads, heavy imports, model loads, inference, training, rollouts, and OpenVLA-OFT execution were not performed.

## Runtime-Ready Misread As Result-Ready

Risk: Runtime dependency readiness or load-only success may be mistaken for a research result.

Impact: The project could overclaim from engineering smoke tests.

Mitigation: Keep load-only, single-sample interface, feature-cache, and tiny head-only training smoke reports labeled as smoke/interface checks only. No paper-level empirical claim is allowed automatically; paper claims remain external stop gates.

## Feature Cache Contract Drift

Risk: Later SmolVLA feature extraction may produce records that do not match the head-only training interface.

Impact: Head-only pilots fail late or silently train on inconsistent metadata.

Mitigation: Use `reports\feature_cache_interface_plan.md`, `tca_map.features.cache`, and `scripts\19_plan_feature_cache.ps1` to validate manifest and JSONL schema with dummy features before any real extraction.

## Cached-Feature Consumer Drift

Risk: Cached features may be valid on disk but unusable by TCA-Map heads or offline metric code.

Impact: The first head-only pilot fails after expensive feature extraction or produces invalid proxy metrics.

Mitigation: Run `scripts\25_eval_feature_cache_smoke.ps1 -PrepareDummyCache` and `tests\test_feature_cache_eval_smoke.py` to validate the consumer path with dummy cached features before real SmolVLA extraction.

## Accidental Tiny Training Scope Creep

Risk: A risk-assessed tiny head-only smoke may drift into longer training, real benchmark evaluation, or GPU-heavy experimentation.

Impact: Unapproved GPU use, invalid local results, or policy drift beyond the bounded autopilot session.

Mitigation: Require bounded local pilot runners to run only after a green risk assessment and task-local gates such as `ALLOW_TINY_TRAINING=1`, use cached/dummy/synthetic/tiny local non-paper data, enforce max 300 steps after stable smaller smoke, max 200 samples, max 30 minutes, max 14GB VRAM, refuse OpenVLA and paper-claim gates, and make no paper claim. Stop if the assessment is ambiguous or outside budget.

## LoRA Tiny Smoke Scope Creep

Risk: The required LoRA track may drift from a bounded adapter smoke into real training, model loading, GPU-heavy work, or a paper claim.

Impact: Unapproved compute use, invalid local evidence, or confusion between LoRA adaptation gains and the TCA-Map / Distributional TCA-Select contribution.

Mitigation: LoRA execution is autonomous only for risk-assessed bounded local pilots. A future execution runner must require task-local `ALLOW_TINY_TRAINING=1`, train adapter weights only, freeze the backbone, keep max 300 steps after stable smaller smoke, max 200 samples, max 30 minutes, max 14GB VRAM, and always compare ActionMap + LoRA against TCA-Map + LoRA.

## LoRA Attribution Risk

Risk: LoRA adaptation may dominate the observed gains and make TCA-Map or Distributional TCA-Select look incidental.

Impact: The method claim becomes unclear and weakens publishability.

Mitigation: Require the comparison matrix in `reports\lora_comparison_plan.md`: TCA-Map + LoRA vs ActionMap + LoRA, and TCA-Map + LoRA + Distributional TCA-Select vs TCA-Map + LoRA only. Report head gain, LoRA gain, and inference-time selection gain separately.

## Offline Proxy Overinterpretation

Risk: A tiny ActionMap vs TCA-Map offline proxy comparison may be mistaken for standard success or paper-grade evidence.

Impact: Misleading claims and poor research decisions.

Mitigation: `scripts\36_compare_head_only_tiny_pilot.ps1` labels the output as offline proxy only, reports `not_standard_success=true` and `not_paper_grade=true`, and avoids SOTA or paper-grade language.

## Tiny LoRA Smoke Overinterpretation

Risk: The bounded tiny LoRA smoke may be mistaken for evidence that LoRA improves real SmolVLA behavior.

Impact: Misleading method claims and premature paper conclusions.

Mitigation: `scripts\37_tiny_lora_smoke.ps1` uses cached/dummy features only, reports `offline_proxy_only=true`, `not_standard_success=true`, and `not_paper_grade=true`, and keeps future real-data or rollout claims behind risk assessment and paper-claim stop gates.

## Tiny LoRA Comparison Overinterpretation

Risk: The tiny LoRA comparison report may be mistaken for a real adapter benchmark.

Impact: LoRA or TCA-Select contributions could be overstated before real data or rollout validation.

Mitigation: `scripts\38_compare_tiny_lora_pilot.ps1` reads only cached/dummy offline proxy smoke outputs, refuses execution gates, labels the report as not standard success and not paper-grade, and keeps all real benchmark claims behind risk assessment and paper-claim stop gates.

## Consolidated Status Overinterpretation

Risk: A consolidated local pilot status report may be mistaken for a paper-ready result.

Impact: The project could move to claims before simulator rollouts, real benchmark data, or stronger baselines exist.

Mitigation: `scripts\39_generate_local_pilot_status.ps1` labels the report as summary-only, offline proxy only, not standard success, and not paper-grade. It lists explicit hard-stop boundaries for the next stage.

## QLoRA Tooling Drift

Risk: QLoRA may require bitsandbytes/PEFT behavior, CUDA support, or PyTorch compatibility that is not stable on native Windows.

Impact: The local environment could be destabilized by package or CUDA/PyTorch changes before the method is ready.

Mitigation: Keep `scripts\35_check_qlora_feasibility.ps1` check-only. Defer QLoRA execution to Linux/WSL/cloud if tooling is missing or Windows support is uncertain, and never install packages or change CUDA/PyTorch without a green risk assessment.

## Autonomous Scope Confusion

Risk: The risk-assessed autonomous policy may be misread as permission for OpenVLA-OFT, paper claims, token/license gates, or unbounded training/rollouts.

Impact: A bounded local smoke could turn into an invalid experiment or paper claim.

Mitigation: Use `scripts\27_summarize_hard_stop_status.ps1`, `scripts\31_generate_go_no_go_report.ps1`, `scripts\39_generate_local_pilot_status.ps1`, `reports\codex_delegation_manual.md`, and `reports\go_no_go_status.md`. Autonomy covers only tasks whose risk assessment is inside budget. Stop for ambiguous/out-of-budget assessments, OpenVLA-OFT execution, token/secret/payment/license gates, system-level changes, external irreversible actions, and paper-level claims.

## Risk Assessment Drift

Risk: The new risk-assessed autonomous policy may be interpreted as permission to run any download, GPU job, training run, dataset setup, simulator task, or rollout without first checking concrete budgets.

Impact: Disk exhaustion, unstable local environment, invalid claims, or accidental crossing into OpenVLA-OFT, token/license, system-level, or paper-claim territory.

Mitigation: Before any bounded download/GPU/training/dataset/simulator/rollout step, Codex must write or print a risk assessment covering source, expected size, target path, disk free before/after estimate, runtime, RAM/VRAM, budget, official/documented source status, token/license/payment requirements, decision, and reason. Proceed only if the decision is `proceed`; stop if ambiguous or outside budget.

## 24GB System RAM

Risk: System RAM can become a bottleneck during dataset loading, simulator setup, or feature caching.

Impact: Slowdowns, crashes, or failed preprocessing.

Mitigation: tiny subsets, streaming/cached features, no full dataset sweeps locally, and cloud/remote handoff for larger runs.

## OpenVLA-OFT OOM

Risk: OpenVLA-OFT large work may exceed local VRAM/RAM.

Impact: OOM or unstable local machine.

Mitigation: Keep large local OpenVLA-OFT forbidden. Use only separately approved frozen/load smoke locally; move paper-grade baseline work to larger GPU resources.

## Simulator / WSL2 Risk

Risk: LIBERO/RoboSuite/simulator stack may not work reliably on native Windows.

Impact: Rollout metrics blocked.

Mitigation: Treat Windows as planning/readiness path. Use WSL2/Linux checks before simulator work. Do not run rollouts until simulator paths and checks pass.

## Baseline Reproducibility

Risk: ActionMap, native VLA, OpenVLA-OFT, and augmentation baselines may differ from published settings.

Impact: Weak or non-comparable paper claims.

Mitigation: Track configs, compute budgets, trainable parameters, latency, VRAM, and exact baseline scope. Avoid SOTA claims without strong baseline reproduction.

## SOTA Claim Risk

Risk: Overclaiming state-of-the-art from a low-compute or offline-only pilot.

Impact: Paper rejection or misleading results.

Mitigation: Restrict claims to low-compute target-conditioned action decoding/counterfactual robustness. Require ActionMap/OpenVLA-OFT-level baselines before stronger claims.

## Custom Benchmark Risk

Risk: Improvements may only hold on a custom counterfactual split.

Impact: Poor generality.

Mitigation: Include standard LIBERO-style subsets, counterfactual target-swap or LIBERO-CF-style splits, nuisance/paraphrase checks, and later simulator rollouts.

## Privileged Inference Risk

Risk: Accidentally using simulator state or labels at default inference.

Impact: Invalid method comparison.

Mitigation: Keep default inference free of privileged state. Use simulator labels only for supervision, metrics, or explicit oracle ablations.

## LIBERO Dataset Source Ambiguity

Risk: A LIBERO/LIBERO-CF-style dataset task could start from an ambiguous source, unknown size, missing local paths, or token/license/payment requirement.

Impact: Unsafe downloads, invalid offline proxy evidence, blocked rollout setup, or accidental paper-grade framing.

Mitigation: Use `scripts\42_plan_libero_dataset_risk.ps1` first. Proceed only with an official/documented source, known expected size, enough disk margin, no token/license/payment/license-click gate, or an already-present tiny local subset under `LIBERO_DATA_ROOT`.

## Simulator Readiness Scope Creep

Risk: A simulator readiness task could accidentally become MuJoCo/RoboSuite/LIBERO import, render smoke, rollout, policy execution, or paper-grade benchmark evaluation.

Impact: Native Windows instability, missing dependency failures, unapproved rollout work, or invalid standard-success claims.

Mitigation: Use `scripts\43_plan_simulator_readiness.ps1` first. It is planning-only and reports path/WSL2/Linux readiness without importing simulators or executing rollouts. A later simulator import-smoke task must be separate, bounded, WSL2/Linux-oriented, and still no-rollout unless a rollout risk assessment passes.

## Local Pilot Step Budget Drift

Risk: Config files, compute budget checks, and autonomy policy could disagree on whether local pilots are capped at 300 or 1000 steps.

Impact: A bounded smoke could accidentally become a longer local training run than the current risk policy allows.

Mitigation: Keep `configs\compute_budget.yaml`, head-only pilot configs, and planning docs aligned to `max_steps<=300` for bounded local pilots. Tiny smoke runners may enforce stricter caps.

## Bounded Local Pilot Extension Overinterpretation

Risk: A longer cached-feature local pilot extension could be mistaken for real benchmark evidence or standard success.

Impact: Premature method claims, invalid comparison language, or drift into real dataset training without simulator/benchmark readiness.

Mitigation: `scripts\44_bounded_local_pilot_extension.ps1` uses cached/dummy features only, keeps the runner cap at 100 steps, reports offline proxy labels, and keeps real dataset training, simulator execution, rollout, OpenVLA-OFT, and paper claims behind separate risk gates.

## LIBERO Full Dataset Size Risk

Risk: The official LIBERO demonstrations dataset is documented through the official LIBERO repo and Hugging Face, but the dataset card reports about 100 GB total file size.

Impact: Automatic full-dataset acquisition could consume substantial disk and still fail before a tiny/offline interface smoke is ready.

Mitigation: Only `https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets` may use the dedicated 180 GB LIBERO-only acquisition budget, and only if at least 250 GB disk remains after acquisition. `scripts\49_acquire_libero_data.ps1` must run the source/disk/license/token risk gate and writes ignored reports. It must stop on ambiguous source, token/login/payment/license click-through, expected size over 180 GB, disk-after below 250 GB, simulator/rollout/OpenVLA-OFT requirements, or repeated acquisition failure.

## LIBERO Dataset Commit Risk

Risk: Downloaded LIBERO demonstrations or Hugging Face cache files could accidentally be committed or copied into the repository.

Impact: Repository bloat, license/distribution problems, and accidental publication of local assets.

Mitigation: Keep all LIBERO data under `C:\assets\data\libero` and cache under `C:\assets\hf_home`. Never commit `C:\assets`, dataset files, cache files, or `configs\paths.local.yaml`. Runtime acquisition reports are ignored by git.

## Source Repo Setup Scope Creep

Risk: A safe LIBERO/RoboSuite code checkout task could drift into simulator installation, simulator import, render smoke, rollout, training, or benchmark evaluation.

Impact: Native Windows instability, hidden MuJoCo/RoboSuite work, or accidental paper-grade framing.

Mitigation: `scripts\46_prepare_libero_robosuite_sources.ps1` is source-repo setup only. It may shallow-clone official code repos with task-local `ALLOW_DOWNLOADS=1` and create a data root marker, but it does not install packages, import simulators, render, rollout, train, run GPU jobs, import heavy VLA models, execute OpenVLA-OFT, access tokens, or make paper claims.

## LIBERO Path-Ready Misread As Rollout-Ready

Risk: Creating `LIBERO_ROOT`, `LIBERO_DATA_ROOT`, and `ROBOSUITE_ROOT` can make paths exist even when no real demonstration files are present and no simulator import/render/rollout has run.

Impact: A checker could incorrectly report rollout readiness from directories alone.

Mitigation: `scripts\11_check_real_assets.ps1` now separates `ready_for_libero_path_check`, `libero_dataset_files_present`, and `ready_for_libero_rollout`. Empty data roots or marker-only roots are path-ready only, not rollout-ready.

## LIBERO Metadata-Only Overclaim Risk

Risk: A metadata-only LIBERO task/counterfactual manifest could be mistaken for real offline dataset evaluation.

Impact: Invalid evidence claims, premature comparison language, or hidden drift toward training without demonstration files.

Mitigation: `scripts\47_build_libero_metadata_subset.ps1` labels outputs as metadata-only, refuses execution gates, reports `ready_for_real_dataset_interface_smoke=false` when demo files are absent, and keeps all offline dataset smoke, simulator execution, rollout, and paper claims behind later risk gates.

## LIBERO Offline Interface Gate Drift

Risk: A structural offline file check could drift into real dataset training or be described as standard success.

Impact: Invalid evidence, accidental training, or premature claims from file-format readiness alone.

Mitigation: `scripts\48_plan_libero_offline_interface_smoke.ps1` is check-only, refuses execution gates, reports rollout readiness as false, and labels its result as not standard success and not paper-grade evidence. It proceeds only when a tiny local file has readable instruction/action-like fields.

## LIBERO HDF5 Reader Dependency Risk

Risk: The official LIBERO data may be present, but HDF5 inspection can fail if `h5py` is missing or incompatible.

Impact: Offline real-data interface smoke remains blocked even though the dataset is acquired; ad hoc dependency installation could accidentally change broader environment packages.

Mitigation: Use `scripts\50_check_libero_hdf5_reader.ps1` first. If `h5py` is missing, perform a separate dependency risk assessment before installing `h5py>=3.11` from the Python package index. Locally, `h5py 3.16.0` was installed from a Windows wheel after a green risk assessment without CUDA/PyTorch changes or simulator packages. Future environments must keep the same guard: do not change CUDA/PyTorch versions, install simulator stacks, train, rollout, import heavy VLA models, execute OpenVLA-OFT, or make paper claims.

## LIBERO HDF5 Report Size Risk

Risk: Inspecting real LIBERO HDF5 files can dump thousands of dataset paths and shapes into terminal output and runtime reports.

Impact: Routine safe checks become noisy, slow to review, and harder to diagnose.

Mitigation: `tca_map.datasets.libero_offline_interface.inspect_hdf5` records bounded samples: `dataset_count`, `dataset_sample_limit`, `datasets_sample`, and `action_dataset_paths_sample`. It keeps action-field readiness while avoiding full HDF5 tree dumps.

## LIBERO Counterfactual Split Overclaim Risk

Risk: A tiny HDF5-backed counterfactual split could be mistaken for benchmark evaluation or rollout evidence.

Impact: Offline proxy plumbing could be overinterpreted as standard success, counterfactual success, or paper-grade evidence.

Mitigation: `scripts\51_build_libero_offline_counterfactual_split.ps1` is check-only and manifest-only. It reads local BDDL metadata and HDF5 structure, sets `offline_proxy_only=true`, and keeps training, simulator execution, rollouts, heavy VLA imports, OpenVLA-OFT, token access, and paper claims forbidden.

## LIBERO Offline Head Comparison Overclaim Risk

Risk: A deterministic HDF5 action proxy comparison could be mistaken for a trained ActionMap/TCA-Map result.

Impact: The project could overstate offline plumbing as model performance or paper-grade evidence.

Mitigation: `scripts\52_compare_libero_offline_actionmap_tca.ps1` labels outputs as offline proxy only, not standard success, not paper-grade, and not trained. It refuses execution gates and keeps downloads, GPU jobs, training, simulator execution, rollouts, heavy VLA imports, OpenVLA-OFT, token access, and paper claims forbidden.

## LIBERO Offline LoRA Proxy Overinterpretation Risk

Risk: The required LIBERO offline LoRA comparison may be mistaken for a full SmolVLA LoRA adapter result.

Impact: Tiny NumPy adapter smoke over HDF5 snippets could be overstated as real model adaptation, rollout success, or paper-grade evidence.

Mitigation: `scripts\53_compare_libero_offline_lora.ps1` requires `ALLOW_TINY_TRAINING=1`, trains only tiny NumPy low-rank adapter matrices, sets `offline_proxy_only=true`, and reports `not_standard_success=true` and `not_paper_grade=true`. It keeps GPU jobs, heavy model imports, model loading, model inference, rollouts, simulator execution, OpenVLA-OFT, token access, downloads, and paper claims forbidden.

## LIBERO Offline Bounded Pilot Report Overclaim Risk

Risk: A consolidated local pilot report could be mistaken for paper-grade benchmark evidence.

Impact: Offline proxy summaries could be incorrectly described as standard LIBERO success or rollout success.

Mitigation: `scripts\54_generate_libero_offline_bounded_pilot_report.ps1` is summary-only and labels outputs as offline proxy, not standard success, not rollout success, and not paper-grade. The report keeps simulator readiness, rollout, and paper claims behind separate risk gates.

## LIBERO Rollout Readiness Overstatement

Risk: Asset checkers could mark `ready_for_libero_rollout=true` merely because LIBERO, RoboSuite, and data paths exist.

Impact: A path/data-ready state could be mistaken for permission to run simulator rollouts.

Mitigation: `scripts\11_check_real_assets.ps1` and hard-stop summaries keep `ready_for_libero_rollout=false` unless a separate simulator import/render/rollout risk gate is implemented and passes. Path/data readiness is reported separately.

## Simulator Readiness Status Blind Spot

Risk: The simulator readiness planner could run and produce a clear stop/proceed decision, but the consolidated local pilot or go/no-go status reports might omit that decision.

Impact: The project could repeat already-cleared planning steps or misread the current blocker before simulator import/render/rollout work.

Mitigation: `scripts\39_generate_local_pilot_status.ps1` and `scripts\31_generate_go_no_go_report.ps1` now read `reports\simulator_readiness_plan_report.json` when present. They expose platform, path readiness, import-smoke readiness, render-smoke readiness, rollout readiness, warnings, and stop reasons while remaining summary-only.

## Simulator Import Smoke Scope Creep

Risk: A bounded import-only smoke could drift into rendering, creating or stepping simulator environments, rolling out policies, installing simulator packages, or claiming benchmark readiness.

Impact: Native/WSL simulator instability, unapproved rollout work, dependency drift, or invalid standard-success claims.

Mitigation: `scripts\55_bounded_simulator_import_smoke.ps1` requires task-local `ALLOW_SIMULATOR_IMPORT_SMOKE=1`, reruns the planning gate, imports only `robosuite` and `libero`, and keeps render smoke, rollouts, environment steps, installs, downloads, GPU jobs, OpenVLA-OFT execution, token access, and paper claims false.

## WSL Simulator Dependency Gap

Risk: WSL paths and Python can be present while the WSL Python environment lacks basic simulator dependencies such as `numpy`.

Impact: Simulator import smoke fails before any render or rollout gate, and ad hoc dependency installation could drift into a larger simulator stack install.

Mitigation: Use `scripts\56_check_wsl_simulator_deps.ps1` before any install. Missing WSL simulator dependencies are handled by the WSL simulator dependency ladder standing approval: after a green risk assessment, Codex may install only minimal WSL Python packaging tools, create or reuse `~/.venvs/tca_map_sim`, install minimal import-readiness Python dependencies, and rerun import smoke. The current local venv has cleared bounded import-only readiness for `robosuite` and `libero` without sudo or apt. Stop for sudo password, token/login, license/payment, CUDA/driver/toolkit, graphics-stack, Windows driver, OpenVLA-OFT, paper-claim, or oversized package/download gates. Render smoke, reset/step smoke, and tiny rollout diagnostics remain separate risk gates.

## WSL Dependency Bootstrap Scope Creep

Risk: A minimal WSL dependency bootstrap could drift into broad apt installs, system graphics changes, MuJoCo license workarounds, CUDA/PyTorch replacements, or simulator rollout attempts.

Impact: The local machine or WSL environment could become unstable, and the project could cross into unapproved benchmark or paper-grade work.

Mitigation: The standing approval is narrow. Allowed apt packages are `python3-pip`, `python3-venv`, `python3-dev` if needed, `build-essential` only for Python package builds, `git` if missing, and `curl` or `wget` only for official setup checks. Every package install, render smoke, reset/step smoke, or tiny diagnostic rollout must print/write a risk assessment with command, WSL distro/version, expected size, target path, disk estimate, runtime, RAM/VRAM, sudo status, token/license/payment status, CUDA/driver/graphics impact, decision, and reason.

## Simulator Render/Reset Planner Overinterpretation

Risk: A green render/reset-step planning report could be mistaken for render readiness, reset/step readiness, rollout readiness, or benchmark evidence.

Impact: The project could move from import-only readiness into simulator execution or paper claims without the separate bounded render and reset/step gates.

Mitigation: `scripts\58_plan_simulator_render_reset.ps1` is planning-only. It refuses execution gates, requires a passed import-only report, keeps `render_smoke_performed=false`, `reset_step_smoke_performed=false`, `rollouts_performed=false`, and keeps `ready_for_rollout=false`. Render smoke, reset/step smoke, and tiny diagnostic rollout must each use a separate risk assessment and task-local gate.

## WSL Simulator Source-Link Drift

Risk: Linking local RoboSuite/LIBERO source checkouts into WSL could be mistaken for permission to install broad simulator packages, create a repo-local venv, download assets, or run environments.

Impact: The local Windows/WSL setup could drift away from the documented reproducible path or accidentally cross into simulator execution.

Mitigation: `scripts\60_link_wsl_simulator_sources.ps1` is limited to the existing `/home/jiheon/.venvs/tca_map_sim` venv, local source checkouts, `--no-index --no-deps --no-build-isolation`, a LIBERO `.pth` entry, and noninteractive WSL `~/.libero/config.yaml`. It reports no downloads, no repo-local venv creation, no render, no reset/step, no rollout, no GPU job, no training, no heavy VLA import, no OpenVLA-OFT execution, no token access, and no paper claim.

## WSL Offscreen Render Overinterpretation

Risk: A passing tiny MuJoCo OSMesa render smoke could be mistaken for LIBERO/RoboSuite reset readiness, rollout readiness, benchmark success, or paper-grade evidence.

Impact: The project could jump from a tiny render-only proof to environment stepping or policy evaluation without the separate simulator safety checks.

Mitigation: `scripts\59_bounded_simulator_render_smoke.ps1` performs only a tiny MuJoCo offscreen render and does not create, reset, or step LIBERO/RoboSuite environments. Reset/step smoke and tiny diagnostic rollout remain separate task-local gated risk assessments, and `ready_for_rollout` remains false until those gates pass.

## Simulator Reset/Step Overinterpretation

Risk: A passing tiny MuJoCo reset/step smoke could be mistaken for LIBERO/RoboSuite environment readiness, policy rollout readiness, benchmark success, or paper-grade evidence.

Impact: The project could move from physics plumbing into rollout or benchmark claims before LIBERO/RoboSuite env reset, policy control, and success metrics are separately validated.

Mitigation: `scripts\61_bounded_simulator_reset_step_smoke.ps1` uses only a tiny in-memory MuJoCo XML model, caps steps to 5, and reports `libero_robosuite_env_created=false`, `rollouts_performed=false`, `policy_inference_performed=false`, and `ready_for_rollout=false`. Tiny diagnostic rollout requires a separate green risk assessment and task-local `ALLOW_TINY_ROLLOUT=1`.

## Rollout Planner Overinterpretation

Risk: A green tiny rollout envelope could be mistaken for benchmark rollout readiness or paper-grade evidence.

Impact: The project could overclaim a toy MuJoCo diagnostic as LIBERO/RoboSuite success, standard success, or method evidence.

Mitigation: `scripts\62_plan_tiny_diagnostic_rollout.ps1` remains planning-only and refuses execution gates. `scripts\63_bounded_tiny_diagnostic_rollout.ps1` requires task-local `ALLOW_TINY_ROLLOUT=1`, caps execution at 5 toy tasks, 1 episode each, 5 steps each, and reports benchmark rollout readiness and paper-claim readiness as false.

## Bounded Tiny Diagnostic Rollout Overinterpretation

Risk: A passing bounded tiny diagnostic rollout could be mistaken for LIBERO/RoboSuite benchmark rollout success.

Impact: Simulator plumbing evidence could be described as standard success, counterfactual success, or paper-grade evidence.

Mitigation: The bounded rollout runner uses only toy in-memory MuJoCo models, no learned policy, no VLA inference, no LIBERO/RoboSuite benchmark environment, no training, no GPU job, no OpenVLA-OFT, no multi-seed run, and no paper claim. Reports must label it as tiny diagnostic simulator plumbing only.

## RoboSuite / MuJoCo API Drift

Risk: LIBERO expects RoboSuite 1.4-era APIs, while a newer local RoboSuite checkout or MuJoCo 3.x wheel can break environment creation.

Impact: LIBERO/RoboSuite diagnostic rollout can fail before reset with errors such as `mj_fullM()` argument mismatch.

Mitigation: Keep the local RoboSuite source checkout aligned to the official `v1.4.0` tag for LIBERO compatibility. In the selected WSL venv, keep the bounded diagnostic path on `mujoco==2.3.7` unless a new risk assessment proves a later version is compatible. Do not change Windows drivers, CUDA, or system packages for this fix.

## WSL Simulator Dependency Drift

Risk: The selected WSL venv may import `robosuite` and `libero` but still miss runtime dependencies needed for actual LIBERO environment creation.

Impact: Diagnostic rollout can fail progressively on small dependencies such as `bddl`, `future`, `easydict`, `matplotlib`, `cloudpickle`, or `gym`, or on NumPy ABI mismatches.

Mitigation: Add only minimal WSL venv packages after green risk assessments, prefer packages listed in official LIBERO/RoboSuite requirements, and validate after each compatibility change. The current bounded diagnostic path uses `bddl==1.0.1`, `future==0.18.2`, `easydict==1.9`, `matplotlib==3.5.3`, `numpy==1.22.4`, `cloudpickle==2.1.0`, `gym==0.25.2`, and `mujoco==2.3.7`.

## LIBERO/RoboSuite Diagnostic Rollout Overinterpretation

Risk: A passing real LIBERO/RoboSuite zero-action diagnostic rollout could be mistaken for benchmark success or policy performance.

Impact: The project could overclaim simulator plumbing as standard success, counterfactual success, or paper-grade evidence.

Mitigation: `scripts\65_bounded_libero_robosuite_diagnostic_rollout.ps1` uses zero actions only, no learned policy, no VLA inference, no training, no GPU job, no multi-seed run, no OpenVLA-OFT, and no benchmark/SOTA/paper claim. It keeps `ready_for_benchmark_rollout=false` and `ready_for_paper_claim=false`. Benchmark rollout and learned-policy rollout require separate risk assessment and policy gates.

## Learned-Policy Rollout Topology Risk

Risk: SmolVLA runtime readiness is established in the Windows conda environment, while LIBERO/RoboSuite simulator readiness is established in WSL. A learned-policy rollout needs policy inference and simulator stepping in one reliable execution topology.

Impact: A naive rollout could fail through missing WSL SmolVLA dependencies, cross-process IPC latency, image serialization bugs, or inconsistent asset paths.

Mitigation: Prefer the WSL-only topology first. Use `scripts\66_plan_libero_policy_rollout_readiness.ps1` to check local paths, passed diagnostic rollout evidence, SmolVLA file completeness, tokenizer dependency presence, and WSL module-spec readiness without loading the model or running rollout. Only create a tiny learned-policy rollout runner if that planner is green; otherwise reduce scope to WSL SmolVLA runtime setup/readiness.

## Tiny Benchmark Rollout Overinterpretation

Risk: A future tiny learned-policy LIBERO rollout could be mistaken for standard benchmark success or paper-grade evidence.

Impact: A one-task or few-step local pilot could be overstated as general LIBERO performance, counterfactual robustness, or SOTA evidence.

Mitigation: Keep the first learned-policy rollout capped by task count, steps, runtime, and VRAM. Label outputs as bounded benchmark diagnostic or local pilot until a documented benchmark protocol, baselines, ablations, and repeated validation exist. Paper-grade candidate reports may be generated only from verified outputs with explicit evidence labels and limitations.

## WSL SmolVLA Runtime Setup Timeout Risk

Risk: A venv-local package setup can hit the 30 minute timeout after package installation work has mostly completed, leaving the setup report marked failed even though module specs are present.

Impact: The project could either retry unnecessarily or hide a real timeout failure.

Mitigation: After timeout, check for residual pip/install processes, rerun the WSL module-spec planner, and rerun the setup guard. Current local state: no residual pip process remained, all required module specs were present, and the second guard run reported setup complete without further installs.

## WSL SmolVLA Dependency Completeness Risk

Risk: The WSL runtime setup currently verifies module specs and installs `lerobot==0.4.4` with `--no-deps` to avoid broad simulator venv drift. A later WSL model-load or policy-inference smoke may reveal additional runtime dependencies.

Impact: Tiny learned-policy rollout may fail at load time or inference time despite module-spec readiness.

Mitigation: Add a separate WSL SmolVLA load-only or policy-action smoke before long rollouts if the tiny rollout runner cannot safely combine load and rollout. Keep each missing dependency fix venv-local, risk-assessed, and bounded; do not change system CUDA/drivers or OpenVLA-OFT. Current local single-action smoke exposed and fixed the required WSL LeRobot import packages `draccus`, `datasets`, `imageio[ffmpeg]`, `diffusers`, `pyserial`, `deepdiff`, `av`, and `einops`.

## WSL Single-Action Smoke Overinterpretation

Risk: A passing WSL SmolVLA single-action smoke could be mistaken for learned-policy LIBERO rollout success.

Impact: The project could overclaim one synthetic action as benchmark evidence or paper-grade progress.

Mitigation: Label the result as model-load/action-interface smoke only. It used CPU, synthetic input, one action, no simulator rollout, no training, no GPU job, no OpenVLA-OFT, and no paper claim. Learned-policy LIBERO rollout requires a separate task-local runner and risk assessment.

## Learned-Policy Diagnostic Rollout Overinterpretation

Risk: A passing one-task, three-step learned-policy LIBERO rollout could be mistaken for benchmark success, standard performance, counterfactual robustness, or paper-grade evidence.

Impact: The project could overstate a topology/integration result even though the current diagnostic success check is `false` and reward sum is `0.0`.

Mitigation: Label `scripts\72_bounded_tiny_learned_policy_rollout.ps1` outputs as tiny learned-policy diagnostic evidence only. Report success checks, reward, step count, policy latency, action shape, and failure modes honestly. Do not make standard-success, SOTA, or paper-grade claims until a documented benchmark protocol, baselines, ablations, and repeated validation exist.

## Metric Summary Overclaim Risk

Risk: A clean metric summary could make a failed task execution look like positive policy performance because the rollout wrapper passed.

Impact: The project could confuse integration success with manipulation success.

Mitigation: `scripts\73_generate_tiny_learned_policy_metric_summary.ps1` reports wrapper/source pass separately from diagnostic success count, diagnostic success rate, reward sum, and failure modes. Current local result explicitly records diagnostic success rate `0.0` and failure mode `diagnostic_success_check_false`.

## Premature Rollout Matrix Scaling Risk

Risk: After proving learned-policy simulator integration, the project could scale to multiple tasks before the policy shows any diagnostic task-success signal.

Impact: Larger rollouts could spend time measuring obvious failures and make the research state look stronger than it is.

Mitigation: `scripts\74_plan_bounded_learned_policy_rollout_matrix.ps1` reduces scope when diagnostic success rate is `0.0`. Current local decision is `reduce_scope`: run a one-task, 10-step longer diagnostic before any multi-task rollout matrix.

## Action Interface Misalignment Risk

Risk: The local SmolVLA action output can drive the LIBERO environment without crashing, but its scale, coordinate convention, gripper padding, state inputs, camera mapping, or language prompt may not match the policy's expected deployment interface.

Impact: Rollouts can execute cleanly while reward and success remain at zero, making raw execution stability uninformative about method quality.

Mitigation: After the reduced-scope metric summary, add targeted action-interface diagnostics before scaling: action magnitude/range logs, gripper behavior, observation key audit, language prompt audit, and a small comparison against zero-action or replay-style baselines. Keep all results diagnostic/local-pilot until task success appears under a documented protocol.

## WSL Bash CRLF Risk

Risk: PowerShell scripts stored with CRLF line endings can inject carriage returns into generated WSL bash command strings.

Impact: WSL command execution can fail before the Python runner starts, with bash errors such as `$'\r': command not found` or `ambiguous redirect`.

Mitigation: Strip `\r` from generated bash command strings before `bash -lc` in rollout runners, and keep tests that assert the guard exists.

## Action Interface Diagnosis Gap

Risk: The project could continue rollout scaling without first auditing the action/control interface that likely explains zero reward.

Impact: Additional rollouts would consume runtime while repeating the same interface mismatch.

Mitigation: `scripts\77_plan_action_interface_diagnostics.ps1` prioritizes action dimension/gripper mapping, action normalization/scale, and observation state mapping before larger rollout matrices.

## Confirmed Action-Interface Metadata Risks

Risk: The current bridge likely does not match SmolVLA's intended deployment interface.

Impact: The policy can emit finite actions and still produce zero reward because action dimensions, gripper control, state mapping, or camera naming are wrong.

Mitigation: `scripts\78_audit_action_interface_metadata.ps1` records the current high-priority risks and blocks larger rollout scaling. If adapter metadata is absent, add an explicit action/state adapter patch plan. If adapter metadata is present and zero reward remains, move to adapter-strategy/action-scale diagnosis before scaling.

## Zero-Action Comparison Misinterpretation Risk

Risk: A zero-action versus learned-policy diagnostic comparison could be mistaken for benchmark evidence.

Impact: The project could over-claim from a single diagnostic task with no repeated benchmark protocol.

Mitigation: `scripts\79_compare_zero_action_policy_diagnostic.ps1` is summary-only, sets all claim flags false, and keeps `ready_for_rollout_scaling=false` when learned-policy actions do not outperform zero-action.

## Direct Adapter Patch Regression Risk

Risk: Changing rollout action/state mapping directly could create a new simulator behavior without a clear adapter contract.

Impact: Later diagnostic results would be hard to attribute to action mapping, gripper strategy, state mapping, or camera alias changes.

Mitigation: `scripts\80_plan_action_state_adapter_patch.ps1` requires pure adapter helpers and unit tests before rollout wiring, and keeps rollout scaling blocked until adapter metadata is reported.

## Adapter Helper Semantics Risk

Risk: Adapter helper defaults could be mistaken for the final correct LIBERO control convention.

Impact: A diagnostic gripper or state strategy could be over-trusted before empirical validation.

Mitigation: `tca_map.smolvla.interface_adapters` exposes named strategies and metadata, refuses unsupported dimensions, and keeps rollout wiring as a later separately validated step.

## Adapter Metadata Wiring Risk

Risk: Adapter helper wiring could accidentally be interpreted as simulator validation.

Impact: Synthetic single-sample evidence might be over-read as rollout evidence.

Mitigation: The single-sample path remains synthetic-only, records `simulator_executed=false` and `real_rollouts_performed=false`, and uses adapter metadata only for interface validation.

## Rollout Bridge Wiring Scope Risk

Risk: Wiring adapters into rollout code could accidentally trigger execution or expand into rollout evaluation.

Impact: The project could blur code wiring with empirical evidence.

Mitigation: `scripts\81_plan_rollout_bridge_adapter_wiring.ps1` is planning-only and keeps `ready_for_rollout_execution=false`; actual rollout execution remains a separate bounded diagnostic gate.

## Wired Adapter Diagnostic Attribution Risk

Risk: A rerun after explicit adapter wiring could improve, degrade, or leave unchanged the diagnostic rollout, but that effect could be misattributed to the paper method rather than interface plumbing.

Impact: The project could over-credit TCA-Map or Distributional TCA-Select for a change caused by action/state/image bridge mechanics.

Mitigation: Treat the next rollout as an interface diagnostic only. Compare against prior zero-action and legacy learned-policy diagnostics, log adapter metadata, and keep all paper-grade claim flags false until method baselines and repeated benchmark protocol exist.

## Adapter-Wired Zero-Reward Risk

Risk: The bridge can now record clean explicit adapter metadata while still producing zero reward and no diagnostic success.

Impact: Larger rollouts would likely repeat the same failure while consuming runtime and making the evidence ladder look more mature than it is.

Mitigation: Keep rollout scaling blocked. Run small adapter-strategy/action-scale diagnostics first, including gripper strategy comparison, action magnitude checks, language prompt inspection, image source mapping checks, and state adapter sufficiency checks.

## Adapter Diagnostic Combinatorics Risk

Risk: Gripper strategies, action-scale factors, prompt formats, and camera mappings can multiply into a large rollout matrix.

Impact: The project could spend local runtime on many weak diagnostic variants before establishing which axis matters.

Mitigation: `scripts\82_plan_adapter_strategy_action_scale_diagnostics.ps1` limits the first runner to one task, at most 10 steps per variant, and at most three gripper-strategy variants. Action-scale, prompt, and camera variants are planned as later rungs only after the first strategy diagnostic is summarized.

## Adapter-Strategy Diagnostic Overinterpretation

Risk: A bounded gripper-strategy diagnostic can pass execution while still showing zero reward and no task success.

Impact: The project could mistake a clean wrapper result for improved policy behavior or scale rollout prematurely.

Mitigation: Treat `scripts\83_bounded_adapter_strategy_diagnostic.ps1` as diagnostic/local-pilot evidence only. The current result shows all three gripper strategies execute cleanly but still produce diagnostic success rate `0.0` and reward sum `0.0`. Keep rollout scaling blocked and move next to action-scale, prompt, camera-source, or state-sufficiency diagnostics.

## Action-Scale Diagnostic Overinterpretation

Risk: A bounded action-scale diagnostic can pass execution and show expected action magnitude changes while still showing zero reward and no task success.

Impact: The project could mistake action-scaling plumbing for improved policy behavior or scale rollout prematurely.

Mitigation: Treat `scripts\85_bounded_action_scale_diagnostic.ps1` as diagnostic/local-pilot evidence only. The current result shows scales `0.25`, `0.5`, and `1.0` execute cleanly and scale action magnitude as expected, but still produce diagnostic success rate `0.0` and reward sum `0.0`. Keep rollout scaling blocked and move next to prompt-format, camera-source, or state-sufficiency diagnostics.

## Prompt-Format Diagnostic Overinterpretation

Risk: A bounded prompt-format diagnostic can pass execution and change policy actions while still showing zero reward and no task success.

Impact: The project could mistake prompt sensitivity for improved policy behavior or scale rollout prematurely.

Mitigation: Treat `scripts\87_bounded_prompt_format_diagnostic.ps1` as diagnostic/local-pilot evidence only. The current result shows stem-derived and BDDL-language prompts execute cleanly and change action previews, but still produce diagnostic success rate `0.0` and reward sum `0.0`. Keep rollout scaling blocked and move next to camera-source or state-sufficiency diagnostics.

## Camera-Source Diagnostic Overinterpretation

Risk: A bounded camera-source diagnostic can pass execution and change image-source metadata/actions while still showing zero reward and no task success.

Impact: The project could mistake camera-source sensitivity for improved policy behavior or scale rollout prematurely.

Mitigation: Treat `scripts\89_bounded_camera_source_diagnostic.ps1` as diagnostic/local-pilot evidence only. The current result shows camera alias variants execute cleanly and change image sources/action previews, but still produce diagnostic success rate `0.0` and reward sum `0.0`. Keep rollout scaling blocked and move next to state-sufficiency diagnostics.

## State-Sufficiency Diagnostic Overinterpretation

Risk: A bounded state-sufficiency diagnostic can pass execution and change state-vector metadata/actions while still showing zero reward and no task success.

Impact: The project could mistake state sensitivity for improved policy behavior or scale rollout prematurely.

Mitigation: Treat `scripts\91_bounded_state_sufficiency_diagnostic.ps1` as diagnostic/local-pilot evidence only. The current result shows state variants execute cleanly and change state adapter metadata/action previews, but still produce diagnostic success rate `0.0` and reward sum `0.0`. Keep rollout scaling blocked and generate a diagnostic synthesis/no-go report before any larger rollout matrix.

## Repeated Zero-Reward Diagnostic Ladder Risk

Risk: Multiple bounded diagnostics can pass wrapper execution while all task reward and success signals remain zero.

Impact: The project could spend local runtime on additional variants that are unlikely to clarify the core compatibility issue.

Mitigation: After adapter strategy, action scale, prompt format, camera source, and state sufficiency all produce zero reward, prefer synthesis, no-go analysis, and environment-policy compatibility inspection over rollout scaling. Any further learned-policy diagnostic should have a specific compatibility hypothesis and remain one-task bounded.

## Rollout Scaling After No-Go Synthesis Risk

Risk: The project could continue to larger learned-policy rollout matrices even after the diagnostic synthesis reports no positive signal.

Impact: Local runtime would be spent on a path that the current evidence says is unlikely to produce useful method evidence, and the boundary between diagnostic evidence and benchmark evidence could blur.

Mitigation: `scripts\92_generate_learned_policy_diagnostic_synthesis.ps1` records `decision=no_go_rollout_scaling`, `positive_diagnostic_signal_found=false`, and `ready_for_rollout_scaling=false`. The next step must be a bounded environment-policy compatibility audit or a narrowly justified one-task compatibility fix, not broader rollout scaling or paper-grade claims.

## Environment-Policy Compatibility Blocker Risk

Risk: The learned-policy diagnostic path may be incompatible with the selected LIBERO task because checkpoint provenance, VLM loading policy, action convention, or observation convention is mismatched.

Impact: Additional rollout variants could continue to produce zero reward regardless of TCA-Map method quality, obscuring whether the issue is policy-environment compatibility or the proposed method.

Mitigation: `scripts\93_audit_environment_policy_compatibility.ps1` keeps rollout scaling blocked and recommends a bounded offline LIBERO HDF5 demonstration interface audit before any further learned-policy rollout variant.

## HDF5-To-Policy Adapter Reproduction Risk

Risk: The rollout bridge may adapt simulator observations differently from the local LIBERO HDF5 demonstrations, especially around 7D demonstration actions, 6D policy actions, camera count, image resolution, and state key naming.

Impact: A learned-policy rollout can remain at zero reward because adapter inputs are not faithful to the training/evaluation data convention.

Mitigation: `scripts\94_audit_libero_hdf5_interface.ps1` blocks rollout scaling and recommends a report-only offline adapter reproduction check from the first HDF5 timestep before additional rollout variants.

## Gripper Strategy Default Risk

Risk: The current zero-hold gripper default may not match local LIBERO demonstration gripper semantics.

Impact: Learned-policy diagnostics can send plausible 6D motion while the gripper command remains incompatible with the demonstrated action convention, keeping reward at zero.

Mitigation: `scripts\95_check_offline_adapter_reproduction.ps1` shows the first demonstration action is exactly reproduced by the gripper-close adapter, not zero-hold. Any next rollout must be a one-task compatibility diagnostic for this specific hypothesis, not rollout scaling.

## Duplicate Gripper-Close Diagnostic Risk

Risk: The project could rerun an identical gripper-close rollout diagnostic even after a previous close-strategy variant already produced zero reward and no diagnostic success.

Impact: Local runtime is spent repeating a known zero-signal variant, and the evidence ladder may look busier without becoming more informative.

Mitigation: `scripts\96_plan_gripper_close_compat_diagnostic.ps1` checks the previous adapter-strategy diagnostic report before authorizing another close diagnostic. If close already ran cleanly with zero signal, the planner returns `decision=reduce_scope` and recommends an HDF5-aligned task/initial-state/action-sign compatibility check instead.

## HDF5 Initial-State Alignment Risk

Risk: Offline adapter reproduction can match the first demonstration action while learned-policy rollout starts from a different simulator reset state.

Impact: A gripper/action strategy can look correct in HDF5 replay space yet still produce zero rollout reward because the environment state, object placements, or hidden simulator state are not aligned.

Mitigation: `scripts\97_audit_hdf5_rollout_alignment.ps1` checks task-name alignment, HDF5 `init_state`/`states` availability, and whether the rollout bridge appears to set the demonstration initial state. If not, it recommends a bounded HDF5 initial-state or first-action replay planner before another learned-policy rollout.

## HDF5 Replay Scope Creep Risk

Risk: A first-action replay diagnostic could drift into learned-policy rollout, benchmark evaluation, or multi-step policy testing.

Impact: The project could conflate simulator/data-convention debugging with policy performance evidence.

Mitigation: `scripts\98_plan_hdf5_initial_state_replay.ps1` keeps replay planning separate from execution and defines a first runner capped to one HDF5 demo and one replayed demonstration action, with no learned-policy inference, model loading, training, GPU jobs, OpenVLA-OFT, multi-seed evaluation, or paper claims.

## HDF5 Replay Execution Overinterpretation Risk

Risk: A successful first-action HDF5 replay could be mistaken for learned-policy performance.

Impact: The project could overstate data/simulator compatibility as manipulation success.

Mitigation: `scripts\100_bounded_hdf5_initial_state_replay.ps1` reports `hdf5_replay_diagnostic_performed` separately from learned-policy inference and benchmark rollout flags. It keeps learned-policy inference, training, GPU jobs, OpenVLA-OFT, benchmark rollout, multi-seed evaluation, and paper claims false.

## Init-State Learned-Policy Recheck Scope Risk

Risk: A learned-policy recheck from an HDF5 demonstration initial state could be mistaken for benchmark rollout scaling or paper-grade evidence.

Impact: The project could overclaim a narrow compatibility diagnostic or expand to broader rollout evaluation before a positive, repeated signal exists.

Mitigation: `scripts\101_plan_init_state_learned_policy_recheck.ps1` is planning-only and authorizes only a future separately gated one-task runner with one HDF5 demo initial state and at most five policy-controlled steps. It keeps rollout scaling, multi-seed evaluation, GPU jobs, training, OpenVLA-OFT, and paper claims blocked.

## Init-State Recheck Zero-Reward Risk

Risk: The init-state learned-policy recheck can pass execution and set the HDF5 demonstration state while still producing zero reward and no diagnostic task success.

Impact: The project could misread topology success as evidence that the learned policy is working on LIBERO.

Mitigation: Treat `scripts\102_bounded_init_state_learned_policy_recheck.ps1` as diagnostic/local-pilot evidence only. The current result has task success `false` and reward sum `0.0`, so rollout scaling and paper-grade claims remain blocked.

## Rollout Variant Exhaustion Risk

Risk: After reset-only and HDF5-init-state learned-policy diagnostics all show zero reward and no task success, continuing to add rollout variants may spend time without isolating the root cause.

Impact: The evidence ladder could grow broader but not more informative, and the project could drift toward weak rollout claims.

Mitigation: `scripts\103_generate_init_state_recheck_metric_summary.ps1` records `decision=no_go_rollout_scaling`. The next work should be report-only or offline analysis of checkpoint/task alignment, VLM loading policy, and demonstration-conditioned action decoding before any further learned-policy rollout.
