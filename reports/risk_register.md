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

Mitigation: Use `scripts\50_check_libero_hdf5_reader.ps1` first. If `h5py` is missing, perform a separate dependency risk assessment before installing `h5py>=3.11` from the Python package index. Do not change CUDA/PyTorch versions, install simulator stacks, train, rollout, import heavy VLA models, execute OpenVLA-OFT, or make paper claims.

## LIBERO Rollout Readiness Overstatement

Risk: Asset checkers could mark `ready_for_libero_rollout=true` merely because LIBERO, RoboSuite, and data paths exist.

Impact: A path/data-ready state could be mistaken for permission to run simulator rollouts.

Mitigation: `scripts\11_check_real_assets.ps1` and hard-stop summaries keep `ready_for_libero_rollout=false` unless a separate simulator import/render/rollout risk gate is implemented and passes. Path/data readiness is reported separately.
