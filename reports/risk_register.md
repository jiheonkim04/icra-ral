# Risk Register

## P-Hacking And Novelty Overclaim Risk

Risk: ActionMap vs TCA-Map, TCA-Select, LoRA, or QLoRA comparisons could be
made to look positive by changing metrics, samples, seeds, baselines, tuning
budget, or visualizations after seeing results.

Impact: A paper candidate would be untrustworthy, and weak or negative results
could be hidden instead of driving a correct kill/pivot decision.

Mitigation: `reports/research_integrity_evaluation_policy.md` fixes primary
metrics, baseline list, ablation list, split/sample policy, tuning budget, and
kill/pivot criteria before confirmatory evaluation. Failed runs and weak
results must be logged. Exploratory debugging must be labeled separately. If
ActionMap + LoRA or ActionMap + counterfactual augmentation matches TCA-Map,
report weak novelty. If TCA-Select adds no measurable gain or offline gains
disappear in rollout, report it directly.

## Unbounded Autopilot Loop Risk

Risk: Codex could continue through multiple major research milestones in one execution, producing large diffs or planner chains without a clear evidence checkpoint.

Impact: Harder review, weaker research integrity, accidental scope creep, and unclear distinction between planning/scaffolding and actual loss/metric/rollout evidence.

Mitigation: `reports/autopilot_bounded_execution_policy.md` now caps each execution at one major research milestone. Codex must stop before commit if more than 50 files or more than 5,000 changed lines would be included, and must report changed-file count, line diff count, training/rollout/loss/scaffolding status, validation results, and merge justification before every merge.

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

## Checkpoint-Task Alignment Risk

Risk: The local SmolVLA checkpoint may not be task-provenance-aligned with the selected LIBERO diagnostic task, and the local bounded diagnostics may disable VLM weights for memory-safe execution.

Impact: More rollout variants could keep producing zero reward for checkpoint/task or VLM-loading reasons rather than revealing anything about TCA-Map or Distributional TCA-Select.

Mitigation: `scripts\104_audit_smolvla_libero_checkpoint_task_alignment.ps1` records checkpoint provenance fields, selected BDDL language, VLM loading policy, action-dimension convention, HDF5 adapter evidence, and the zero-signal init-state recheck result. If it passes, proceed to a planning-only offline demonstration-conditioned action-decoding gate before any further learned-policy rollout.

## Offline Action-Decoding Scope Risk

Risk: A useful offline action-decoding check could accidentally become simulator rollout, repeated model evaluation, training, or a paper-grade claim.

Impact: The project could blur a one-sample diagnostic with benchmark evidence or spend local compute before the checkpoint/task alignment issue is understood.

Mitigation: `scripts\105_plan_offline_demo_conditioned_action_decoding.ps1` is planning-only and requires a separate task-local future gate for any one-sample model inference. The future runner must cap itself to one local HDF5 observation/action pair, CPU by default, no simulator environment, no rollout, no training, no downloads, no OpenVLA-OFT, and no paper claims.

## Offline Action-Decoding Overinterpretation Risk

Risk: A one-sample offline action-decoding diagnostic could be mistaken for rollout success, benchmark evidence, or paper-grade action accuracy.

Impact: The project could overclaim from a single HDF5 timestep even if the model action is finite or partially aligned with the expert action.

Mitigation: `scripts\106_bounded_offline_demo_action_decoding.ps1` labels the output as one-sample offline diagnostic evidence only, reports action L1/MSE without success claims, and keeps rollout scaling, benchmark claims, SOTA claims, and paper-grade claims false.

## Weak Offline Alignment Risk

Risk: The bounded offline decoder can produce finite actions while remaining far from the expert demonstration action.

Impact: More learned-policy rollout variants may keep failing for decoder/checkpoint/VLM-loading reasons unrelated to TCA-Map.

Mitigation: `scripts\107_summarize_offline_demo_action_decoding.ps1` classifies one-sample action alignment and keeps rollout scaling blocked when the offline alignment signal is weak.

## VLM-Disabled Diagnostic Risk

Risk: The local SmolVLA checkpoint config requests VLM weights, but bounded local diagnostics may run with `load_vlm_weights=false` for memory and dependency safety.

Impact: Zero-reward rollout diagnostics and weak offline action decoding may reflect a disabled VLM path rather than the final intended policy behavior.

Mitigation: `scripts\108_plan_vlm_loading_policy_action_normalization_audit.ps1` records the config-vs-observed load policy mismatch and treats any VLM-enabled load or full SmolVLM2 weight acquisition as a separate risk-assessed task.

## Action Unnormalization and Clipping Risk

Risk: The policy uses ACTION `MEAN_STD` unnormalization and the 6D-to-7D adapter may clip continuous values before comparing or stepping the LIBERO environment.

Impact: The learned-policy bridge may distort actions even when policy inference succeeds, causing zero reward or weak expert-action alignment.

Mitigation: Keep rollout scaling blocked and plan a repeated offline HDF5 decoding diagnostic that logs unnormalization, clipping, gripper strategy, and expert-action error before another learned-policy rollout.

## Repeated Offline Decoding Scope Risk

Risk: A repeated offline action-decoding diagnostic could drift from a tiny diagnostic into repeated inference, benchmark evaluation, or a rollout substitute.

Impact: The project could overclaim from a small number of HDF5 timesteps or spend compute before the VLM/action-normalization issue is understood.

Mitigation: `scripts\109_plan_repeated_offline_demo_action_decoding.ps1` caps the future runner at three local HDF5 timesteps, CPU SmolVLA inference only, no simulator environment, no rollout, no training, no downloads, no GPU job, no OpenVLA-OFT, and no paper claim.

## Repeated Offline Runner Overinterpretation Risk

Risk: A bounded repeated offline runner can load SmolVLA and compute expert-action distances, but those distances are not simulator success or benchmark evidence.

Impact: Passing execution could be mistaken for task performance, especially if some action dimensions look closer to expert actions.

Mitigation: `scripts\110_bounded_repeated_offline_demo_action_decoding.ps1` labels results as tiny repeated offline diagnostic evidence only. It keeps rollout scaling, benchmark claims, SOTA claims, and paper-grade claims false regardless of execution success.

Current status: the runner passed, but repeated offline alignment remained weak and every adapted action clipped one value. This raises the priority of VLM-loading policy and action-normalization diagnosis before any further learned-policy rollout.

## VLM Weight Acquisition and Loading Risk

Risk: Enabling `load_vlm_weights=true` may require full SmolVLM2 weights, extra disk/RAM, or a gated/token/license condition.

Impact: The project could accidentally download a large or gated dependency, overrun local memory, or confuse VLM-disabled diagnostics with intended policy behavior.

Mitigation: `scripts\111_plan_vlm_enabled_loading_risk.ps1` is metadata-only and must pass source/license/size/disk checks before any VLM weight acquisition. VLM-enabled load smoke remains a later separate gate.

Current status: metadata risk is green for acquisition planning. The remaining risk is load behavior and memory once full VLM files are present; that must be tested by a separate bounded load-smoke gate.

## VLM Required Files Present But Not Load-Validated

Risk: The required SmolVLM2 files are now present locally, but VLM-enabled SmolVLA loading may still exceed CPU RAM/runtime budgets or expose API/config mismatches.

Impact: Enabling `load_vlm_weights=true` without a separate bounded load-smoke plan could create a long or memory-heavy model-load task and confuse acquisition readiness with policy capability.

Mitigation: Treat `scripts\112_acquire_vlm_required_files.ps1` as acquisition evidence only. Before any VLM-enabled model load, add or run a separate bounded load-smoke planner that estimates RAM/runtime, refuses rollout/training/GPU/OpenVLA gates, and labels output as engineering load-smoke evidence only.

## VLM-Enabled Load-Smoke Scope Creep Risk

Risk: A `load_vlm_weights=true` smoke can accidentally become inference, rollout, training, GPU execution, or a paper result.

Impact: The project could overrun RAM/runtime budgets or overinterpret a construction test as evidence that policy behavior improved.

Mitigation: Use `scripts\113_plan_vlm_enabled_load_smoke.ps1` first. A future runner must require `ALLOW_HEAVY_IMPORT=1` and `ALLOW_VLM_ENABLED_LOAD_SMOKE=1`, stay CPU-first and load-only, record RAM/runtime, perform no inference/training/rollout/GPU job/OpenVLA-OFT/token access, and label output as engineering load-smoke evidence only.

Current status: the bounded CPU load-only runner passed with `load_vlm_weights=true` and no CUDA allocation. Remaining risk is behavioral: VLM-enabled construction may still produce weak offline action alignment, so the next step should be a bounded repeated offline decoding recheck before any rollout scaling.

## VLM-Enabled Offline Recheck Overinterpretation Risk

Risk: A VLM-enabled repeated offline decoding recheck may improve or worsen action distance but still remain non-rollout diagnostic evidence.

Impact: The project could mistake offline action similarity for standard success or counterfactual robustness.

Mitigation: Plan and run at most three HDF5 timestep decodes, compare directly against the previous `load_vlm_weights=false` metrics, keep rollout scaling blocked until alignment improves, and label results as offline diagnostic evidence only.

Current status: the VLM-enabled recheck passed and improved mean action L1/MSE versus the previous no-VLM repeated offline diagnostic, but the resulting alignment signal is still `weak` and clipped action values remain. Treat this as evidence that VLM loading matters, not as evidence of benchmark success. Continue with VLM-on/off summary and action-normalization/provenance analysis before any rollout scaling.

## VLM-On/Off Summary Overinterpretation Risk

Risk: A compact VLM-on/off summary may make the improvement look more conclusive than it is.

Impact: The project could move to learned-policy rollout scaling even though the offline alignment signal is still weak and action clipping persists.

Mitigation: `scripts\117_summarize_vlm_enabled_offline_decoding.ps1` labels the comparison as report-only diagnostic evidence, keeps rollout scaling, benchmark claims, and paper claims false, and routes the next step to action-normalization/provenance analysis when weak alignment or clipping remains.

Current status: the summary passed and found meaningful VLM-on improvement, but alignment remained `weak` and clipping persisted. This keeps the overinterpretation risk active and routes the next safe task to action-normalization/provenance audit.

## Action Normalization Provenance Mismatch Risk

Risk: The local SmolVLA processor action statistics may come from a robot/action convention that is not aligned with local LIBERO demonstration actions.

Impact: Learned-policy actions can be finite and VLM-sensitive while still being systematically mis-scaled or mis-adapted for LIBERO, causing zero-reward rollouts and weak offline expert-action alignment.

Mitigation: `scripts\118_audit_action_normalization_provenance.ps1` inspects processor safetensor action mean/std keys and magnitudes, compares them against local LIBERO action previews, and keeps rollout scaling blocked if provenance or scale mismatch remains.

Current status: the audit found SO100 action-stat prefixes and action-stat magnitudes far outside the local LIBERO expert-action preview range. Learned-policy rollout scaling remains blocked until an action-stat mapping or checkpoint/task-provenance correction plan is created and validated.

## Premature Action-Stat Correction Risk

Risk: A provenance correction plan could be misread as permission to bypass postprocessing, alter model behavior, or run another rollout immediately.

Impact: The project could introduce an ad hoc adapter fix without first proving local LIBERO action statistics differ from checkpoint stats in a controlled report-only audit.

Mitigation: `scripts\119_plan_action_stat_provenance_correction.ps1` is planning-only and selects a report-only LIBERO action-stat subset audit first. It does not authorize model modification, training, rollout, checkpoint downloads, GPU jobs, OpenVLA-OFT, or paper claims.

Current status: the plan selected a report-only LIBERO action-stat subset audit. This keeps policy changes and rollouts blocked until local LIBERO action statistics are measured directly.

## LIBERO Action-Stat Audit Scope Risk

Risk: Computing LIBERO action statistics could be mistaken for a corrected policy or rollout evidence.

Impact: The project could overinterpret dataset-stat confirmation as learned-policy compatibility.

Mitigation: `scripts\120_audit_libero_action_stats.ps1` is report-only and keeps model loading, inference, training, rollouts, GPU jobs, OpenVLA-OFT, policy behavior changes, and paper claims false.

Current status: the audit confirmed 7D unit-scale LIBERO actions and 6D SO100 large-scale checkpoint action statistics. Learned-policy rollout failures under this bridge should not be interpreted as paper-relevant SmolVLA/LIBERO performance until provenance or normalized-action-space handling is resolved.

## Normalized Action-Space Probe Scope Risk

Risk: A normalized-action-space probe plan could be misread as permission to bypass the postprocessor, alter the policy adapter, run another learned-policy rollout, or claim that the checkpoint is LIBERO-compatible.

Impact: The project could hide a checkpoint/task provenance mismatch behind an ad hoc action-space transform and produce misleading rollout diagnostics.

Mitigation: `scripts\121_plan_normalized_action_space_probe.ps1` is planning-only. When SO100-prefixed checkpoint stats and 7D unit-scale LIBERO actions are confirmed, it selects checkpoint/task provenance resolution before any normalized-action runner. It keeps model loading, inference, training, rollouts, downloads, GPU jobs, OpenVLA-OFT, policy behavior changes, and paper claims false.

Current status: the planner selected checkpoint/task provenance resolution and left the future normalized-action runner disabled until a separate gate exists.

## Checkpoint Provenance Misuse Risk

Risk: The current `lerobot/smolvla_base` checkpoint could be treated as a LIBERO learned-policy baseline even if its processor stats, action shape, and model-card cues indicate a base/SO100-like provenance mismatch.

Impact: Zero-reward or weak offline diagnostics could be misreported as SmolVLA/LIBERO policy failure or as TCA-Map evidence, when they may mainly reflect checkpoint/action-convention mismatch.

Mitigation: `scripts\122_resolve_checkpoint_task_provenance.ps1` is report-only and blocks learned-policy rollout scaling when checkpoint metadata remains incompatible with local LIBERO HDF5 action stats. It routes the project toward offline/head TCA-Map plus required LoRA tracks or a separate LIBERO-aligned checkpoint source plan.

Current status: the audit blocked learned-policy LIBERO rollout scaling for the current checkpoint and marked offline/head TCA-Map plus required LoRA evidence, or a separate LIBERO-aligned checkpoint source plan, as the safe next direction.

## Offline Pivot Overinterpretation Risk

Risk: Pivoting to real-LIBERO offline/head and LoRA proxy evidence could be mistaken for standard success, benchmark rollout success, or paper-grade empirical proof.

Impact: The paper path could overclaim from useful but offline diagnostic evidence while learned-policy rollout scaling remains blocked by checkpoint provenance.

Mitigation: `scripts\123_plan_offline_tca_map_lora_pivot.ps1` is report-only and explicitly selects an evidence table/gap report, not a paper claim. It keeps learned-policy rollout scaling, standard success, benchmark success, SOTA claims, paper claims, OpenVLA-OFT, downloads, GPU jobs, and heavy imports blocked.

Current status: the pivot plan selected an offline evidence table and gap report. Learned-policy rollout scaling remains blocked for the current checkpoint.

## Offline Evidence Table Overinterpretation Risk

Risk: A consolidated evidence table may make offline proxy deltas look paper-grade even though no valid learned-policy rollout success exists.

Impact: The project could overstate TCA-Map or LoRA gains from deterministic/offline proxy scaffolds.

Mitigation: `scripts\124_generate_offline_evidence_gap_report.ps1` labels every arm as offline proxy/not paper-grade and includes explicit gaps for standard success, learned-policy rollout, and paper claims.

Current status: the evidence gap report passed and selected bounded LoRA/offline-proxy scale-up planning as the next safe step. Current-checkpoint learned-policy rollout scaling and paper claims remain blocked.

## LoRA Offline Scale-Up Scope Risk

Risk: A LoRA/offline proxy scale-up could drift into full fine-tuning, model loading, GPU-heavy execution, rollout, or paper-grade claims.

Impact: The project could confuse bounded proxy adaptation with real learned-policy benchmark evidence or destabilize the local machine.

Mitigation: `scripts\125_plan_bounded_lora_offline_scaleup.ps1` is planning-only and caps any future runner to CPU-only offline proxy training with at most 16 pairs, 64 samples, 64 steps, LoRA rank 4, frozen base weights, no full fine-tuning, no model load, no heavy imports, no GPU job, no rollout, no OpenVLA-OFT, and no paper claim.

Current status: the plan passed and authorized a future separately gated CPU-only offline LoRA scale-up runner under `ALLOW_TINY_TRAINING=1`.

## Bounded Offline LoRA Scale-Up Overinterpretation Risk

Risk: A passing bounded LoRA scale-up over local HDF5 action snippets could be mistaken for a real SmolVLA adapter result, standard success, rollout success, or paper-grade benchmark evidence.

Impact: The project could overclaim useful proxy trends before validating an aligned learned-policy checkpoint or simulator rollout path.

Mitigation: `scripts\126_bounded_lora_offline_scaleup.ps1` requires `ALLOW_TINY_TRAINING=1`, trains only tiny NumPy LoRA matrices on CPU, labels outputs as offline proxy only, and reports rollout and paper-claim readiness as false. It forbids downloads, GPU jobs, heavy VLA imports, model loading, inference, rollouts, simulator execution, OpenVLA-OFT, full fine-tuning, token access, and paper claims.

Current status: the runner has been added as the next bounded execution gate after the scale-up plan.

Current result: the runner passed under task-local `ALLOW_TINY_TRAINING=1` and reported offline proxy metrics only. It remains unsuitable for standard success, rollout success, or paper-grade claims.

## Scale-Up Evidence Table Overinterpretation Risk

Risk: Adding bounded LoRA scale-up rows to the consolidated evidence table could make the table look closer to a benchmark result than it is.

Impact: Readers could confuse real-LIBERO offline proxy diagnostics with simulator success or learned-policy rollout evidence.

Mitigation: The scale-up-aware evidence gap report labels all rows as `not_standard_success` and `not_paper_grade`, keeps training in the report policy false because the refresh itself is report-only, and leaves rollout, benchmark, and paper-claim readiness false.

Current status: the evidence gap generator can include bounded LoRA scale-up rows only from an existing local runtime report; it does not train or rerun the scale-up.

Current result: the refreshed evidence table included bounded scale-up rows and still reported no rollout or paper-claim readiness. The remaining risk is overattribution: TCA-Map + LoRA improves over ActionMap + LoRA in the proxy, but Distributional TCA-Select adds no extra LoRA proxy gain in this bounded runner.

## Selection-Gain Attribution Gap Risk

Risk: TCA-Map + LoRA proxy gains could be incorrectly attributed to Distributional TCA-Select even though the current bounded LoRA runner shows zero additional selection delta.

Impact: The method narrative could overstate the inference-time selection component before a candidate-ambiguity stress test demonstrates selection-specific value.

Mitigation: `scripts\127_synthesize_scaleup_attribution_gaps.ps1` records the zero selection delta as an attribution gap and routes the next safe task to a report-only TCA-Select ambiguity/stress-test plan.

Current status: synthesis scaffolding is added as a report-only guard before any further selection-gain claims.

Current result: the synthesis passed and explicitly routed the next step to a TCA-Select ambiguity/stress-test plan. This keeps selection-gain claims blocked until a stronger proxy test exists.

## TCA-Select Ambiguity Stress-Test Design Risk

Risk: A weak stress test could still fail to isolate selection-specific gain, or could accidentally use privileged target/state information.

Impact: The project could either miss a real inference-time selection benefit or overstate one using invalid information.

Mitigation: `scripts\128_plan_tca_select_ambiguity_stress_test.ps1` requires candidate diversity, full-vs-masked condition sensitivity, target-conditioned action consistency, and explicitly forbids privileged simulator state, external verifiers, rollout outcomes, and paper-grade success labels.

Current status: the stress test is planning-only and must pass before any offline stress-test runner is implemented.

Current result: the plan passed with explicit forbidden inputs and offline-only metrics. The next runner must preserve those constraints.

## Offline TCA-Select Stress Runner Overinterpretation Risk

Risk: A passing ambiguity stress test could be mistaken for real task success or model policy improvement.

Impact: Offline candidate-selection gains could be overstated as simulator or paper-grade evidence.

Mitigation: `scripts\129_run_tca_select_ambiguity_stress_test.ps1` labels outputs as offline proxy only, reports no model loading/inference/training/rollout/GPU jobs, and leaves paper readiness false.

Current status: the runner scaffolding is added and must be validated before the stress-test evidence can be synthesized.

Current result: the runner passed and produced selection-specific offline proxy evidence: wrong-target proxy delta -1.0 and action L1 delta -0.164299 against a top-heatmap baseline. The mitigation remains active because this is synthetic candidate ambiguity over offline HDF5 snippets, not rollout success or a paper-grade benchmark.

## Stress-Aware Synthesis Overinterpretation Risk

Risk: Combining bounded LoRA scale-up and TCA-Select ambiguity stress evidence in one synthesis could make the method look paper-ready.

Impact: Offline proxy evidence could be overread as standard success or learned-policy rollout evidence.

Mitigation: The refreshed attribution synthesis records `tca_select_ambiguity_stress_included` separately from `bounded_lora_scaleup_included`, keeps rollout and paper readiness false, and routes the next step to report-only evidence-table refresh rather than benchmark rollout.

## Stress Row Evidence Table Overinterpretation Risk

Risk: Adding a TCA-Select ambiguity-stress row to the evidence table could make synthetic candidate-selection proxy results look like a real benchmark arm.

Impact: The project could overstate TCA-Select readiness before learned-policy rollout validation.

Mitigation: The row uses evidence type `real-LIBERO offline ambiguity stress proxy`, marks `not_standard_success` and `not_paper_grade`, stores top-heatmap deltas separately, and leaves learned-policy rollout and paper readiness false.

Current status: the evidence row is present, and the refreshed synthesis routes to report-only candidate-generation readiness planning rather than model inference or rollout.

## Candidate-Generation Readiness Scope Risk

Risk: Planning real candidate generation could drift into model loading, inference, rollout, or GPU execution.

Impact: The project could cross from report-only attribution into a heavy execution gate without a separate risk assessment.

Mitigation: `scripts\130_plan_candidate_generation_readiness.ps1` refuses heavy/import/inference/rollout gates, records `ready_for_real_candidate_generation_smoke_execution=false`, and selects a synthetic-tensor contract checker as the next safe task.

Current status: readiness planning passed without model loading, inference, training, rollout, GPU jobs, simulator execution, OpenVLA-OFT, or paper claims.

## Candidate Contract Overreach Risk

Risk: A contract checker could accidentally become a real policy inference runner.

Impact: A safe interface validation step could cross into heavy model loading, GPU work, or unapproved rollout-adjacent execution.

Mitigation: `scripts\131_check_candidate_generation_contract.ps1` uses synthetic tensors only, refuses heavy/import/inference/training/rollout gates, validates forbidden metadata keys, and leaves real candidate-generation smoke execution false.

Current status: synthetic contract checking passed and routes to a planning-only real candidate-generation smoke risk gate, not direct model inference.

## Real Candidate-Generation Smoke Gate Risk

Risk: A green plan for real candidate generation could be mistaken for permission to run unrestricted model inference.

Impact: A bounded interface smoke could drift into long-running GPU inference, rollout, training, or paper claims.

Mitigation: `scripts\132_plan_real_candidate_generation_smoke.ps1` is planning-only, refuses execution gates while planning, caps any future smoke to one sample, max 4 candidates, grid 8, 10 minutes, 14GB VRAM, and requires task-local `ALLOW_REAL_CANDIDATE_GENERATION_SMOKE=1`, `ALLOW_HEAVY_IMPORT=1`, and `ALLOW_SINGLE_SAMPLE_INFERENCE=1`.

Current status: planning is green for implementation, but execution remains blocked by default and must require all three task-local gates.

## Real Candidate-Generation Smoke Overinterpretation Risk

Risk: A passing bounded real candidate-generation smoke could be mistaken for learned-policy benchmark evidence.

Impact: The project could overclaim from one synthetic input, one local model action decode, and an approximate low-resolution candidate heatmap.

Mitigation: `scripts\133_bounded_real_candidate_generation_smoke.ps1` labels outputs as engineering smoke only, keeps standard success, rollout success, benchmark claims, and paper claims false, and requires all three task-local gates before any heavy import or inference. The smoke forbids downloads, training, rollouts, simulator execution, OpenVLA-OFT, external verifiers, privileged state, token access, and paper claims.

Current status: the scaffold is implemented and default execution is refusal-only. Any execution result must be synthesized as interface evidence only.

Current result: bounded execution passed on CPU with one synthetic action decode, four candidates, grid 8, selected target index 0, wrong-target proxy false, and CUDA max allocated `0.0 MB`. The overinterpretation risk remains active because this is still synthetic-input engineering evidence, not rollout or benchmark evidence.

## Learned-Policy Zero-Reward Over-Debugging Risk

Risk: Repeated bounded rollout diagnostics after zero reward could keep expanding bridge/planner work without producing loss, evaluation metrics, or a stronger diagnosis.

Impact: The project could spend execution budget on rollout variants even though the current evidence already points away from a simple action-shape bridge bug.

Mitigation: Treat the completed gripper, scale, prompt, camera, state, and HDF5 init-state diagnostics as a concrete no-go for learned-policy rollout scaling with the current checkpoint. Only rerun rollout diagnostics if a new specific blocker is identified and the command directly tests it. Otherwise, move to fixed-integrity offline ActionMap vs TCA-Map training/evaluation on real LIBERO HDF5 snippets.

Current result: all bounded learned-policy diagnostic variants executed but produced zero reward and zero diagnostic success. The explicit 6D-to-7D action bridge reported no silent padding or truncation, and HDF5 demo init-state setting worked. The overinterpretation risk remains active: this is diagnostic failure evidence, not a benchmark or paper result.

## Tiny Head-Only Negative Result Overinterpretation Risk

Risk: The tiny ActionMap vs TCA-Map offline training/eval could be selectively described as positive because TCA-Map improved action L1 and counterfactual margin while ignoring worse target accuracy and wrong-target proxy.

Impact: The project could drift into p-hacking or novelty overclaiming before LoRA/QLoRA and rollout evidence exist.

Mitigation: Record the conclusion as `weakens_tca_map` for this exploratory split. Any next LoRA comparison must keep ActionMap + LoRA as the central baseline and must report if LoRA gains dominate the TCA-Map contribution. TCA-Select also must remain separate because it added no measured gain in this run.

Current result: ActionMap loss decreased `0.162408 -> 0.010239`; TCA-Map loss decreased `0.855555 -> 0.126224`; TCA-Map eval action L1 improved by `-0.019147`, but standard proxy delta was `-0.434797`, target top1 delta was `-0.5`, and wrong-target proxy delta was `+0.5`.

## LoRA Negative Attribution Result Risk

Risk: The tiny LoRA attribution comparison could be cherry-picked or reframed as positive because TCA-Map + LoRA improved action L1 and counterfactual margin while losing on target accuracy, wrong-target proxy, standard proxy, and action-target consistency.

Impact: Scaling LoRA before diagnosing the target-conditioning failure could waste compute and encourage p-hacking.

Mitigation: Record the conclusion as `lora_weakens_tca_map` for this split. The next milestone should debug TCA target labels/conditioning on the same split, not broaden the split to hunt for a positive result. Distributional TCA-Select should also remain unsupported by this diagnostic because it added no measured gain despite non-degenerate candidate scores.

Current result: ActionMap + LoRA standard proxy `0.454351`, wrong-target `0.5`; TCA-Map + LoRA standard proxy `0.0`, wrong-target `1.0`; TCA-Select + LoRA delta `0.0` on standard proxy and wrong-target proxy.

## TCA Target-Classifier Failure Risk

Risk: The weak TCA-Map result could be misdiagnosed as a label, metric, or TCA-Select implementation bug when the actual blocker is brittle target prediction/generalization on the held-out tiny split.

Impact: The project could waste compute scaling LoRA, rollout, or larger offline training while preserving the same target-conditioning failure.

Mitigation: `scripts\55_debug_tca_label_conditioning.ps1` now audits the fixed 8-sample split at sample level and checks label alignment, target-conditioning variation, metric direction, one-sample overfit, target-shuffle, oracle-target evaluation, constant-target baseline, and TCA-Select score degeneracy. The current audit found no concrete label/metric/candidate-space bug and reported `verified_no_label_or_metric_bug_but_target_classifier_failure`.

Current result: TCA wrong-target proxy `1.0` means both eval target predictions were wrong. Oracle-target TCA standard proxy improved from `0.0` to `0.86561`, while constant-target baseline standard proxy was `0.442708`. The next safe milestone is to revise/debug TCA target-conditioning design on the same split, not to scale LoRA or rollout.

## Target-Prior Rescue Overinterpretation Risk

Risk: The instruction-text prior rescue result could be overstated as TCA-Map evidence even though it is a tiny offline proxy and uses candidate/task text that may not be available in the same form at paper-grade inference time.

Impact: The project could replace a failed learned target head with a metadata-like shortcut and accidentally claim a method improvement that does not transfer to rollout or benchmark evaluation.

Mitigation: `scripts\56_debug_tca_target_prior_rescue.ps1` labels oracle and instruction-text-prior arms as diagnostic, not paper-grade. It keeps the same fixed split, reports constant/majority and soft-marginalization controls, and keeps rollout, LoRA, scaling, and paper claims blocked.

Current result: learned hard target TCA standard proxy `0.0`, oracle-target TCA `0.86561`, instruction-text-prior TCA `0.86561`, soft marginalization `0.0`, and constant target baseline `0.444499`. The immediate next step is a minimal target-prior/classifier fix followed by the same head-only ActionMap vs TCA-Map rerun, not a paper claim.

## Target-Prior-Fixed Comparison Overinterpretation Risk

Risk: The instruction-text-prior TCA arm now beats ActionMap on the tiny fixed split, but this may rely on clean task/candidate text and remains exploratory offline proxy evidence only.

Impact: The project could overstate a target-prior engineering fix as a paper-grade method result before proving that the same information is available without privileged metadata in rollout or larger evaluation.

Mitigation: Keep oracle-target and instruction-text-prior arms explicitly labeled as diagnostic. Preserve the hard learned-target failure in reports. Before scaling, document whether any text prior uses only non-privileged inference-time information.

Current result: ActionMap standard proxy `0.434797`, hard learned-target TCA `0.0`, instruction-text-prior TCA `0.86561`, oracle-target TCA `0.86561`.

## Distributional TCA-Select Redundancy Risk

Risk: Distributional TCA-Select adds `0.0` standard-proxy, wrong-target, action-target-consistency, and counterfactual-margin delta over the best non-oracle target prior in the target-prior-fixed diagnostic.

Impact: Keeping TCA-Select unchanged could add complexity without evidence of contribution.

Mitigation: Revise the TCA-Select objective before scaling. The next TCA-Select experiment must show a measurable contribution beyond target-prior correctness or honestly log that TCA-Select is unsupported by the diagnostic.

## Fixed-Fusion Overfitting Risk

Risk: The fixed learned+text fusion recovers the tiny split by downweighting the learned prior when it conflicts with the text prior. This may be a useful calibration rule, but it could also be overfit to a two-sample eval split if treated as a general result too early.

Impact: The project could move from a failed learned target head to a text-prior-dominant shortcut without proving robustness.

Mitigation: Keep the result labeled exploratory offline proxy. Rerun LoRA attribution on the same split first for attribution consistency, then cautiously scale only after logging the fixed rule and preserving ActionMap, hard learned-target, instruction-text, fixed-fusion, and oracle baselines.

Current result: fixed learned+text fusion standard proxy `0.86561`, wrong-target `0.0`, matching instruction-text and oracle-target TCA on the tiny split.

## TCA-Select Weak-Delta Risk

Risk: The revised uncertainty-marginalized TCA-Select produced only a weak top-k-uniform standard-proxy delta of `+0.005128` and did not improve wrong-target proxy. This is not strong evidence that TCA-Select is a publishable contribution.

Impact: Keeping TCA-Select central could distract from the stronger target-prior TCA finding.

Mitigation: De-emphasize TCA-Select unless a future execution-first experiment shows nontrivial gains over the corresponding non-select TCA prior. Report weak or null selector results honestly.

## Fixed-Prior LoRA Positive-Result Overinterpretation Risk

Risk: The fixed-prior LoRA rerun strongly beats ActionMap + LoRA on the tiny 8-sample offline proxy split, which could tempt overclaiming after earlier negative head-only and hard-learned LoRA diagnostics.

Impact: The project could present target-prior-corrected TCA as generally solved before showing that the target prior is non-privileged, robust, and transferable to larger offline splits or rollouts.

Mitigation: Keep the result labeled exploratory offline proxy. Preserve the hard learned-target failure, ActionMap + LoRA baseline, instruction-text prior, fixed fusion, oracle upper bound, and TCA-Select ablation in any scaled rerun. Require rollout evidence before paper-grade claims.

Current result: fixed-fusion TCA + LoRA standard proxy `0.910293` vs ActionMap + LoRA `0.454351`; wrong-target proxy `0.0` vs `0.5`; hard learned-target TCA + LoRA remains `0.0` standard proxy and `1.0` wrong-target proxy.

## TCA-Select LoRA Redundancy Risk

Risk: TCA-Select again adds no measurable gain when the target prior is already correct.

Impact: Keeping TCA-Select as the central novelty could add complexity without empirical support.

Mitigation: Treat TCA-Select as a secondary ablation unless a future scaled selector diagnostic shows a nontrivial gain over the corresponding non-select TCA prior.

Current result: fixed-fusion TCA + LoRA and fixed-fusion TCA-Select + LoRA both scored standard proxy `0.910293` and wrong-target proxy `0.0`; selector delta was `0.0`.

## Scaled Fixed-Prior Offline Proxy Overinterpretation Risk

Risk: The fixed-prior TCA advantage survived a 16-sample split, but the evidence is still offline proxy only and still depends on a corrected target prior.

Impact: The project could overclaim a robust method result before validating larger splits, seeds, or rollout behavior.

Mitigation: Preserve ActionMap, hard learned-target TCA, fixed-prior TCA, oracle-target TCA, and TCA-Select ablations in every scaled rerun. Label all current results exploratory offline proxy. Require rollout evidence before paper-grade claims.

Current result: fixed-prior TCA + LoRA standard proxy `0.854` vs ActionMap + LoRA `0.427546`; wrong-target proxy `0.0` vs `0.5`; TCA-Select delta `0.0`.

## Scaled Learned Target Head Bottleneck Risk

Risk: The corrected target prior works, but the learned target head remains weaker and may not generalize without a redesign.

Impact: The method may depend on text-prior availability rather than learning robust target selection from observations/instructions.

Mitigation: Keep hard learned-target TCA as a required baseline in scale-up runs. Do not hide hard learned-target failures behind fixed-prior improvements.

Current result: hard learned-target TCA head-only standard proxy `0.0`, wrong-target `1.0`; hard learned-target TCA + LoRA standard proxy `0.642331`, wrong-target `0.25`, still below fixed-prior TCA + LoRA.

## Multi-Seed Offline Proxy Overconfidence Risk

Risk: Fixed-prior TCA advantage is stable across five seeds on the 16-sample split, but the split is still small and offline proxy only.

Impact: Stable seed results could be mistaken for benchmark robustness or paper-grade evidence.

Mitigation: Require a larger offline split or rollout evidence before making stronger claims. Preserve all weak/negative findings: hard learned-target weakness, LoRA underperforming head-only fixed-prior TCA, and TCA-Select null gains.

Current result: fixed-prior TCA + LoRA beat ActionMap + LoRA in `5 / 5` seeds, mean advantage `0.426798`, std `0.004095`; rollout and paper claims remain false.

## LoRA Performance-Claim Risk

Risk: Because LoRA is a required experiment track, it could be described as improving the method even when it hurts fixed-prior TCA relative to head-only.

Impact: The paper narrative could overstate LoRA as a performance contributor instead of an attribution/robustness arm.

Mitigation: Report fixed-prior head-only vs fixed-prior LoRA directly. Treat LoRA as required evidence for parameter-efficient adaptation, not as the main novelty or guaranteed performance improvement.

Current result: fixed-prior TCA + LoRA underperformed fixed-prior TCA head-only in `5 / 5` seeds, mean standard-proxy delta `-0.090299`.

## 32-Record Offline Proxy Overconfidence Risk

Risk: Fixed-prior TCA advantage survived a 32-record, 5-seed offline proxy split, but this is still not rollout success or paper-grade evidence.

Impact: The result may look stable enough to overclaim before testing the 64-record capacity, rollout behavior, or the learned target-head bottleneck.

Mitigation: Keep the result labeled exploratory offline proxy. Run the available 64-record split next with the same fixed metrics and baselines. Preserve weak/negative results: LoRA underperforming fixed-prior head-only TCA and TCA-Select null gains.

Current result: fixed-prior TCA + LoRA beat ActionMap + LoRA in `5 / 5` seeds with mean advantage `0.429379`, std `0.003737`; wrong-target proxy improved in `5 / 5` seeds; rollout and paper claims remain false.

## 32-Record Task-Imbalance Risk

Risk: The 32-record split uses `10` tasks but is still imbalanced by instruction count, with some tasks appearing once or twice and others appearing more often.

Impact: Fixed-prior TCA advantage could partially reflect task/instruction distribution rather than a broadly robust action-conditioning gain.

Mitigation: Report per-task counts in the runtime report. Do not cherry-pick tasks. Use the deterministic 64-record split next to reduce sensitivity, and inspect per-task breakdown if the advantage shrinks.

Current result: target classes are balanced `{0: 16, 1: 16}`, but instruction/task counts are not uniform. The next 64-record run should preserve deterministic ordering and report the same breakdown.

## TCA-Select Core-Contribution Kill Risk

Risk: TCA-Select again shows no nontrivial gain at 32 records across five seeds.

Impact: Keeping TCA-Select as a central contribution may weaken the research story by adding unsupported complexity.

Mitigation: Treat TCA-Select as secondary. If the 64-record split also shows `0` nontrivial gains, kill it as a core contribution and focus the method claim on target-prior-conditioned action decoding.

Current result: TCA-Select nontrivial gain count was `0 / 5` at 32 records.

## 64-Record Offline Proxy Overconfidence Risk

Risk: Fixed-prior TCA advantage survived the full 64-record deterministic offline proxy split across three seeds, which may look stable enough to overclaim before rollout evidence exists.

Impact: The result could be mistaken for standard success or paper-grade evidence even though it is still an offline proxy using local LIBERO HDF5/counterfactual snippets.

Mitigation: Keep the result labeled exploratory offline proxy. Preserve ActionMap, fixed-prior TCA, hard learned-target TCA, LoRA attribution, oracle-target upper bound, and TCA-Select ablation in follow-up evaluations. Require a separate green rollout risk gate and simulator benchmark evidence before any paper-grade success claim.

Current result: fixed-prior TCA + LoRA beat ActionMap + LoRA in `3 / 3` seeds with mean advantage `0.427353`, std `0.002126`; wrong-target proxy improved in `3 / 3` seeds. Rollout and paper claims remain false.

## 64-Record Learned Target Head Bottleneck Risk

Risk: Fixed-prior TCA remains strong, but hard learned-target TCA does not match the fixed-prior variant at 64 records.

Impact: The current formulation may rely on a target prior rather than a learned target head that generalizes from instruction-derived features.

Mitigation: Keep hard learned-target TCA as a required baseline, do not hide it behind fixed-prior results, and prioritize learned target-head redesign or calibration as the next method milestone.

Current result: hard learned-target TCA + LoRA standard proxy mean/std was `0.374657 / 0.230835`, with wrong-target proxy mean/std `0.5625 / 0.270031`, while fixed-prior TCA + LoRA reached `0.856646 / 0.002967` and wrong-target `0.0 / 0.0`.

## 64-Record TCA-Select Core-Contribution Kill Risk

Risk: TCA-Select again shows no nontrivial gain at 64 records across three seeds.

Impact: Keeping TCA-Select as the main contribution could weaken the research story by adding an unsupported mechanism on top of the stronger fixed target-prior action-conditioning evidence.

Mitigation: Treat TCA-Select as non-core or killed as a central contribution unless a future targeted selector stress test produces nontrivial gain beyond the corresponding non-select prior. The main next research target should be learned target-head/prior robustness.

Current result: TCA-Select nontrivial gain count was `0 / 3` at 64 records.

## Fixed-Prior Target-Concentration Risk

Risk: The 64-record publishability audit shows that fixed-prior TCA gains are not broad across every target group.

Impact: The method could look strong in aggregate because it corrects ActionMap's wrong-target behavior on target `1`, while offering no meaningful gain or slight underperformance on target `0`.

Mitigation: Do not proceed directly to limited rollout as if the fixed-prior gain is uniformly broad. Preserve per-target breakdowns in future reports, and redesign or calibrate the learned target head / target prior robustness while keeping ActionMap and hard learned-target baselines.

Current result: fixed-prior TCA + LoRA beat ActionMap + LoRA across all seeds on `8 / 9` eval task groups, but only `1 / 2` target groups. Target `0` mean standard-proxy delta was `-0.000098`; target `1` mean delta was `+0.878226`.

## Prior-Source Assumption Risk

Risk: The fixed-prior audit is valid only under the assumption that candidate/task natural-language text is available at test time.

Impact: If future rollout or benchmark inference cannot access equivalent non-privileged candidate text, the fixed-prior TCA result would become metadata-assisted rather than a valid method result.

Mitigation: Carry forward the prior-source audit fields in every scaled or rollout evaluation: BDDL metadata, eval labels, dataset target labels, task id/filename/manifest target proxy, and test-time availability. Downgrade the result if any non-oracle prior uses unavailable information.

Current result: instruction-text prior and fixed learned+text fusion are classified as `A_valid_test_time_semantic_prior`; both avoid BDDL metadata, eval labels, dataset target labels, filenames, task ids, and manifest target fields at inference. Fixed fusion still uses train-split target labels for supervised target-head training.

## Selector No-Headroom Risk

Risk: TCA-Select may have no useful headroom on the current fixed-prior candidate pool.

Impact: Continuing to treat TCA-Select as a central contribution could distract from the stronger target-prior-conditioned action decoding result and weaken publishability.

Mitigation: Kill TCA-Select as a core contribution unless a future targeted selector stress test or revised candidate generator shows a meaningful oracle selector upper-bound gap. Keep it as a secondary ablation only.

Current result: current TCA-Select turnover rate was `0.0`, oracle selector delta over non-select fixed-prior TCA was `0.0`, candidate diversity was not collapsed, and score diversity was non-degenerate.

## Representation-Collapse Overclaim Risk

Risk: The representation sensitivity audit did not extract full VLA hidden states, only cached proxy `hidden_tokens`.

Impact: Claiming target-information collapse would overstate the evidence and could undermine research integrity.

Mitigation: State that representation collapse is unsupported by this audit. Frame the current supported result as target-prior reinjection/action-pathway grounding and wrong-target correction. Require a separate safe hidden-state extraction audit before any collapse claim.

Current result: proxy target-swap cosine mean/std was `-0.094343 / 0.288315`, proxy L2 mean/std was `3.071515 / 0.491991`, and full hidden extraction was false.

## Target-Prior Reinjection Validity Risk

Risk: Fixed-prior TCA remains strong only under the assumption that candidate/task natural-language target text is available at test time.

Impact: If rollout inference cannot access equivalent non-privileged candidate text, the method would become metadata-assisted rather than a valid test-time method.

Mitigation: Preserve prior-source audit fields in the next rollout diagnostic. Stop or downgrade the method if BDDL metadata, eval labels, dataset target labels, task IDs, filenames, or manifest target fields become inference-time target proxies.

Current result: fixed-prior TCA + LoRA beat ActionMap + LoRA by `+0.427337` standard proxy and `-0.5` wrong-target proxy, while still depending on explicit non-leaking semantic target prior text.

## Fixed-Prior Rollout Action-Bridge Risk

Risk: The current fixed-prior offline ActionMap/TCA proxy records use only `4D` action prefixes, while LIBERO/RoboSuite rollout expects `7D` actions.

Impact: Running a simulator diagnostic now would test an undefined bridge rather than TCA-Map behavior. Any zero reward or wrong-target movement could be caused by missing rotation/gripper/coordinate mapping rather than the method.

Mitigation: Do not run the limited fixed-prior rollout diagnostic until a `7D` fixed-prior rollout-action path is implemented and validated against local HDF5. The bridge must report action dimension, scale, clipping, gripper command statistics, rotation/coordinate convention, and whether ActionMap/TCA actions differ before simulator stepping.

Current result: `scripts\138_gate_fixed_prior_rollout_readiness.ps1` reports risk gate `red`, simulator plumbing green, target-prior source green, and action bridge red with `unsupported action dimension mapping: policy_dim=4, env_action_dim=7`.

## Limited Fixed-Prior Rollout Zero-Signal Risk

Risk: The first bounded fixed-prior rollout diagnostic completed with valid `7D` action stepping but zero reward and zero success for ActionMap-style, fixed-prior TCA, and oracle-upper-bound variants.

Impact: Offline fixed-prior proxy gains may not transfer to rollout behavior, or the diagnostic may still be too short, too proxy-like, or insufficiently aligned with demonstration replay to expose task progress.

Mitigation: Treat the result as partial action-bridge support only. Before any larger rollout, run the smallest direct diagnosis that checks whether HDF5 demonstration-aligned replay can produce reward/success or object/EEF target-directed movement under the same task/init-state conditions. Do not make paper-grade rollout claims from the current diagnostic.

Current result: readiness gate green; `30` simulator steps executed; reward `0.0`, success `false`; fixed-prior TCA EEF displacement exceeded the ActionMap-style mean baseline but did not improve reward or success.


## Short-Horizon Sparse-Reward Misdiagnosis Risk

Risk: A bounded rollout diagnostic with horizons `10`, `25`, or `50` can return zero reward for all variants even when the HDF5 expert demonstration would only receive reward much later.

Impact: The project could incorrectly classify fixed-prior TCA or the action bridge as failed when the actual issue is sparse reward and insufficient horizon.

Mitigation: Before scaling learned-policy rollout variants, run a separately bounded expert replay sanity check up to the HDF5 first positive reward/done index for one task. Treat any result before that check as diagnostic plumbing evidence only, not standard success or paper-grade evidence.

Current result: zero action, ActionMap-style mean, HDF5 expert replay, and fixed-prior proxy actions all had reward `0.0` and success `false` through 50 steps. The HDF5 first positive reward/done index is `271`, so the current blocker is classified as `sparse_reward_or_short_horizon`.

## Naive Target-Distance Rollout Metric Risk

Risk: Matching an instruction to one object-position key, such as `moka_pot_1_pos`, may not capture the real task objective, gripper phase, contact dynamics, or multi-object goal state.

Impact: A variant can appear more target-directed by moving the end effector closer to the named object while still failing the actual task, or an expert trajectory can look weak early because it first moves toward an intermediate subgoal.

Mitigation: Treat object-distance movement as a secondary diagnostic only. Preserve reward/success, expert replay, init-state, gripper, rotation, and coordinate checks as primary rollout sanity gates. Report when wrong-target object keys are unavailable rather than inventing a proxy.

Current result: the intended object metric matched `moka_pot_1_pos`, but wrong-target movement was unavailable. At 50 steps, ActionMap-style mean reduced the matched distance more than HDF5 expert/fixed-prior replay under this naive metric, so the metric does not currently support a fixed-prior target-directed movement advantage.

## Default-Reset Replay Misinterpretation Risk

Risk: HDF5 expert actions can succeed from the exact demonstration init state but fail from a default environment reset.

Impact: A learned-policy or fixed-prior rollout run from default reset could fail because the initial state distribution is mismatched, not because the action bridge or method is invalid.

Mitigation: Future longer-horizon method rollout diagnostics must report whether they use matched HDF5 init states. Do not compare method actions against expert replay unless both use the same reset/init-state path. Treat default-reset rollouts as a separate diagnostic gate.

Current result: exact-init HDF5 expert replay reached reward/done/success at observed index `260`; default-reset expert replay stayed at reward `0.0` and success `false`.

## Expert Replay Timing Mismatch Risk

Risk: The simulator replay reached reward/done/success at observed index `260`, while the HDF5 file records first reward/done at index `271`.

Impact: Minor timing differences may affect horizon selection and step-aligned comparisons if treated as exact equality.

Mitigation: Treat the bridge as green for diagnostic rollout because raw expert actions succeeded, but report the timing mismatch in every follow-up. Use a horizon around the expert success window rather than assuming exact index equality.

Current result: exact-init expert replay reward sum was `1.0`, final success was `true`, and zero-action control stayed at reward `0.0` / success `false`.

## Expert-Action Leakage In Rollout Diagnostics

Risk: A fixed-prior or TCA rollout candidate can appear successful because it copies future HDF5 expert actions at the same timestep, rather than because a deployable policy or decoder generated the action online.

Impact: The project could accidentally claim rollout-level method support from expert/candidate replay, which would be invalid as closed-loop policy evidence.

Mitigation: Every rollout diagnostic must report action source, exact/near match rate to HDF5 expert actions, action L2 to expert, and whether future HDF5 actions unavailable at deployment time are used. Successful candidate replay must be labeled as candidate-replay / action-bridge evidence only.

Current result: fixed-prior TCA candidate replay has near-match rate `1.0` and mean L2 `0.0` to the HDF5 expert sequence. It succeeds in matched-init replay, but fixed-prior valid rollout-level support is `false` because it uses future HDF5 expert actions.

## Offline Candidate-Replay Overclaim Risk

Risk: ActionMap-style and hard learned-target rollout proxies can use future positive/counterfactual HDF5 action sequences or aggregates while being described as method rollouts.

Impact: Even failed or weak rollout results could be misinterpreted, and successful results would not be deployment-valid.

Mitigation: Use the wording `candidate-replay diagnostic / action-bridge evidence only; not closed-loop policy rollout success` unless actions are generated online by a non-leaking policy/head at inference time.

Current result: ActionMap-style actions are a mean aggregate of future positive/counterfactual HDF5 sequences and hard learned-target proxy actions come from future counterfactual HDF5 candidates. Both are invalid for closed-loop method rollout claims.

## Online ActionMap/TCA 7D Head Absence Risk

Risk: current ActionMap/TCA rollout variants do not generate deployable 7D LIBERO actions online. The smoke heads emit 4D actions and the offline NumPy heads consume HDF5-derived proxy features rather than current simulator observations.

Impact: any rollout-level method claim would either silently invent missing rotation/gripper dimensions or fall back to future-HDF5 candidate replay, both of which would invalidate the evidence.

Mitigation: require a non-leaking online 7D ActionMap/TCA diagnostic head before any ActionMap/TCA rollout claim. Keep HDF5 actions limited to expert replay upper bound, training data, offline labels, and action-distribution references.

Current result: native SmolVLA produced online actions in a bounded 25-step exact-init diagnostic, but ActionMap/TCA produced no valid online 7D source. Fixed-prior TCA valid rollout-level support remains `false`.
