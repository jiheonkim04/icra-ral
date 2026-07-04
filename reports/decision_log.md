# Decision Log

## SmolVLA-First Local Path

Decision: Use SmolVLA as the first real-adapter smoke target.

Reason: The local machine has an RTX 5080 with 16GB VRAM and 24GB system RAM. SmolVLA is the lower-cost path for interface validation, cached features, ActionMap/TCA-Map heads, and later tiny pilots.

Consequence: OpenVLA-OFT remains important as a paper-grade baseline target, but not as the first local execution target.

## No Large Local OpenVLA-OFT

Decision: Large OpenVLA-OFT local fine-tuning is forbidden.

Reason: Local GPU and RAM limits make full OpenVLA-OFT fine-tuning, large sweeps, and full rollouts too risky for the low-compute path.

Consequence: Any OpenVLA-OFT local use must be separately approved and limited to frozen/load smoke unless a later explicit branch changes the policy.

## Offline Proxy Is Not Standard Success

Decision: Offline proxy metrics must not be called standard success.

Reason: Paper-grade standard success requires simulator rollouts. Offline action, heatmap, target, and counterfactual metrics are engineering proxies.

Consequence: Use names such as `offline_standard_proxy` or `standard_proxy_score`.

## Path-Ready Versus Adapter-Smoke-Ready

Decision: SmolVLA readiness states are separated.

Reason: An empty checkpoint directory can exist but still be unusable.

Consequence:

- path-ready means the path exists,
- checkpoint-complete means config/tokenizer/weights exist,
- adapter-smoke-ready requires checkpoint completeness plus cache roots, lightweight guard success, and memory estimate.

## Windows Bash Shim Handling

Decision: Bash-specific tests reject the WindowsApps `bash.exe` shim.

Reason: The WindowsApps shim can resolve through `PATH` but exit with status 2 instead of behaving like GNU Bash.

Consequence: Tests use a real Bash from `BASH_EXE`, `PATH`, Git Bash, or WSL-style environments when available; otherwise the Bash-specific test skips clearly.

## Explicit Python Path Requirement

Decision: Use `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe` for Python-backed validation.

Reason: Plain `python` may resolve to the Microsoft Store alias or fail in Windows shells.

Consequence: Validation commands should use the explicit interpreter unless `python` is first verified to resolve correctly.

## Distributional TCA-Select As Main Inference Trick

Decision: The publishable low-compute method requires Distributional TCA-Select.

Reason: TCA-Map alone risks looking like ActionMap plus a target head. Distributional TCA-Select adds inference-time target/action distribution consistency without external verifiers or privileged simulator state.

Consequence: LoRA/QLoRA are required experimental tracks after head-only validation, but not the core novelty.

## Self-Check Gate Policy

Decision: Codex must self-check routine project state and only ask the user at dangerous gates.

Reason: Branch, commit, git status, pytest, safe runner, asset path readiness, checkpoint file completeness, and script policy fields can be checked from the repository, filesystem, git, and existing scripts.

Consequence: Codex should not ask whether checkpoint files were placed or whether readiness/pytest/safe runner should be checked. Codex should inspect, report, update state/action docs if needed, and stop only at asset gates, checkpoint-file gates, validation failures, external installation/credential requirements, or risk gates that fail or cannot be evaluated.

## Official SmolVLA Checkpoint Source

Decision: Use `lerobot/smolvla_base` as the official SmolVLA checkpoint source for local acquisition.

Reason: The previous plan identified the required local layout and readiness checks but did not name a specific official source. The source ambiguity is now resolved.

Consequence: SmolVLA checkpoint acquisition may use `ALLOW_DOWNLOADS=1` only for `lerobot/smolvla_base` and only for files needed under `C:\assets\checkpoints\smolvla`. This does not authorize OpenVLA-OFT downloads, datasets, token access, model inference, heavy VLA imports, GPU jobs, training, or rollouts.

## SmolVLA Source Acquisition Result

Decision: Treat `lerobot/smolvla_base` acquisition as complete for the approved source.

Reason: The acquired source contains `config.json`, `model.safetensors`, processor JSON files, and processor safetensors. Its preprocessor references `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.

Consequence: The referenced tokenizer/processor dependency requires a source/size/license/token risk assessment before acquisition.

## SmolVLA Tokenizer Dependency Acquisition Result

Decision: Acquire tokenizer/processor/config files from `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` and update readiness checks to recognize that external dependency under `HF_HOME`.

Reason: `policy_preprocessor.json` in `lerobot/smolvla_base` names this dependency for tokenization/model support, and the user explicitly approved tokenizer/processor dependency acquisition only.

Consequence: Full SmolVLM2 model weights remain forbidden for this step and were avoided. `ready_for_smolvla_adapter_smoke=true` now means file/interface readiness only. It does not authorize heavy imports, model loading, inference, GPU execution, training, rollouts, OpenVLA-OFT, or paper-grade claims.

## SmolVLA Load-Only Smoke Planning Guard

Decision: Add a planning-only SmolVLA load-only smoke guard before any heavy import/model-load task.

Reason: Readiness was true, but actual SmolVLA/SmolVLM loading crossed the heavy import/model-load gate before the later bounded risk policy.

Consequence: `scripts\15_plan_smolvla_load_only_smoke.ps1` may run safely without `ALLOW_HEAVY_IMPORT=1`. It writes a planning report, refuses to run when `ALLOW_HEAVY_IMPORT=1` is already set, and does not import SmolVLA, load models, run inference, train, rollout, download assets, access secrets, or execute OpenVLA-OFT.

## SmolVLA Load-Only Execution Scaffold

Decision: Add a bounded load-only smoke execution scaffold that stops before unsafe runtime behavior.

Reason: Readiness is true, but local runtime packages for SmolVLA loading were missing at scaffold time. Installing large packages or changing CUDA/PyTorch is a hard-stop gate.

Consequence: `scripts\16_smolvla_load_only_smoke.ps1` and `tca_map.smolvla.load_only_smoke` check gates, files, runtime dependencies, and memory policy. They report blockers without downloading assets, importing heavy VLA code, loading a model, running inference, training, rollouts, or OpenVLA-OFT.

## SmolVLA Runtime Dependency Boundary

Decision: Add a check-only runtime dependency script and a separate install plan.

Reason: Local SmolVLA files were ready, but `torch`, `transformers`, `lerobot`, and `safetensors` were not installed in the environment. Installing large packages or changing CUDA/PyTorch versions requires a package/runtime risk assessment and remains blocked if it needs system-wide changes.

Consequence: `scripts\17_check_smolvla_runtime_deps.ps1` reports package readiness without installing anything. Any actual install must have pinned versions, rollback/validation steps, and a green package/runtime risk assessment.

## SmolVLA Runtime Install Risk Boundary

Decision: Add a planning-only runtime install risk report before any package installation or CUDA/PyTorch changes.

Reason: The environment is missing SmolVLA runtime packages, but installing PyTorch, LeRobot, Transformers, Safetensors, Accelerate, or Hugging Face Hub dependencies can destabilize the local Windows/CUDA setup.

Consequence: `scripts\18_plan_smolvla_runtime_install.ps1` may run safely as a check-only planner. It refuses dangerous gates and does not install packages, download assets, import heavy VLA models, load models, infer, train, rollout, access tokens, or execute OpenVLA-OFT.

## SmolVLA Runtime Install Execution Result

Decision: Complete the explicitly approved SmolVLA runtime package install in the local `tca_map` environment.

Reason: The user approved runtime package installation after the planner identified the missing SmolVLA runtime packages.

Consequence: The environment now has `torch==2.10.0+cu128`, `torchvision==0.25.0+cu128`, `transformers==4.57.6`, `lerobot==0.4.4`, `safetensors==0.8.0`, `accelerate==1.14.0`, and `huggingface-hub==0.35.3`. The later bounded load-only debug path also identified and installed `num2words==0.5.14` for the SmolVLM processor. This clears the runtime dependency gate only. It does not authorize inference, GPU execution, training, rollouts, simulator execution, OpenVLA-OFT, token access, package upgrades, CUDA toolkit changes, or paper-grade claims outside the current risk-assessed bounded pilot policy.

## Feature Cache Interface Before Real Extraction

Decision: Define and test the feature-cache file contract with dummy hidden tokens before any real SmolVLA extraction.

Reason: Head-only ActionMap/TCA-Map work needs a stable cached-feature interface, but real SmolVLA extraction is blocked by runtime install and heavy import gates.

Consequence: `scripts\19_plan_feature_cache.ps1` and `tca_map.features.cache` may write dummy caches under ignored `runs\` paths. They do not download assets, run GPU jobs, import heavy models, load models, infer, train, rollout, or execute OpenVLA-OFT.

## Eval-Only Cached-Feature Smoke

Decision: Add an eval-only cached-feature smoke before any head training.

Reason: The next safe interface risk is whether cached hidden-token records can feed TCA-Map heads and offline proxy metrics without invoking SmolVLA or training.

Consequence: `scripts\25_eval_feature_cache_smoke.ps1` may prepare a dummy cache and compute offline proxy metrics. It does not download assets, run GPU jobs, import heavy VLA models, load models, perform VLA inference, train, rollout, or execute OpenVLA-OFT.

## Tiny Head-Only Pilot Approval Boundary

Decision: Add a planning-only gate for the first tiny ActionMap/TCA-Map head-only pilot.

Reason: The configs are within the low-compute policy, but any actual head training still requires a green bounded-training risk assessment.

Consequence: `scripts\26_plan_tiny_head_only_pilot.ps1` checks the configs and reports that training is not safe to run yet. It does not download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, or execute OpenVLA-OFT.

## Consolidated Hard-Stop Status

Decision: Add a summary-only risk-gate status command.

Reason: The next meaningful steps require consistent risk-gate reporting across runtime install, heavy import/load-only smoke, and tiny training plans.

Consequence: `scripts\27_summarize_hard_stop_status.ps1` records the current risk-gate choices without installing packages, downloading assets, running GPU jobs, importing heavy VLA models, loading models, inferring, training, rolling out, accessing tokens, executing simulators, executing OpenVLA-OFT, or making paper-grade claims.

## SmolVLA Autonomous Pilot Risk Envelope

Decision: Treat the expected low-compute SmolVLA pilot path as autonomous when it stays inside the bounded risk envelope.

Reason: The checkpoint/dependency files are ready, runtime dependencies are installed, safe runner and pytest pass, and repeated approval prompts for predictable load-only/interface/tiny-smoke steps prevent autonomous research-engineering progress.

Consequence: Codex should continue without asking before bounded SmolVLA load-only heavy import/model construction, load-only debugging, one synthetic or dummy single-sample interface smoke, tiny feature-cache/interface validation, and tiny local pilots with frozen backbone, max 300 steps after a stable smaller smoke, max 200 samples, max 30 minutes, and max 14GB VRAM. Codex may set task-local gates such as `ALLOW_HEAVY_IMPORT=1` or `ALLOW_TINY_TRAINING=1` only inside a green risk-assessed bounded task. Codex must still stop before OpenVLA-OFT download/import/load/execution, dataset/simulator/rollout work whose risk assessment fails or is ambiguous, real benchmark claims, training outside the bounded local pilot budget, jobs over 30 minutes, more than 14GB VRAM, major CUDA/PyTorch changes, unplanned large package installs, token/secret/login requirements, multi-seed experiments, paper-level empirical claims, external submission/upload/publishing, or destructive deletion outside repository/approved cache cleanup.

## SmolVLA Load-Only Smoke Execution Result

Decision: Treat the bounded CPU SmolVLA load-only smoke as passed.

Reason: The local checkpoint, local tokenizer/processor dependency, runtime packages, memory policy, and heavy-import gate were sufficient to construct the SmolVLA policy from local files with `load_vlm_weights=false`.

Consequence: This is an engineering smoke result only, not a paper-grade result. It performed no inference, training, rollout, OpenVLA-OFT execution, token access, or downloads. The next autonomous step is a single-sample interface smoke with synthetic or dummy inputs inside the same bounded pilot policy.

## SmolVLA Single-Sample Interface Smoke Scaffold

Decision: Add a bounded single-sample SmolVLA interface smoke after load-only construction passed.

Reason: The next interface risk is whether the local checkpoint, tokenizer, synthetic image/state/text batch, and policy action interface agree end to end.

Consequence: `scripts\28_smolvla_single_sample_interface_smoke.ps1` requires `ALLOW_HEAVY_IMPORT=1` and `ALLOW_SINGLE_SAMPLE_INFERENCE=1` inside the bounded task, uses CPU by default, and writes an ignored report. It must not download assets, train, rollout, access simulator/datasets/tokens, execute OpenVLA-OFT, or make paper claims.

## SmolVLA Single-Sample Interface Smoke Result

Decision: Treat the bounded CPU single-sample interface smoke as passed.

Reason: The local checkpoint, local tokenizer, synthetic image/state/text batch, and one CPU `select_action` call produced a finite action tensor with shape `[1, 6]`.

Consequence: This is an engineering interface smoke only. It is not a benchmark or paper result. The next autonomous step is tiny feature-cache/interface validation without training, rollouts, simulator execution, OpenVLA-OFT, dataset evaluation, token access, or paper claims.

## Dummy Feature-Cache Interface Validation Result

Decision: Treat the dummy feature-cache planner and eval-only cached-feature smoke as passed.

Reason: The cache contract wrote and validated `manifest.json` plus `features.jsonl`, then the eval-only path consumed 4 dummy records through the ActionMap/TCA-Map head and offline metric interface.

Consequence: This is still not real SmolVLA feature extraction and not paper evidence. It clears the dummy cached-feature/head contract. The next autonomous step is a tiny head-only smoke runner with hard caps and no rollout, simulator, OpenVLA-OFT, real dataset evaluation, or paper claims.

## Tiny Head-Only Smoke Runner

Decision: Add a bounded tiny head-only smoke runner over cached/dummy feature records.

Reason: After cached-feature eval passed, the next safe interface risk is whether head-only optimization and offline proxy metric plumbing work without importing SmolVLA, using GPU, touching rollouts, or training a backbone.

Consequence: `scripts\29_tiny_head_only_smoke.ps1` requires `ALLOW_TINY_TRAINING=1` only inside the bounded task and refuses download, heavy-import, GPU, rollout, runtime-install, and single-sample inference gates. It trains tiny CPU NumPy ActionMap/TCA-Map heads for at most 100 steps and writes an ignored report. Passing it is interface validation only, not a paper-grade result.

## Tiny Head-Only Smoke Result

Decision: Treat the bounded tiny head-only smoke as passed.

Reason: The runner trained tiny CPU NumPy ActionMap and TCA-Map heads over 4 cached/dummy records for 16 steps, stayed under the 100-step and 900-second caps, and produced finite offline proxy metrics.

Consequence: This validates only the cached-feature head-only optimization path. It did not download assets, run GPU jobs, import or load SmolVLA/OpenVLA, perform VLA inference, train a backbone, rollout, execute simulators, or make paper claims. The next safe non-heavy task is a go/no-go/status summary; real dataset training, rollouts, simulator execution, OpenVLA-OFT, and paper claims remain hard-stop gates.

## Go/No-Go Status Summary

Decision: Add a summary-only go/no-go report for the next larger experimental stage.

Reason: The safe local smoke stack has passed, but the next larger stage would require real dataset setup, simulator rollout, larger training, or OpenVLA-OFT decisions.

Consequence: `scripts\31_generate_go_no_go_report.ps1` reads local reports and emits a no-go for paper-grade or larger experimental claims until the user explicitly approves exactly one true next gate. It does not download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims.

## Required LoRA/QLoRA Experiment Tracks

Decision: LoRA/QLoRA are required experimental tracks, but not the main novelty.

Reason: A publishable low-compute VLA paper should show that TCA-Map works both in head-only mode and under parameter-efficient adaptation.

Consequence: The required matrix now includes ActionMap + LoRA, TCA-Map + LoRA, TCA-Map + LoRA + Distributional TCA-Select, and a QLoRA feasibility arm if memory/tooling allows. Full backbone fine-tuning remains forbidden locally, and LoRA gains must be separated from TCA-Map and Distributional TCA-Select gains.

## LoRA Adapter Construction Planner

Decision: Add a planning-only LoRA adapter construction/readiness scaffold.

Reason: Required LoRA tracks need an adapter construction boundary before any LoRA tiny smoke can be considered.

Consequence: `scripts\32_plan_lora_adapter_construction.ps1` validates LoRA/QLoRA configs and local checkpoint file inputs without downloading assets, running GPU jobs, importing heavy VLA models, loading models, inferring, training, rolling out, executing simulators, accessing tokens, executing OpenVLA-OFT, or making paper claims.

## LoRA Tiny Smoke Scaffold

Decision: Add a planning-only scaffold for the required LoRA tiny smoke.

Reason: The required LoRA track needs an explicit tiny-smoke boundary before any adapter update is allowed.

Consequence: `scripts\33_plan_lora_tiny_smoke.ps1` validates the LoRA/QLoRA configs and future tiny-smoke envelope. It is planning-only, but a later bounded runner may execute tiny LoRA smoke under the risk-assessed local pilot limits. The scaffold itself does not construct adapters, train, download assets, run GPU jobs, import heavy VLA models, load models, infer, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims.

## TCA-Map + LoRA Comparison Plan

Decision: Add a planning-only comparison matrix for the required LoRA arms.

Reason: The required LoRA track must not let LoRA gains obscure the TCA-Map or Distributional TCA-Select contribution.

Consequence: `scripts\34_plan_lora_comparison.ps1` fixes the ActionMap + LoRA, TCA-Map + LoRA, TCA-Map + LoRA + Distributional TCA-Select, and QLoRA-if-feasible comparisons without training, constructing adapters, downloading assets, running GPU jobs, importing heavy VLA models, loading models, inferring, rolling out, executing simulators, accessing tokens, executing OpenVLA-OFT, or making paper claims.

## QLoRA Feasibility Check

Decision: Add a check-only QLoRA feasibility gate.

Reason: QLoRA is a required feasibility track if memory/tooling allows, but it must not force unapproved package installs or CUDA/PyTorch changes.

Consequence: `scripts\35_check_qlora_feasibility.ps1` checks config validity and local QLoRA tooling availability without installing packages, downloading assets, running GPU jobs, importing heavy VLA models, loading models, inferring, training, rolling out, executing simulators, accessing tokens, executing OpenVLA-OFT, or making paper claims. It keeps `safe_to_run_qlora_now=false`.

## LoRA/QLoRA Go/No-Go Update

Decision: Extend the go/no-go generator to summarize LoRA/QLoRA planning readiness.

Reason: After the required LoRA/QLoRA planning stack, the project needs a clear distinction between risk-assessed bounded local pilots and larger paper-grade stop gates.

Consequence: `scripts\31_generate_go_no_go_report.ps1` now reports LoRA adapter planning, tiny-smoke scaffold, comparison planning, QLoRA feasibility status, `ready_for_bounded_local_pilot`, and `blocked_for_larger_paper_grade_stage`. It remains summary-only and no-go for paper-grade or larger experimental stages, but it must not block bounded local SmolVLA-only pilots inside the risk-assessed limits.

## Bounded Local Pilot Risk Envelope

Decision: Treat bounded local SmolVLA-only pilot experiments as autonomous when the risk assessment is green.

Reason: The safe local smoke stack is complete, and stopping after every smoke prevents meaningful low-compute research progress.

Consequence: Codex should autonomously continue through bounded local head-only, LoRA, TCA-Map + LoRA, Distributional TCA-Select, QLoRA feasibility, offline proxy, and tiny comparison tasks when the risk assessment stays within max 300 steps after stable smaller smoke, max 200 samples, max 30 minutes, max 14GB VRAM, batch size 1, frozen backbone except LoRA adapter weights, no rollout unless separately risk-assessed, no simulator unless separately risk-assessed, no OpenVLA-OFT, no full fine-tuning, and no paper claim. Codex must still stop before true external irreversible/OpenVLA/paper-claim gates.

## Head-Only Tiny Comparison Report

Decision: Add a bounded local ActionMap vs TCA-Map tiny comparison report.

Reason: The tiny head-only smoke already trains both heads, but the autonomous pilot path needs an explicit comparison artifact before moving to LoRA diagnostics.

Consequence: `scripts\36_compare_head_only_tiny_pilot.ps1` reads the existing tiny smoke report and emits offline proxy deltas only. It does not download assets, run GPU jobs, train, import heavy VLA models, load models, infer, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims.

## Tiny LoRA Smoke Runner

Decision: Add a bounded local tiny LoRA smoke runner.

Reason: LoRA/QLoRA are required experimental tracks after head-only validation, and the project needs a minimal adapter-update check before larger LoRA comparisons.

Consequence: `scripts\37_tiny_lora_smoke.ps1` requires `ALLOW_TINY_TRAINING=1` and trains only tiny NumPy LoRA adapter matrices over cached/dummy features. It covers ActionMap + LoRA, TCA-Map + LoRA, and TCA-Map + LoRA + Distributional TCA-Select as offline proxy diagnostics only. It does not download assets, run GPU jobs, import heavy VLA models, load models, infer, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims.

## Tiny LoRA Comparison Report

Decision: Add a bounded local tiny LoRA comparison report.

Reason: The required LoRA track needs an explicit analysis artifact that separates TCA-Map + LoRA gains from Distributional TCA-Select gains.

Consequence: `scripts\38_compare_tiny_lora_pilot.ps1` reads the existing tiny LoRA smoke report and emits offline proxy deltas only. It does not download assets, run GPU jobs, train, import heavy VLA models, load models, infer, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims.

## Consolidated Local Pilot Status

Decision: Add a summary-only local pilot status report.

Reason: The bounded local offline proxy tier now has several runtime reports, and the repository needs one artifact that says what has passed and what requires risk assessment next.

Consequence: `scripts\39_generate_local_pilot_status.ps1` reads existing reports and writes a consolidated status without training, downloading, running GPU jobs, importing heavy VLA models, loading models, inferring, rolling out, executing simulators, accessing tokens, executing OpenVLA-OFT, or making paper claims. It marks the next meaningful steps as risk-assessed gates rather than routine approval prompts.

## Risk-Assessed Autonomous Execution

Decision: Replace broad approval-based hard-stops with risk-assessed autonomous execution.

Reason: Codex can inspect many risks directly: source, size, disk, RAM, VRAM, runtime, dependencies, license/token requirements, and repository policy. Asking for permission merely because a task involves downloads, GPU, training, datasets, simulator readiness, or bounded rollout slows autonomous research-engineering work.

Consequence: Codex must write or print a short risk assessment before bounded download/GPU/training/dataset/simulator/rollout steps. If source and setup are clear and the task is inside budget, Codex proceeds autonomously. Codex stops only when risk is ambiguous or outside budget, when token/secret/payment/license/system-level/external irreversible action is required, when OpenVLA-OFT execution is involved, or when paper-level claims would be made. The default budgets are 80GB download soft limit with at least 100GB disk remaining, <=14GB VRAM, <=30 minutes runtime, batch size 1, SmolVLA-only bounded training with frozen backbone or LoRA/QLoRA adapters, and <=300 local pilot steps after smaller smoke is stable.

## LIBERO Dataset Risk Planner

Decision: Add a planning-only LIBERO/LIBERO-CF-style dataset risk planner.

Reason: The next meaningful step after the local SmolVLA smoke stack is to evaluate whether real dataset readiness or tiny-subset setup is safe without drifting into downloads, simulator execution, rollout, or paper-grade claims.

Consequence: `scripts\42_plan_libero_dataset_risk.ps1` checks local LIBERO paths, a shallow dataset-file probe, optional official source/size metadata, disk budget, and token/license/payment gates. It writes ignored runtime reports and does not download, train, rollout, import simulators/heavy VLA models, execute OpenVLA-OFT, or make paper claims.

## Simulator Readiness Planner

Decision: Add a planning-only simulator readiness risk planner before any LIBERO/RoboSuite/MuJoCo import, render smoke, or rollout.

Reason: The project needs simulator evidence eventually, but native Windows and missing local simulator paths make direct execution risky. The safe next step is to separate path/OS readiness from actual simulator import/render/rollout execution.

Consequence: `scripts\43_plan_simulator_readiness.ps1` checks local LIBERO and RoboSuite paths plus WSL2/Linux suitability. It writes ignored runtime reports and does not install packages, download assets, import simulators, render, rollout, run GPU jobs, train, import heavy VLA models, execute OpenVLA-OFT, access tokens, or make paper claims.

## Local Pilot Step Budget Alignment

Decision: Align the local pilot compute budget and head-only pilot configs to a 300-step maximum.

Reason: The risk-assessed autonomous policy caps bounded local pilot training at 300 steps after smaller smoke is stable, but older config files still allowed 1000 initial steps.

Consequence: `configs\compute_budget.yaml`, `configs\actionmap_head_only_lowcompute.yaml`, and `configs\tca_map_head_only_lowcompute.yaml` now use `300` as the local pilot step ceiling. Tiny smoke runners may still use narrower caps such as 100 steps.

## Bounded Local Pilot Extension

Decision: Add a bounded cached-feature local pilot extension runner.

Reason: After the smaller head-only and LoRA smokes passed, the next safe local execution step is a slightly longer cached-feature head-only smoke inside the 300-step risk policy, without real datasets or simulator execution.

Consequence: `scripts\44_bounded_local_pilot_extension.ps1` runs the existing cached-feature head-only smoke path with a stricter 100-step runner cap and a 64-step default. It writes ignored runtime reports and labels the result as offline proxy only, not standard success and not paper-grade evidence.

## Bounded Extension Status Consolidation

Decision: Include the bounded local pilot extension report in consolidated status and go/no-go summaries.

Reason: Once the bounded extension runner exists, the repository status reports should not ignore its runtime report.

Consequence: `scripts\39_generate_local_pilot_status.ps1` and `scripts\31_generate_go_no_go_report.ps1` read `reports\bounded_local_pilot_extension_report.json` when present. They remain summary-only and do not train, download, use GPU, import heavy models, rollout, execute simulators, execute OpenVLA-OFT, or make paper claims.

## LIBERO/RoboSuite Official Source Resolution

Decision: Treat official LIBERO/RoboSuite source resolution as an autonomous risk-assessed task instead of stopping because source/size metadata is missing.

Reason: The project can inspect public documentation and record official sources, expected sizes, license/token/payment status, target paths, and disk budget without running rollouts, simulators, training, heavy VLA imports, or OpenVLA-OFT.

Consequence: `scripts\45_resolve_libero_robosuite_sources.ps1` records official source candidates. LIBERO and RoboSuite code checkouts are small enough for bounded source setup, while the full official LIBERO demonstrations dataset is about 100 GB and now requires the dedicated LIBERO-only acquisition gate. `scripts\46_prepare_libero_robosuite_sources.ps1` may shallow-clone only the official code repos with task-local `ALLOW_DOWNLOADS=1`; it must not download the full dataset, run simulators, run rollouts, train, use GPU, import heavy VLA models, access tokens, execute OpenVLA-OFT, or make paper claims.

## LIBERO/RoboSuite Source Repo Setup Result

Decision: Treat bounded LIBERO/RoboSuite source repo setup as complete.

Reason: The official LIBERO and RoboSuite code repos were risk-green for shallow clone, while the full official LIBERO dataset was left for a separate acquisition gate because it is about 100 GB.

Consequence: `LIBERO_ROOT` and `ROBOSUITE_ROOT` are now path-ready under `C:\assets\repos`, and `LIBERO_DATA_ROOT` exists with a marker explaining that the full dataset was not downloaded. This clears source path setup only. It does not clear tiny offline dataset readiness, simulator import readiness as an executed result, rollout readiness, real benchmark readiness, OpenVLA-OFT, or paper claims.

## LIBERO Metadata-Only Subset Construction

Decision: Add a metadata-only LIBERO task/counterfactual manifest builder.

Reason: The official full LIBERO demonstrations dataset is too large for the current autonomous local budget, but the official source checkout contains BDDL/task metadata that is enough to validate target/counterfactual split plumbing.

Consequence: `scripts\47_build_libero_metadata_subset.ps1` may read local BDDL/task metadata and write ignored metadata reports. It must not download data, run GPU jobs, train, rollout, import simulators or heavy VLA models, execute OpenVLA-OFT, access tokens, or make paper-grade claims. Metadata-only readiness does not imply real dataset interface readiness.

## LIBERO Offline Interface Smoke Gate

Decision: Add a check-only gate for tiny local LIBERO-style data files.

Reason: After metadata-only task plumbing is available, the next safe boundary is distinguishing absent data from a tiny local data file that can be structurally read without training, rollout, simulator execution, or heavy imports.

Consequence: `scripts\48_plan_libero_offline_interface_smoke.ps1` inspects only local JSON/JSONL/NPZ/HDF5-like files under `LIBERO_DATA_ROOT` and reports `proceed` only if instruction/action-like fields are readable. In the current state it should report `stop` because no real demo files are present.

## Consolidate LIBERO Data Gates Into Status Reports

Decision: Include LIBERO metadata/offline-interface gate reports in the consolidated local pilot and go/no-go summaries.

Reason: Once the LIBERO source, metadata, and offline-interface gates exist, the main status reports should make the current blocker visible without requiring manual inspection of individual runtime reports.

Consequence: `scripts\39_generate_local_pilot_status.ps1` and `scripts\31_generate_go_no_go_report.ps1` now read the metadata subset and offline interface smoke gate reports when present. The summaries remain report-only and perform no downloads, GPU jobs, training, rollouts, simulator execution, heavy imports, OpenVLA-OFT execution, token access, or paper claims.

## Official LIBERO Data Acquisition Budget

Decision: Raise the autonomous acquisition budget only for the official LIBERO demonstrations dataset.

Reason: The local machine has about 500 GB free, and the official LIBERO dataset source recorded in the repository is about 100 GB with no token/login/payment/license click-through requirement. This makes a bounded acquisition safe if disk remains above a stricter post-download floor.

Consequence: `scripts\49_acquire_libero_data.ps1` and `tca_map.datasets.libero_data_acquisition` may acquire only `yifengzhu-hf/LIBERO-datasets` into `C:\assets\data\libero` using `C:\assets\hf_home` as cache. The task-local budget is 180 GB with at least 250 GB free disk remaining after acquisition. The command still performs no GPU jobs, training, simulator execution, rollouts, heavy VLA imports, OpenVLA-OFT execution, token access, external upload, or paper claims.

## LIBERO HDF5 Reader Gate And Rollout Semantics

Decision: Separate acquired-data readiness, HDF5 reader readiness, and rollout readiness.

Reason: After official LIBERO data acquisition, HDF5 files can exist locally while `h5py` is unavailable. Also, path/data readiness must not imply simulator rollout readiness.

Consequence: `scripts\50_check_libero_hdf5_reader.ps1` checks `h5py` availability without installing packages. `scripts\11_check_real_assets.ps1` reports path/data readiness separately and keeps `ready_for_libero_rollout=false` until a separate simulator import/render/rollout risk gate exists and passes.

## h5py As LIBERO Reader Dependency

Decision: Declare `h5py>=3.11` as the required reader dependency for local LIBERO HDF5 offline interface inspection.

Reason: The official LIBERO demonstrations are HDF5 files. Without `h5py`, the repository can confirm file presence but cannot inspect instruction/action-like fields for the offline interface smoke.

Consequence: `requirements.txt` includes `h5py>=3.11`, and `pyproject.toml` exposes a `libero` optional dependency group. Installing it still requires a green dependency risk assessment and must not change CUDA/PyTorch versions, install simulators, train, rollout, import heavy VLA models, execute OpenVLA-OFT, or make paper claims.

## h5py Reader Install And LIBERO Offline Interface Gate

Decision: Install `h5py` as a reader-only dependency after a green risk assessment and run the LIBERO offline interface smoke gate.

Reason: Official LIBERO demonstrations are present locally as HDF5 files. The environment needed a lightweight HDF5 reader to inspect action fields without simulator execution, training, rollout, model loading, or paper claims.

Consequence: `h5py 3.16.0` was installed from a Windows wheel only; no CUDA/PyTorch versions, simulator packages, VLA models, OpenVLA-OFT assets, tokens, or dataset sources were changed. `scripts\50_check_libero_hdf5_reader.ps1` reports ready, and `scripts\48_plan_libero_offline_interface_smoke.ps1` reports `ready_for_offline_interface_smoke=true`.

## Bounded LIBERO HDF5 Report Output

Decision: Keep LIBERO HDF5 interface inspection reports bounded by recording dataset counts and a sample of dataset shapes instead of every HDF5 dataset path.

Reason: Real LIBERO files contain many demonstrations and observation datasets. Dumping every HDF5 dataset shape makes terminal output and runtime reports unnecessarily large while adding no readiness value.

Consequence: `tca_map.datasets.libero_offline_interface.inspect_hdf5` now records `dataset_count`, `dataset_sample_limit`, `datasets_sample`, and `action_dataset_paths_sample`. It still detects action fields for interface readiness, but keeps reports small enough for routine safe-runner use.

## LIBERO Offline Counterfactual Split Manifest

Decision: Add a tiny local LIBERO HDF5-backed counterfactual split manifest builder.

Reason: After local LIBERO HDF5 files became readable, the next safe stage is linking BDDL task metadata to acquired demo files and constructing counterfactual target/action pairs without training, simulator execution, rollout, or model loading.

Consequence: `scripts\51_build_libero_offline_counterfactual_split.ps1` writes ignored split reports from local metadata and HDF5 structure only. It marks outputs as offline proxy only and prepares the next tiny real/offline ActionMap vs TCA-Map comparison gate.

## LIBERO Offline ActionMap vs TCA-Map Proxy Comparison

Decision: Add a tiny local LIBERO ActionMap vs TCA-Map proxy comparison over HDF5 action snippets.

Reason: Once counterfactual HDF5 pairs are available, the next safe stage is validating comparison plumbing for ActionMap, TCA-Map, and TCA-Map + Distributional TCA-Select without model loading, training, simulator execution, rollout, or paper claims.

Consequence: `scripts\52_compare_libero_offline_actionmap_tca.ps1` reads the split manifest and a bounded number of HDF5 action rows, then writes ignored offline proxy reports. The report is not a trained baseline and not paper-grade evidence; it only clears the path for the required tiny real/offline LoRA comparison scaffold.

## LIBERO Offline Required LoRA Proxy Comparison

Decision: Add a tiny local LIBERO LoRA proxy comparison over HDF5 action snippets.

Reason: LoRA/QLoRA are required experimental tracks after the head-only path. After the LIBERO offline ActionMap vs TCA-Map gate passes, the next safe local step is comparing ActionMap + LoRA, TCA-Map + LoRA, and TCA-Map + LoRA + Distributional TCA-Select using bounded NumPy adapters only.

Consequence: `scripts\53_compare_libero_offline_lora.ps1` requires `ALLOW_TINY_TRAINING=1`, reads local LIBERO HDF5 snippets, trains tiny NumPy low-rank adapter matrices, and writes ignored offline proxy reports. It remains no-GPU, no-rollout, no-heavy-import, no-OpenVLA-OFT, and not paper-grade evidence.

## LIBERO Offline Bounded Pilot Report

Decision: Add a summary-only bounded pilot report for the LIBERO offline proxy ladder.

Reason: After the offline interface, counterfactual split, head-only comparison, and required LoRA comparison gates pass, the repository needs one local artifact that summarizes what is ready and what remains blocked before simulator/rollout work.

Consequence: `scripts\54_generate_libero_offline_bounded_pilot_report.ps1` reads existing ignored runtime reports only and writes an ignored bounded pilot report. It does not train, download, use GPU, import heavy VLA models, load models, infer, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims.

## Simulator Readiness Status Integration

Decision: Treat the simulator readiness planner report as a first-class input to local pilot and go/no-go summaries.

Reason: After the LIBERO offline bounded pilot passes, the current blocker is simulator import/render/rollout readiness. The consolidated status reports should show whether `scripts\43_plan_simulator_readiness.ps1` has run, which platform it selected, and why import/render/rollout remain blocked.

Consequence: `scripts\39_generate_local_pilot_status.ps1` and `scripts\31_generate_go_no_go_report.ps1` now read `reports\simulator_readiness_plan_report.json` when present. They still perform no simulator imports, render smoke, rollouts, downloads, GPU jobs, training, heavy VLA imports, OpenVLA-OFT execution, token access, or paper claims.

## Bounded Simulator Import Smoke Scaffold

Decision: Add a task-local gated simulator import-only smoke for WSL/Linux readiness.

Reason: The readiness planner can show that paths and WSL are available, but the next safe question is whether local `robosuite` and `libero` packages are import-visible without rendering, stepping environments, rolling out policies, or changing dependencies.

Consequence: `scripts\55_bounded_simulator_import_smoke.ps1` requires `ALLOW_SIMULATOR_IMPORT_SMOKE=1` after a green risk assessment and attempts only package imports in WSL. The output is ignored runtime readiness evidence, not standard success, not rollout success, and not paper-grade evidence.

Follow-up: The first local bounded run found that WSL `libero` imports but `robosuite` needs `numpy` in WSL Python. Dependency installation is not folded into the import-smoke script; it requires a separate risk-planned dependency task.

## WSL Simulator Dependency Checker

Decision: Add a check-only WSL simulator dependency report before any simulator dependency installation.

Reason: The bounded simulator import smoke found a concrete WSL dependency blocker, but WSL lacks `pip` and `ensurepip`; jumping directly to `apt` would cross into system/package-management setup.

Consequence: `scripts\56_check_wsl_simulator_deps.ps1` records WSL `python3`, `pip`, `ensurepip`, `numpy`, and missing modules from the import-smoke report. It performs no installs, downloads, render smoke, rollouts, simulator environment steps, GPU jobs, training, heavy VLA imports, OpenVLA-OFT execution, token access, or paper claims.

## WSL Simulator Dependency Ladder Standing Approval

Decision: Minimal WSL simulator dependency bootstrap is standing-approved after risk assessment.

Reason: The simulator blocker is no longer ambiguous: WSL path and `python3` probes pass, LIBERO imports, and RoboSuite is blocked by missing WSL Python packaging/dependencies such as `numpy`. Requiring repeated user approval for each small WSL dependency step stalls the bounded simulator readiness ladder.

Consequence: Codex may autonomously inspect WSL, check `python3`/`pip`/`venv`, install minimal WSL Python packaging tools, create `~/.venvs/tca_map_sim`, install minimal import-readiness Python dependencies, rerun bounded simulator import smoke, and proceed to bounded render/reset-step/tiny diagnostic stages if each stage has a green risk assessment. Codex must still stop for sudo password prompts, credentials, token/secret/login, payment/license click-through, CUDA driver/toolkit or graphics-stack changes, Windows driver changes, OpenVLA-OFT download/import/load/execution, full fine-tuning, training over 30 minutes, VRAM over 14GB, rollout beyond tiny diagnostic limits, benchmark or paper-grade claims, multi-seed experiments, external upload/submission/publishing, or deletion outside approved repo/cache cleanup.

## WSL Simulator Dependency Setup Result

Decision: Use a WSL-local venv at `~/.venvs/tca_map_sim` for simulator import-readiness dependencies.

Reason: Global WSL Python lacks `pip`, `ensurepip`, and `numpy`, but `python3 -m venv --without-pip` works and `curl` is available for venv-local pip bootstrapping. The setup can avoid sudo, apt, CUDA/driver changes, token access, render, rollout, OpenVLA-OFT, and paper claims.

Consequence: `scripts\57_setup_wsl_simulator_deps.ps1` creates or reuses the venv, bootstraps pip only if missing, and installs only bounded import-readiness Python packages there. It reuses the venv by default and offers `-ClearVenv` only for intentional clean rebuilds.

## Bounded Simulator Import-Only Smoke Result

Decision: Treat WSL simulator import-only readiness as passed for the selected venv.

Reason: The bounded task-local import smoke selected `~/.venvs/tca_map_sim/bin/python` and imported both `robosuite` and `libero`.

Consequence: This clears only the import-only readiness rung. It is not render evidence, rollout evidence, benchmark success, or paper-grade evidence. Render smoke, reset/step smoke, tiny diagnostic rollout, and all benchmark claims remain separate risk gates.

## WSL Simulator Source Link Result

Decision: Link local RoboSuite and LIBERO source checkouts into the existing WSL venv instead of creating a repo-local venv.

Reason: The selected WSL venv at `/home/jiheon/.venvs/tca_map_sim` already has Python, pip, NumPy, and MuJoCo. `robosuite` was missing from that venv, and LIBERO's nested source layout plus first-import config prompt needed a bounded local fix.

Consequence: `scripts\60_link_wsl_simulator_sources.ps1` performs offline editable linking with `--no-index --no-deps --no-build-isolation`, writes a `.pth` entry for `LIBERO/libero`, and writes noninteractive WSL `~/.libero/config.yaml` pointing to the local LIBERO source/data roots. It does not create a repo-local `.venv`, download packages, render, reset/step, rollout, train, use GPU, import heavy VLA models, execute OpenVLA-OFT, access tokens, or make paper claims.

## Simulator Render/Reset-Step Planner

Decision: Add a planning-only render/reset-step risk gate after import-only readiness.

Reason: Passing `robosuite` and `libero` imports is not enough to justify rendering, resetting an environment, stepping an environment, or running rollouts. The next risk boundary needs to require the passed import-only report and explicitly refuse execution gates during planning.

Consequence: `scripts\58_plan_simulator_render_reset.ps1` reads the readiness/import/render reports and reports whether separate bounded render-smoke or reset/step-smoke branches may be created. It performs no render, reset/step, rollout, install, download, GPU job, training, heavy VLA import, OpenVLA-OFT execution, token access, or paper claim. Current local result is ready for a separate bounded reset/step-smoke branch, while rollout remains blocked.

## Bounded Simulator Render Smoke Result

Decision: Treat the current local bounded render-smoke attempt as passed for tiny MuJoCo offscreen-render readiness.

Reason: After the user completed the WSL offscreen graphics package step, the bounded task-local render smoke produced a nonblank 64x64 RGB image under `MUJOCO_GL=osmesa`.

Consequence: `scripts\59_bounded_simulator_render_smoke.ps1` remains a tiny MuJoCo render-only check. It did not create, reset, or step LIBERO/RoboSuite environments, roll out policies, use GPU jobs, train, install packages, download assets, import heavy VLA models, execute OpenVLA-OFT, access tokens, or make paper claims. Reset/step and rollout remain separate risk gates.

## Bounded Simulator Reset/Step Smoke Result

Decision: Treat the current local bounded reset/step smoke as passed for tiny MuJoCo physics reset/step readiness.

Reason: After import-only and render-only readiness passed, the reset/step smoke ran under task-local `ALLOW_SIMULATOR_RESET_STEP=1` and performed `mj_resetData`, `mj_forward`, and 3 `mj_step` calls on a tiny in-memory MuJoCo model.

Consequence: `scripts\61_bounded_simulator_reset_step_smoke.ps1` is limited to tiny MuJoCo physics plumbing. It did not create LIBERO or RoboSuite environments, run rollouts, run policy inference, use GPU jobs, train, install packages, download assets, import heavy VLA models, execute OpenVLA-OFT, access tokens, or make paper claims. Bounded tiny diagnostic rollout remains a separate risk gate with its own task-local execution guard.

## Tiny Diagnostic Rollout Policy

Decision: Bounded tiny diagnostic rollout is allowed after a green risk assessment.

Reason: Import, render, and tiny reset/step smoke now pass, and the user updated the policy from no rollout to bounded tiny diagnostic rollout allowed inside strict limits.

Consequence: `scripts\62_plan_tiny_diagnostic_rollout.ps1` reports a max-5-task, one-episode, max-5-step envelope and authorizes execution only through the separate task-local `ALLOW_TINY_ROLLOUT=1` runner. It still keeps benchmark rollout readiness false.

## Bounded Tiny Diagnostic Rollout Result

Decision: Treat the current local bounded tiny diagnostic rollout as passed.

Reason: `scripts\63_bounded_tiny_diagnostic_rollout.ps1` ran 5 toy MuJoCo diagnostic tasks with 1 episode and 5 steps per task through the selected WSL venv. The run completed 25 total steps and reported no LIBERO/RoboSuite benchmark environment, no learned policy inference, no training, no GPU job, no download, no heavy VLA import, no OpenVLA-OFT execution, no multi-seed evaluation, and no benchmark/SOTA/paper-grade claim.

Consequence: This clears only simulator plumbing for bounded tiny diagnostic rollouts. It is not LIBERO success, not standard success, not benchmark evidence, and not paper-grade evidence. Benchmark rollouts, multi-seed rollouts, OpenVLA-OFT, full training, external uploads, and paper-level claims remain separate stop gates.

## LIBERO/RoboSuite Compatibility Alignment

Decision: Align the local simulator compatibility path to LIBERO's RoboSuite expectations inside the existing WSL venv.

Reason: LIBERO's `requirements.txt` requires `robosuite==1.4.0`, `bddl==1.0.1`, `future==0.18.2`, `easydict==1.9`, `matplotlib==3.5.3`, `cloudpickle==2.1.0`, and `gym==0.25.2`. The local RoboSuite checkout was newer than LIBERO's expected API, and `mujoco==3.10.0` broke RoboSuite 1.4's `mj_fullM` call during environment creation.

Consequence: The local `C:\assets\repos\robosuite` checkout was moved to the official `v1.4.0` tag, and the WSL venv `/home/jiheon/.venvs/tca_map_sim` was aligned with the bounded diagnostic dependencies, including `numpy==1.22.4` and `mujoco==2.3.7`. These are local environment readiness changes only. They do not authorize benchmark rollout, learned-policy rollout, OpenVLA-OFT, training, GPU jobs, external upload, or paper claims.

## Bounded LIBERO/RoboSuite Diagnostic Rollout Result

Decision: Treat the current bounded LIBERO/RoboSuite zero-action diagnostic rollout as passed.

Reason: `scripts\65_bounded_libero_robosuite_diagnostic_rollout.ps1` ran under task-local `ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT=1`, created one `libero_10` LIBERO/RoboSuite environment, reset it, stepped it 3 times with a zero action, observed a finite 64x64 `agentview_image`, and closed the environment.

Consequence: This clears only real LIBERO/RoboSuite simulator diagnostic plumbing. It is not standard success, not benchmark success, not SOTA evidence, and not paper-grade evidence. Learned-policy and tiny benchmark rollouts require separate green readiness/risk assessments before execution. Multi-seed rollouts, OpenVLA-OFT, full fine-tuning, external uploads, and unsupported paper-level claims remain stop gates.

## End-to-End Paper-Grade Autopilot Policy

Decision: Move from diagnostic-only autonomy to end-to-end risk-assessed research autopilot.

Reason: The user granted standing approval for local research-engineering actions needed to reach the strongest feasible paper-grade candidate package or a clear kill/pivot decision, as long as each nontrivial action passes automatic risk assessment and avoids human-only gates.

Consequence: Codex should no longer stop merely because the next task is bounded training, learned-policy rollout, benchmark rollout, baseline execution, report generation, visualization, or scale-up from a smaller passed stage. It must still stop for credentials, token/secret/API-key access, paid services, license click-through, external upload/submission/publishing, destructive/system-risk changes, OpenVLA-OFT execution without a separate budget, full fine-tuning outside budget, multi-seed runs before a separate budget, and unsupported empirical claims.

## Learned-Policy LIBERO Rollout Readiness Topology

Decision: Use WSL-only simulator plus policy runtime readiness as the first learned-policy rollout topology.

Reason: LIBERO/RoboSuite simulator readiness is currently established in WSL, while SmolVLA runtime readiness was established in the Windows conda environment. A Windows-policy/WSL-simulator bridge adds IPC, latency, image serialization, and synchronization risk before proving the simpler WSL-only path.

Consequence: `scripts\66_plan_libero_policy_rollout_readiness.ps1` checks whether the WSL venv can see the lightweight SmolVLA runtime modules and local SmolVLA/LIBERO assets without loading models or running rollouts. If it reports `proceed`, create a separately gated tiny learned-policy rollout runner. If it reports `reduce_scope`, prepare WSL SmolVLA runtime setup/readiness first.

## WSL SmolVLA Runtime Setup Result

Decision: Treat the WSL SmolVLA module-spec runtime readiness as passed for the selected WSL venv.

Reason: After a green risk assessment, task-local WSL venv package setup installed CPU torch/torchvision and the lightweight SmolVLA runtime modules into `/home/jiheon/.venvs/tca_map_sim`. The first setup run hit the 1800 second timeout, but no residual install process remained; follow-up probes found all required module specs present, and the setup guard rerun reported `setup_passed=true` without further installs.

Consequence: `scripts\66_plan_libero_policy_rollout_readiness.ps1` now reports the WSL-only simulator plus policy runtime topology as green. This is not model-load, inference, rollout, benchmark, SOTA, or paper-grade evidence. The next safe task is a separately gated tiny learned-policy LIBERO rollout runner.

## WSL SmolVLA Single-Action Smoke Result

Decision: Treat the bounded WSL SmolVLA single-action smoke as passed.

Reason: After green risk assessment and venv-local LeRobot runtime dependency fixes, `scripts\70_bounded_wsl_smolvla_single_action_smoke.ps1` loaded the local SmolVLA policy in WSL on CPU and produced one finite synthetic action with shape `[1, 6]`.

Consequence: This is model-load/action-interface smoke evidence only. It does not establish LIBERO success, benchmark success, standard success, SOTA, or paper-grade evidence. The next safe task is a separately gated tiny learned-policy LIBERO rollout runner capped at one task and a few steps.

## Tiny Learned-Policy LIBERO Rollout Diagnostic Result

Decision: Treat the first bounded tiny learned-policy LIBERO rollout as passed diagnostic evidence.

Reason: `scripts\72_bounded_tiny_learned_policy_rollout.ps1` ran under task-local `ALLOW_TINY_LEARNED_POLICY_ROLLOUT=1`, loaded local SmolVLA in the WSL simulator topology on CPU, created one real `libero_10` RoboSuite/LIBERO environment, made 3 policy calls, and stepped the environment 3 times without timeout or wrapper failure.

Consequence: This clears only the first learned-policy simulator-control topology. The observed diagnostic success check was `false` and reward sum was `0.0`, so this is not benchmark success, not standard success, not SOTA evidence, and not paper-grade evidence. The next safe rung is a tiny benchmark-metric diagnostic summary and then a bounded small rollout matrix if risk assessment remains green.

## Tiny Learned-Policy Metric Summary Result

Decision: Treat the tiny learned-policy metric summary as passed report-only diagnostic evidence.

Reason: `scripts\73_generate_tiny_learned_policy_metric_summary.ps1` read the existing rollout JSON, refused execution gates, and summarized the observed one-task, three-step result without any new model load, inference, simulator execution, rollout, training, GPU job, download, OpenVLA-OFT execution, token access, or paper claim.

Consequence: The summary records that the topology passed but task performance did not: diagnostic success count was 0, diagnostic success rate was 0.0, and reward sum was 0.0. The next safe rung is a bounded small learned-policy rollout matrix planner, still with diagnostic/local-pilot evidence labels only.

## Bounded Learned-Policy Rollout Matrix Planner Result

Decision: Reduce scope before any bounded small multi-task learned-policy rollout matrix.

Reason: `scripts\74_plan_bounded_learned_policy_rollout_matrix.ps1` found a passed topology summary but diagnostic success rate `0.0` and reward sum `0.0`.

Consequence: The next runnable stage should be a separately gated one-task longer diagnostic with at most 10 steps. The multi-task matrix remains blocked until there is a positive diagnostic success signal or a separate reduced-scope research decision.

## Bounded Reduced-Scope Learned-Policy Rollout Result

Decision: Treat the one-task, 10-step learned-policy LIBERO rollout as passed for execution readiness but not for task success.

Reason: `scripts\75_bounded_reduced_scope_learned_policy_rollout.ps1` completed 10 policy-controlled environment steps with local SmolVLA on CPU and no timeout, download, install, training, GPU job, OpenVLA-OFT execution, token access, or paper claim. The task success check remained `false` and reward sum remained `0.0`.

Consequence: The project should not scale directly to a multi-task matrix as performance evidence. The next safe step is a reduced-scope metric summary and then an action-interface/normalization diagnostic before larger rollout claims.

## Reduced-Scope Rollout Metric Summary Result

Decision: Treat the reduced-scope rollout metric summary as passed report-only diagnostic evidence.

Reason: `scripts\76_generate_reduced_scope_rollout_metric_summary.ps1` read the existing reduced-scope rollout JSON and summarized execution, success, reward, latency, action shape, action magnitude, and gripper component without any new model load, inference, simulator execution, rollout, training, GPU job, download, OpenVLA-OFT execution, token access, or paper claim.

Consequence: The summary confirms the selected task remains unsolved: diagnostic success rate `0.0` and reward sum `0.0`. It also records nontrivial action magnitude with gripper component `0.0`, motivating action-interface diagnostics before rollout scaling.

## WSL Bash CRLF Guard

Decision: Strip carriage returns from generated WSL `bash -lc` command strings before execution.

Reason: Running a PowerShell script with CRLF line endings can pass `$'\r'` tokens into WSL bash heredocs and break redirection with errors such as `ambiguous redirect`.

Consequence: Both `scripts\72_bounded_tiny_learned_policy_rollout.ps1` and `scripts\75_bounded_reduced_scope_learned_policy_rollout.ps1` remove `\r` from generated bash command strings before invoking WSL. This is a robustness fix only and does not change rollout scope or evidence labels.

## Action-Interface Diagnostic Planner Result

Decision: Prioritize action-interface diagnostics before scaling learned-policy rollouts.

Reason: `scripts\77_plan_action_interface_diagnostics.ps1` found that the policy action dimension is 6 while the LIBERO environment action dimension is 7, the gripper component is currently padded to `0.0`, action magnitude is nontrivial, and the task still has diagnostic success rate `0.0` and reward sum `0.0`.

Consequence: The next safe step is a metadata/report-only action-interface audit, followed by a bounded zero-action versus SmolVLA-action diagnostic comparison if the audit remains green. Larger rollout matrices remain premature.

## Action-Interface Metadata Audit Result

Decision: Treat the metadata audit as passed and block rollout scaling until interface diagnostics are addressed.

Reason: `scripts\78_audit_action_interface_metadata.ps1` found high-priority action/control risks. After adapter wiring, the remaining risks are 6D policy action versus 7D LIBERO action through an explicit adapter, zero-hold gripper strategy needing validation, and nontrivial policy actions with zero reward.

Consequence: The next safe steps are adapter-strategy, action-scale, prompt, camera, and state-sufficiency diagnostics. Larger rollout matrices are not useful until these interface risks are tested or mitigated.

## Zero-Action Versus Learned-Policy Comparison Scope

Decision: Make the next comparison summary-only by reading the existing zero-action and learned-policy diagnostic reports.

Reason: The current evidence question is whether learned-policy actions improve over already-validated simulator plumbing, not whether another rollout can be executed.

Consequence: If learned-policy actions are nontrivial but success/reward do not improve over zero-action, continue to an explicit action/state adapter patch plan before rollout scaling.

## Zero-Action Versus Learned-Policy Comparison Result

Decision: Do not scale learned-policy rollout yet; plan the action/state adapter patch first.

Reason: `scripts\79_compare_zero_action_policy_diagnostic.ps1` found that zero-action simulator plumbing passed and SmolVLA actions are nontrivial, but diagnostic success and reward did not improve over zero-action.

Consequence: The next safe task is an explicit action/state adapter patch plan covering 6D-to-7D action mapping, gripper semantics, state mapping, and camera key aliases.

## Action/State Adapter Patch Plan Scope

Decision: Plan the adapter patch before changing rollout behavior.

Reason: The current bridge has multiple interacting interface risks, and a direct rollout-code patch would be harder to validate safely.

Consequence: The next implementation should add pure adapter helpers and unit tests first, then wire them into single-sample/interface smoke before another bounded diagnostic rollout.

## Action/State Adapter Patch Plan Result

Decision: Proceed to pure adapter implementation, not rollout scaling.

Reason: `scripts\80_plan_action_state_adapter_patch.ps1` confirmed that action, state, and camera alias adapters are required and can be planned without executing models or simulators.

Consequence: Add pure adapter helpers and unit tests next. Keep rollout scaling blocked until adapter metadata appears in single-sample/interface reports.

## Pure Adapter Helper Scope

Decision: Add pure action, state, and image alias helpers before touching rollout execution.

Reason: Unit-tested pure helpers isolate interface assumptions without simulator/model side effects.

Consequence: The next wiring step must report adapter metadata in synthetic/single-sample checks before any bounded diagnostic rollout is rerun.

## Single-Sample Adapter Metadata Wiring Scope

Decision: Wire adapter helpers into synthetic single-sample smoke before simulator rollout code.

Reason: Synthetic single-sample smoke can validate report metadata, action adaptation, state mapping, and image alias selection without simulator side effects.

Consequence: Bounded rollout remains blocked until adapter metadata is visible in the synthetic/single-sample report.

## Single-Sample Adapter Metadata Result

Decision: Treat synthetic single-sample adapter metadata wiring as passed.

Reason: The bounded single-sample smoke recorded explicit state, image alias, and action adapter metadata while keeping simulator execution, rollouts, training, GPU training, OpenVLA-OFT execution, token access, and paper-grade claims false.

Consequence: The next safe step is to plan rollout-bridge adapter wiring separately before rerunning any bounded diagnostic rollout.

## Rollout Bridge Adapter Wiring Planner Scope

Decision: Add a planning-only gate before changing the learned-policy rollout bridge.

Reason: The rollout bridge currently still contains implicit action padding, state truncation, and local image alias fallback logic. A planner can prove the next code patch is scoped to wiring pure helpers, not executing another rollout.

Consequence: The next implementation may wire pure adapters into the rollout bridge only after this planner passes.

## Rollout Bridge Adapter Wiring Planner Result

Decision: Proceed to rollout bridge adapter wiring implementation, but not rollout execution.

Reason: `scripts\81_plan_rollout_bridge_adapter_wiring.ps1` confirmed that single-sample adapter metadata is present, pure helpers exist, and the rollout bridge still needs explicit action/state/image adapter wiring.

Consequence: The next safe implementation may modify rollout bridge code and tests only. A separately gated bounded diagnostic rollout is required before any execution.

## Pure Adapter Helper Result

Decision: Treat pure adapter helper implementation as passed.

Reason: Unit tests cover explicit 6D-to-7D action adaptation, named gripper strategies, refusal of unsupported action dimensions, explicit state fields without silent truncation/padding, and image alias reporting.

Consequence: The next safe step is wiring adapter metadata into synthetic or single-sample interface smoke without running rollout.

## Rollout Bridge Adapter Wiring Result

Decision: Treat rollout bridge adapter wiring as code-level validation only, not rollout evidence.

Reason: The learned-policy rollout bridge now imports and uses the pure action, state, and image adapter helpers. The bridge no longer contains the previous silent action padding helper, state truncation helper, or local image fallback selector, and task summaries include adapter metadata for later diagnostics.

Consequence: The next safe step is a separately gated bounded diagnostic rollout rerun that compares explicit-adapter behavior against the prior zero-action and legacy learned-policy diagnostics. Do not treat this wiring result as benchmark success or paper-grade evidence.

## Adapter-Wired Learned-Policy Diagnostic Result

Decision: Do not scale rollout yet; move to adapter-strategy and action-scale diagnosis.

Reason: The bounded one-task, 10-step learned-policy diagnostic passed execution with explicit adapter metadata recorded, but diagnostic success remained `0.0` and reward sum remained `0.0`. The zero-action comparison still shows no learned-policy improvement.

Consequence: Keep rollout scaling blocked. Next work should compare diagnostic adapter strategies, action scale/normalization, prompt format, and camera/state mapping before larger rollout matrices or paper claims.

## Adapter-Strategy/Action-Scale Diagnostics Planner Scope

Decision: Add a planning-only gate before implementing more rollout diagnostics.

Reason: The next experiment could run several bounded variants, so the repo needs an explicit envelope for gripper strategies, action-scale checks, prompt checks, runtime, and evidence labels before any execution.

Consequence: Implement `scripts\82_plan_adapter_strategy_action_scale_diagnostics.ps1` first. If it passes, the next safe implementation is a separately gated one-task gripper-strategy diagnostic runner.

## Adapter-Strategy Diagnostic Runner Result

Decision: Do not scale learned-policy rollout after the first gripper-strategy diagnostic.

Reason: `scripts\83_bounded_adapter_strategy_diagnostic.ps1` completed zero-hold, open, and close gripper-strategy variants with explicit adapter metadata, but all variants still had diagnostic success rate `0.0` and reward sum `0.0`.

Consequence: Gripper strategy is now execution-tested as an interface axis, but it did not produce a positive diagnostic signal. The next safe work should test action scale/normalization, prompt format, camera source, and state sufficiency before broader rollout matrices or paper-grade claims.

## Action-Scale Diagnostic Runner Result

Decision: Do not scale learned-policy rollout after the first action-scale diagnostic.

Reason: `scripts\85_bounded_action_scale_diagnostic.ps1` completed action scales `0.25`, `0.5`, and `1.0` with explicit action-scale metadata and expected action magnitude changes, but all variants still had diagnostic success rate `0.0` and reward sum `0.0`.

Consequence: Action scale is now execution-tested as an interface axis, but it did not produce a positive diagnostic signal. The next safe work should test prompt format, camera source, and state sufficiency before broader rollout matrices or paper-grade claims.

## Prompt-Format Diagnostic Runner Result

Decision: Do not scale learned-policy rollout after the first prompt-format diagnostic.

Reason: `scripts\87_bounded_prompt_format_diagnostic.ps1` completed `stem_spaces`, `bddl_language`, and `bddl_language_period` variants with explicit prompt metadata. The BDDL-language prompts changed the generated action previews but all variants still had diagnostic success rate `0.0` and reward sum `0.0`.

Consequence: Prompt format is now execution-tested as an interface axis, but it did not produce a positive diagnostic signal. The next safe work should test camera source and state sufficiency before broader rollout matrices or paper-grade claims.

## Camera-Source Diagnostic Runner Result

Decision: Do not scale learned-policy rollout after the first camera-source diagnostic.

Reason: `scripts\89_bounded_camera_source_diagnostic.ps1` completed `current_aliases`, `camera3_eye_in_hand`, and `all_agentview` variants with explicit camera source metadata. The variants changed image sources and action previews, but all variants still had diagnostic success rate `0.0` and reward sum `0.0`.

Consequence: Camera source selection is now execution-tested as an interface axis, but it did not produce a positive diagnostic signal. The next safe work should test state sufficiency before broader rollout matrices or paper-grade claims.

## State-Sufficiency Diagnostic Runner Result

Decision: Do not scale learned-policy rollout after the first state-sufficiency diagnostic.

Reason: `scripts\91_bounded_state_sufficiency_diagnostic.ps1` completed `eef_pos_quat_first3`, `eef_pos_quat_last3`, and `eef_pos_zero_rot` variants with explicit state adapter metadata. The variants changed state mappings and action previews, but all variants still had diagnostic success rate `0.0` and reward sum `0.0`.

Consequence: State-vector sufficiency is now execution-tested as an interface axis, but it did not produce a positive diagnostic signal. The next safe work is a learned-policy diagnostic synthesis/no-go report or a narrower environment-policy compatibility check, not broader rollout matrices or paper-grade claims.

## Learned-Policy Diagnostic Synthesis Result

Decision: Treat the current learned-policy rollout ladder as no-go for rollout scaling.

Reason: `scripts\92_generate_learned_policy_diagnostic_synthesis.ps1` synthesized zero-action comparison, adapter strategy, action scale, prompt format, camera source, and state sufficiency. All available bounded diagnostics passed wrapper/execution checks, but none produced nonzero reward or diagnostic success, and all keep `ready_for_rollout_scaling=false`.

Consequence: The next safe work is a bounded environment-policy compatibility audit focused on task/checkpoint alignment, action convention, and observation convention. Do not scale learned-policy rollouts or make paper-grade claims from the current evidence.

## Environment-Policy Compatibility Audit Result

Decision: Keep learned-policy rollout scaling blocked and move to offline demonstration interface auditing.

Reason: `scripts\93_audit_environment_policy_compatibility.ps1` found high-severity blockers in task/checkpoint alignment, `load_vlm_weights=false` diagnostic loading, 6D policy action versus 7D environment action convention, and repeated zero-reward diagnostic evidence.

Consequence: The next safe work is a bounded offline LIBERO HDF5 demonstration interface audit. It should inspect action dimensions/ranges, observation keys, camera shapes, and language/task alignment without loading models or running simulator rollout.

## LIBERO HDF5 Interface Audit Result

Decision: Keep rollout scaling blocked and move to offline adapter reproduction checks.

Reason: `scripts\94_audit_libero_hdf5_interface.ps1` confirmed that local LIBERO demonstrations use 7D actions while the SmolVLA policy config exposes 6D actions. It also found a camera-count/resolution preprocessing gap, while confirming that `obs/ee_states` is 6D and compatible with the policy state dimension.

Consequence: The next safe work is report-only adapter reproduction from the first HDF5 timestep. Do not load models or run simulator rollout for that step.

## Offline Adapter Reproduction Check Result

Decision: Treat gripper-close as the next bounded compatibility hypothesis, not as rollout-scaling evidence.

Reason: `scripts\95_check_offline_adapter_reproduction.ps1` reproduced the first local LIBERO demonstration action exactly with `policy_6d_delta_pose_plus_gripper_close`, while `policy_6d_delta_pose_plus_gripper_zero_hold` mismatched the first demonstration gripper value `-1.0`.

Consequence: A future one-task diagnostic may test gripper-close under the existing bounded diagnostic envelope. This does not unblock rollout scaling, multi-seed evaluation, paper-grade claims, or OpenVLA-OFT.

## Gripper-Close Compatibility Diagnostic Planner

Decision: Add a planning-only gate before any new gripper-close compatibility rollout.

Reason: Offline HDF5 evidence identifies gripper-close as the best first-action reproduction strategy, but previous bounded diagnostics may already have tested an equivalent close strategy without reward or success.

Consequence: `scripts\96_plan_gripper_close_compat_diagnostic.ps1` proceeds only when the close hypothesis is specific and not an already-failed duplicate. If a prior close diagnostic has zero reward and zero diagnostic success, the planner reduces scope toward HDF5-aligned task/initial-state/action-sign checks instead of repeating the same rollout.

## HDF5-to-Rollout Alignment Audit

Decision: Add a report-only audit before any HDF5-aligned replay or repeated learned-policy rollout.

Reason: Offline first-action reproduction evidence is meaningful only if the rollout diagnostic uses the same task and a compatible initial-state convention. The local HDF5 demonstrations include `init_state`/`states`, while the current rollout bridge resets the environment without evidence of setting the demonstration initial state.

Consequence: `scripts\97_audit_hdf5_rollout_alignment.ps1` reduces scope toward an HDF5 initial-state or first-action replay planner when task names match but initial-state alignment is not established. It keeps rollout scaling and paper claims blocked.

## HDF5 Initial-State Replay Planner

Decision: Add a planning-only gate for a one-demo HDF5 initial-state/first-action replay diagnostic.

Reason: LIBERO/RoboSuite source code exposes initial-state and flattened-state replay paths, and the selected HDF5 demonstration contains the necessary init-state/action data, but executing replay still needs a separate bounded simulator task.

Consequence: `scripts\98_plan_hdf5_initial_state_replay.ps1` can authorize only a separately gated replay runner. It does not authorize learned-policy inference, rollout scaling, training, OpenVLA-OFT, multi-seed evaluation, or paper-grade claims.

## Bounded HDF5 Initial-State Replay Runner

Decision: Add a separately gated one-demo, one-action replay runner.

Reason: The replay planner is green, and the next compatibility question is whether the simulator can be initialized from the local HDF5 demonstration state and step the first demonstration action without involving SmolVLA.

Consequence: `scripts\100_bounded_hdf5_initial_state_replay.ps1` may execute only under `ALLOW_HDF5_REPLAY_DIAGNOSTIC=1`. Passing it supports a later narrow learned-policy rollout recheck with a documented initial-state convention, but does not unblock rollout scaling or paper-grade claims.

## Init-State Learned-Policy Recheck Planner

Decision: Add a planning-only gate before learned-policy recheck from the validated HDF5 initial-state convention.

Reason: The HDF5 replay diagnostic proves the simulator can set the demonstration initial state and step the first demonstration action, but learned-policy inference must remain a separate bounded diagnostic task with explicit scope and evidence labels.

Consequence: `scripts\101_plan_init_state_learned_policy_recheck.ps1` can authorize only a future separately gated one-task recheck under `ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK=1`, capped at five policy-controlled steps. It does not authorize rollout scaling, multi-seed evaluation, GPU jobs, training, OpenVLA-OFT, or paper-grade claims.

## Bounded Init-State Learned-Policy Recheck Result

Decision: Treat the init-state learned-policy recheck as passed for execution topology but not for task performance.

Reason: `scripts\102_bounded_init_state_learned_policy_recheck.ps1` loaded local SmolVLA in WSL CPU, created one LIBERO/RoboSuite environment, set the HDF5 demonstration initial state, and executed 3 policy-controlled steps without downloads, installs, training, GPU jobs, OpenVLA-OFT, multi-seed evaluation, token access, or paper claims. The diagnostic success check remained false and reward sum remained `0.0`.

Consequence: Initial-state alignment alone does not explain the zero-reward behavior. The next safe work is a report-only metric comparison against previous reset-only learned-policy diagnostics before any further rollout decision.
