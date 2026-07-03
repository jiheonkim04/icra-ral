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

Consequence: Codex should not ask whether checkpoint files were placed or whether readiness/pytest/safe runner should be checked. Codex should inspect, report, update state/action docs if needed, and stop only at asset gates, checkpoint-file gates, validation failures, external installation/credential requirements, or dangerous gates requiring explicit approval.

## Official SmolVLA Checkpoint Source

Decision: Use `lerobot/smolvla_base` as the official SmolVLA checkpoint source for local acquisition.

Reason: The previous plan identified the required local layout and readiness checks but did not name a specific official source. The source ambiguity is now resolved.

Consequence: SmolVLA checkpoint acquisition may use `ALLOW_DOWNLOADS=1` only for `lerobot/smolvla_base` and only for files needed under `C:\assets\checkpoints\smolvla`. This does not authorize OpenVLA-OFT downloads, datasets, token access, model inference, heavy VLA imports, GPU jobs, training, or rollouts.

## SmolVLA Source Acquisition Result

Decision: Treat `lerobot/smolvla_base` acquisition as complete for the approved source.

Reason: The acquired source contains `config.json`, `model.safetensors`, processor JSON files, and processor safetensors. Its preprocessor references `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.

Consequence: The referenced tokenizer/processor dependency requires separate approval before acquisition.

## SmolVLA Tokenizer Dependency Acquisition Result

Decision: Acquire tokenizer/processor/config files from `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` and update readiness checks to recognize that external dependency under `HF_HOME`.

Reason: `policy_preprocessor.json` in `lerobot/smolvla_base` names this dependency for tokenization/model support, and the user explicitly approved tokenizer/processor dependency acquisition only.

Consequence: Full SmolVLM2 model weights remain forbidden for this step and were avoided. `ready_for_smolvla_adapter_smoke=true` now means file/interface readiness only. It does not authorize heavy imports, model loading, inference, GPU execution, training, rollouts, OpenVLA-OFT, or paper-grade claims.

## SmolVLA Load-Only Smoke Planning Guard

Decision: Add a planning-only SmolVLA load-only smoke guard before any heavy import/model-load task.

Reason: Readiness was true, but actual SmolVLA/SmolVLM loading crossed the heavy import/model-load gate before the later autonomous pilot standing approval.

Consequence: `scripts\15_plan_smolvla_load_only_smoke.ps1` may run safely without `ALLOW_HEAVY_IMPORT=1`. It writes a planning report, refuses to run when `ALLOW_HEAVY_IMPORT=1` is already set, and does not import SmolVLA, load models, run inference, train, rollout, download assets, access secrets, or execute OpenVLA-OFT.

## SmolVLA Load-Only Execution Scaffold

Decision: Add a bounded load-only smoke execution scaffold that stops before unsafe runtime behavior.

Reason: Readiness is true, but local runtime packages for SmolVLA loading were missing at scaffold time. Installing large packages or changing CUDA/PyTorch is a hard-stop gate.

Consequence: `scripts\16_smolvla_load_only_smoke.ps1` and `tca_map.smolvla.load_only_smoke` check gates, files, runtime dependencies, and memory policy. They report blockers without downloading assets, importing heavy VLA code, loading a model, running inference, training, rollouts, or OpenVLA-OFT.

## SmolVLA Runtime Dependency Boundary

Decision: Add a check-only runtime dependency script and a separate install plan.

Reason: Local SmolVLA files were ready, but `torch`, `transformers`, `lerobot`, and `safetensors` were not installed in the environment. Installing large packages or changing CUDA/PyTorch versions is a hard-stop gate.

Consequence: `scripts\17_check_smolvla_runtime_deps.ps1` reports package readiness without installing anything. Any actual install must be a separately approved environment task with pinned versions and rollback/validation steps.

## SmolVLA Runtime Install Approval Boundary

Decision: Add a planning-only runtime install request before any package installation or CUDA/PyTorch changes.

Reason: The environment is missing SmolVLA runtime packages, but installing PyTorch, LeRobot, Transformers, Safetensors, Accelerate, or Hugging Face Hub dependencies can destabilize the local Windows/CUDA setup.

Consequence: `scripts\18_plan_smolvla_runtime_install.ps1` may run safely as a check-only planner. It refuses dangerous gates and does not install packages, download assets, import heavy VLA models, load models, infer, train, rollout, access tokens, or execute OpenVLA-OFT.

## SmolVLA Runtime Install Execution Result

Decision: Complete the explicitly approved SmolVLA runtime package install in the local `tca_map` environment.

Reason: The user approved runtime package installation after the planner identified the missing SmolVLA runtime packages.

Consequence: The environment now has `torch==2.10.0+cu128`, `torchvision==0.25.0+cu128`, `transformers==4.57.6`, `lerobot==0.4.4`, `safetensors==0.8.0`, `accelerate==1.14.0`, and `huggingface-hub==0.35.3`. The later bounded load-only debug path also identified and installed `num2words==0.5.14` for the SmolVLM processor. This clears the runtime dependency gate only. It does not authorize inference, GPU execution, training, rollouts, simulator execution, OpenVLA-OFT, token access, package upgrades, CUDA toolkit changes, or paper-grade claims outside the standing-approved bounded pilot policy.

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

Reason: The configs are within the low-compute policy, but any actual head training still crosses an explicit training approval gate.

Consequence: `scripts\26_plan_tiny_head_only_pilot.ps1` checks the configs and reports that training is not safe to run yet. It does not download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, or execute OpenVLA-OFT.

## Consolidated Hard-Stop Status

Decision: Add a summary-only hard-stop approval status command.

Reason: The next meaningful steps are now gated by explicit approval, and those gates are spread across runtime install, heavy import/load-only smoke, and tiny training plans.

Consequence: `scripts\27_summarize_hard_stop_status.ps1` records the current approval choices without installing packages, downloading assets, running GPU jobs, importing heavy VLA models, loading models, inferring, training, rolling out, accessing tokens, executing simulators, executing OpenVLA-OFT, or making paper-grade claims.

## SmolVLA Autonomous Pilot Standing Approval

Decision: Treat the expected low-compute SmolVLA pilot path as standing-approved inside a bounded safety envelope.

Reason: The checkpoint/dependency files are ready, runtime dependencies are installed, safe runner and pytest pass, and repeated approval prompts for predictable load-only/interface/tiny-smoke steps prevent autonomous research-engineering progress.

Consequence: Codex should continue without asking before bounded SmolVLA load-only heavy import/model construction, load-only debugging, one synthetic or dummy single-sample interface smoke, tiny feature-cache/interface validation, and tiny head-only smoke with frozen backbone, max 100 steps, max 15 minutes, and max 14GB VRAM. Codex may set `ALLOW_HEAVY_IMPORT=1` only inside the bounded load-only task. Codex must still stop before OpenVLA-OFT download/import/load/execution, LIBERO/RoboSuite/RoboCasa/dataset downloads, simulator execution, rollouts, real benchmark evaluation, training outside the tiny-smoke budget, jobs over 30 minutes, more than 14GB VRAM, major CUDA/PyTorch changes, unplanned large package installs, token/secret/login requirements, multi-seed experiments, paper-level empirical claims, external submission/upload/publishing, or destructive deletion outside repository/approved cache cleanup.

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

Consequence: `scripts\33_plan_lora_tiny_smoke.ps1` validates the LoRA/QLoRA configs and future tiny-smoke envelope while keeping `safe_to_execute_lora_tiny_smoke_now=false`. It does not construct adapters, train, download assets, run GPU jobs, import heavy VLA models, load models, infer, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims.

## TCA-Map + LoRA Comparison Plan

Decision: Add a planning-only comparison matrix for the required LoRA arms.

Reason: The required LoRA track must not let LoRA gains obscure the TCA-Map or Distributional TCA-Select contribution.

Consequence: `scripts\34_plan_lora_comparison.ps1` fixes the ActionMap + LoRA, TCA-Map + LoRA, TCA-Map + LoRA + Distributional TCA-Select, and QLoRA-if-feasible comparisons without training, constructing adapters, downloading assets, running GPU jobs, importing heavy VLA models, loading models, inferring, rolling out, executing simulators, accessing tokens, executing OpenVLA-OFT, or making paper claims.
