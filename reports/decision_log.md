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

Consequence: LoRA/QLoRA remain optional support tools, not the core novelty.

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

Reason: Readiness is true, but actual SmolVLA/SmolVLM loading crosses the heavy import/model-load gate and needs a separate explicit approval step.

Consequence: `scripts\15_plan_smolvla_load_only_smoke.ps1` may run safely without `ALLOW_HEAVY_IMPORT=1`. It writes a planning report, refuses to run when `ALLOW_HEAVY_IMPORT=1` is already set, and does not import SmolVLA, load models, run inference, train, rollout, download assets, access secrets, or execute OpenVLA-OFT.

## SmolVLA Load-Only Execution Scaffold

Decision: Add a bounded load-only smoke execution scaffold that stops before unsafe runtime behavior.

Reason: Readiness is true, but local runtime packages for SmolVLA loading are currently missing. Installing large packages or changing CUDA/PyTorch is a hard-stop gate.

Consequence: `scripts\16_smolvla_load_only_smoke.ps1` and `tca_map.smolvla.load_only_smoke` check gates, files, runtime dependencies, and memory policy. They report blockers without downloading assets, importing heavy VLA code, loading a model, running inference, training, rollouts, or OpenVLA-OFT.

## SmolVLA Runtime Dependency Boundary

Decision: Add a check-only runtime dependency script and a separate install plan.

Reason: Local SmolVLA files are ready, but `torch`, `transformers`, `lerobot`, and `safetensors` are not installed in the current environment. Installing large packages or changing CUDA/PyTorch versions is a hard-stop gate.

Consequence: `scripts\17_check_smolvla_runtime_deps.ps1` reports package readiness without installing anything. Any actual install must be a separately approved environment task with pinned versions and rollback/validation steps.

## SmolVLA Runtime Install Approval Boundary

Decision: Add a planning-only runtime install request before any package installation or CUDA/PyTorch changes.

Reason: The environment is missing SmolVLA runtime packages, but installing PyTorch, LeRobot, Transformers, Safetensors, Accelerate, or Hugging Face Hub dependencies can destabilize the local Windows/CUDA setup.

Consequence: `scripts\18_plan_smolvla_runtime_install.ps1` may run safely as a check-only planner. It refuses dangerous gates and does not install packages, download assets, import heavy VLA models, load models, infer, train, rollout, access tokens, or execute OpenVLA-OFT.

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
