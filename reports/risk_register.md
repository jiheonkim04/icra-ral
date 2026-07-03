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

Mitigation: The dependency acquisition was separately approved for tokenizer/processor/config files only. The checker now recognizes the dependency under `HF_HOME`. Do not silently download full SmolVLM2 model weights, run heavy imports, or execute inference without separate explicit approval.

## Manual Confirmation Drift

Risk: Codex may waste turns asking the user to confirm routine state that can be checked automatically.

Impact: Slower progress, stale context, and avoidable human coordination overhead.

Mitigation: Follow the self-check gate policy in `reports/codex_delegation_manual.md`. Codex should run existing checkers, branch behavior on checker output, update state docs when useful, and ask only at dangerous gates.

## Windows / CUDA / PyTorch Compatibility

Risk: Local Windows environment may have CUDA or PyTorch compatibility issues, especially with newer RTX 5080 hardware.

Impact: Heavy imports or future GPU inference may fail even if file checks pass.

Mitigation: Keep current checks lightweight. Require explicit approval before heavy import or GPU inference. Prefer WSL2/Linux for simulator or heavier training work.

## RTX 5080 16GB VRAM

Risk: 16GB VRAM may be insufficient for larger VLA models, large heatmaps, full-resolution voxel heads, or non-quantized baselines.

Impact: OOM during future load-only smoke, feature caching, or pilots.

Mitigation: SmolVLA-first, frozen/head-only defaults, low-resolution heatmaps, batch size 1, memory estimates, and optional LoRA/QLoRA only with explicit config.

## Premature Heavy Import

Risk: A load-only smoke task may accidentally become a heavy import, model inference, or GPU execution task without explicit approval and bounded scope.

Impact: CUDA/Windows instability, OOM, hidden inference, or invalid claim that readiness is a result.

Mitigation: Use `scripts\15_plan_smolvla_load_only_smoke.ps1` first. It is planning-only, refuses `ALLOW_HEAVY_IMPORT=1`, and records the future gate. Actual loading requires a separate explicit approval task with no inference, training, rollout, dataset, simulator, or OpenVLA-OFT behavior.

## SmolVLA Runtime Dependency Drift

Risk: Local files are ready and runtime packages are installed now, but later package upgrades or CUDA/PyTorch changes could break the SmolVLA load-only path.

Impact: Load-only execution may fail, or CUDA behavior may change unexpectedly on Windows.

Mitigation: Keep package versions recorded in `reports\project_state.md` and re-run `scripts\17_check_smolvla_runtime_deps.ps1`. Any future package upgrade, CUDA toolkit change, or PyTorch change requires separate explicit approval.

## Unpinned Runtime Upgrade

Risk: Upgrading PyTorch, LeRobot, Transformers, or Safetensors without pinned versions can break the current environment or mismatch the RTX 5080/CUDA stack.

Impact: Failed model load, CUDA errors, dependency conflicts, or a hard-to-reproduce local setup.

Mitigation: Use `reports\smolvla_runtime_dependency_plan.md` and `scripts\17_check_smolvla_runtime_deps.ps1`. Require explicit approval before changing packages, capture environment state before and after, and validate with the safe runner.

## Accidental Runtime Install During Planning

Risk: A planning task for SmolVLA runtime packages may accidentally become a package installation or CUDA/PyTorch change.

Impact: Broken local environment, CUDA mismatch, unexpected downloads, or blocked load-only validation.

Mitigation: Use `scripts\18_plan_smolvla_runtime_install.ps1` and `reports\smolvla_runtime_install_request.md`. The planner refuses dangerous gates and records that installs, downloads, heavy imports, model loads, inference, training, rollouts, and OpenVLA-OFT execution were not performed.

## Runtime-Ready Misread As Load-Ready

Risk: Runtime dependency readiness may be mistaken for permission to import/load SmolVLA.

Impact: A routine check could accidentally cross the heavy import/model-load gate.

Mitigation: Keep `ALLOW_HEAVY_IMPORT=1` as a separate explicit gate. Runtime package installation only clears dependency availability; it does not authorize heavy imports, model loading, inference, GPU execution, training, rollouts, OpenVLA-OFT, or paper-grade claims.

## Feature Cache Contract Drift

Risk: Later SmolVLA feature extraction may produce records that do not match the head-only training interface.

Impact: Head-only pilots fail late or silently train on inconsistent metadata.

Mitigation: Use `reports\feature_cache_interface_plan.md`, `tca_map.features.cache`, and `scripts\19_plan_feature_cache.ps1` to validate manifest and JSONL schema with dummy features before any real extraction.

## Cached-Feature Consumer Drift

Risk: Cached features may be valid on disk but unusable by TCA-Map heads or offline metric code.

Impact: The first head-only pilot fails after expensive feature extraction or produces invalid proxy metrics.

Mitigation: Run `scripts\25_eval_feature_cache_smoke.ps1 -PrepareDummyCache` and `tests\test_feature_cache_eval_smoke.py` to validate the consumer path with dummy cached features before real SmolVLA extraction.

## Accidental Tiny Training

Risk: A planning step for the tiny head-only pilot may accidentally launch training.

Impact: Unapproved GPU use, invalid local results, or policy drift beyond the bounded autopilot session.

Mitigation: Use `scripts\26_plan_tiny_head_only_pilot.ps1`, which refuses training/heavy gates and reports `safe_to_run_training_now=false`. Actual training requires explicit user approval.

## Approval Scope Confusion

Risk: Runtime install, heavy import/load-only smoke, and tiny training approvals may be mixed together accidentally.

Impact: A small approval could unintentionally permit package changes, model loading, GPU use, or training in the same task.

Mitigation: Use `scripts\27_summarize_hard_stop_status.ps1` and `reports\hard_stop_status.md`. Approve at most one gate at a time unless a later task explicitly scopes and justifies a combined approval.

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
