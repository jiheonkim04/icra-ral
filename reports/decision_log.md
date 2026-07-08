# Decision Log

## Literature-First Topic Selection

Decision: stop implementation-first topic testing and switch to memo-only literature-first selection.

Reason: recent routes repeatedly failed after local method scaffolds met stronger simple baselines. The new process must select topics from recent-paper gaps, baseline-kill risk, and direct evidence contracts before any code.

Consequence: do not continue ActionMap implementation, diagnostics, reproduction, extension work, or failure mining. The ActionMap mini-gate commit had already been fast-forward merged into local `main` before the stop steer arrived, but it has not been pushed. The current shortlist is `Constraint-Validated Spline VLA Action Interface` and `Early Failure Detection With Evidence-Calibrated Stop/Retry`, with the spline interface recommended only for a pre-implementation evidence contract.

Execution boundary: memo-only. No experiment, training, rollout, download, GPU job, OpenVLA-OFT, new diagnostic, or new method code occurred for this decision.

## ActionMap Anchor State 1 Result

Decision: kill or reframe the local ActionMap anchor reproduction before failure mining or extension work.

Reason: the bounded diagnostic produced real HDF5-backed action metrics over `8` local LIBERO demos and `1008 / 432` deterministic train/eval records, but the ActionMap-style heatmap/candidate head failed the simple-baseline gate. ActionMap-style action L2 was `0.529931357`, worse than mean-action (`0.466767673`) and matched/beat by cheap MLP (`0.501926707`). It beat the linear/L1 baseline (`0.812610317`) and had oracle candidate headroom (`0.065653208`), but the learned candidate selection collapsed to one rotation bin (`5 / 1 / 2` unique translation/rotation/gripper bins).

Consequence: do not proceed to STATE 2 failure mining and do not propose an ActionMap extension from this result. A reframe must first reproduce an anchor-style head that beats mean-action, linear/L1, and cheap MLP baselines without candidate collapse.

Execution boundary: tiny CPU NumPy training happened and loss was computed. Replay/control, GPU jobs, downloads, heavy VLA imports/model loading, full VLA fine-tuning, OpenVLA-OFT execution, simulator rollout, token access, and paper-grade claims did not occur.

## ActionMap Anchor Reproduction Start

Decision: start a reproduction-first ActionMap anchor route after ContactSet-VLA was archived and pushed.

Reason: recent method-first routes repeatedly failed against simple baselines. The next route must first approximate a strong recent anchor, verify it beats mean and linear/simple heads locally, then mine failures before proposing any extension.

Consequence: no new method or extension is allowed until the ActionMap-style diagnostic produces a real HDF5-backed metric and passes the simple-baseline gate.

Execution boundary: planning and scaffold only at this entry. No rollout, GPU job, download, heavy VLA import, full VLA fine-tuning, OpenVLA-OFT execution, or paper claim.

## ContactSet-VLA State 1 Result

Decision: kill ContactSet-VLA as the current main route before full VLA fine-tuning or replay scale-up.

Reason: the bounded local HDF5 action-head diagnostic computed real held-out 7D action losses over `6` local LIBERO demos and found that full contact-set injection did not beat active single-3D-point injection. Full contact-set action L2 was `1.105028754`, while active single-point action L2 was `0.930495702`. No-geometry action L2 was `0.851451`, and destination-only action L2 was `0.86372`, so simple baselines matched or beat the proposed contact set.

Consequence: do not run ContactSet-VLA full VLA fine-tuning, OpenVLA-OFT, GPU training, or a replay/progress milestone from this evidence. Preserve the geometry extraction, qpos-offset audit, set encoder, runner, and tests as reusable infrastructure only.

Execution boundary: tiny CPU NumPy action-head training happened and loss was computed. Simulator rollout/replay, GPU jobs, downloads, heavy VLA imports/model loading, OpenVLA-OFT execution, token access, and paper-grade claims did not occur.

## ContactSet-VLA Archive

Decision: archive ContactSet-VLA as a hard-killed route and preserve only its diagnostic infrastructure.

Reason: the decisive STATE 1 result already triggered the route-level kill gate. Full contact-set injection lost to active single-point, destination-only, and no-geometry action-head baselines on held-out local LIBERO action L2. Continuing to replay or VLA fine-tuning would violate the baseline-first rule.

Consequence: future action-head geometry topics must beat active single-point, source-only, destination-only, source+destination, and no-geometry baselines before scale-up. ContactSet-VLA artifacts remain useful as a geometry/leakage audit and point-encoding diagnostic harness.

Execution boundary: documentation-only archive. No new experiment, replay, rollout, training, loss computation, GPU job, download, heavy VLA import/model loading, OpenVLA-OFT execution, benchmark rollout, or paper-grade claim occurred.

## PRISM-VLA Archive

Decision: archive PRISM-VLA as a killed main RA-L route and add canonicalization-only to the global simple-baseline screen for future language robustness topics.

Reason: State 2 already produced the decisive anti-baseline result. Canonicalization-only beat the best PRISM variant on held-out paraphrase proxy (`0.474066` versus `0.436356`) and PRIDE (`46.686731` versus `31.985592`), best PRISM primary held-out delta versus canonicalization was `-0.030420`, and counterfactual sensitivity was not preserved. Building a real VLA diagnostic before passing this primary canonicalization gate would violate the repository baseline-first rule.

Consequence: do not continue PRISM-VLA as an RA-L-stable route. Preserve the LIBERO-Para metadata integration, held-out paraphrase split, diagnostic runner, PRIDE/consistency metrics, and counterfactual sensitivity checks as reusable artifacts. Future language robustness routes must beat canonicalization-only on held-out robustness and preserve object/target sensitivity before scale-up.

Execution boundary: documentation-only archive. No new experiment, training, rollout, loss computation, GPU job, download, heavy VLA import, OpenVLA-OFT execution, or paper-grade claim occurred.

## PRISM-VLA State 2 Result

Decision: kill PRISM-VLA as the current main route under the held-out paraphrase/canonicalization dominance gate.

Reason: the base model still showed measurable held-out paraphrase degradation (`0.062428` clean-to-held-out drop), and PRISM+canonicalization beat simple paraphrase augmentation on primary held-out robustness (`+0.055205`). However, canonicalization-only beat every PRISM variant on the primary held-out paraphrase/PRIDE metrics. Canonicalization-only held-out paraphrase proxy was `0.474066` and PRIDE was `46.686731`; the best PRISM variant, `prism_vla_plus_canonicalization`, reached held-out paraphrase proxy `0.436356`, PRIDE `31.985592`, and primary held-out delta versus canonicalization `-0.030420`.

Consequence: do not scale PRISM-VLA, do not run OpenVLA-OFT, and do not claim paper-grade or RA-L-ready evidence. A later real SmolVLA paraphrase feature/adapter diagnostic may be considered only as a separate risk-assessed milestone that directly compares canonicalization-only, PRISM, and PRISM+canonicalization without full fine-tuning, rollout, GPU, downloads, or paper claims.

Execution boundary: tiny CPU training happened and loss was computed. No real VLA diagnostic, GPU job, simulator rollout, heavy VLA import, download, OpenVLA-OFT execution, token access, or paper-grade claim occurred.

## PRISM-VLA State 1 Result

Decision: continue PRISM-VLA only to the next bounded diagnostic, not to heavy training or paper claims.

Reason: the CPU diagnostic produced measurable base paraphrase degradation (`0.080341` clean-to-paraphrase proxy drop) and PRISM beat simple paraphrase augmentation on at least one robustness metric (`+1.398131` PRIDE delta, `+0.000743` consistency-score delta, `+0.000319` paraphrase proxy delta). Clean retention stayed above the predeclared threshold (`0.877609` vs base clean), and counterfactual sensitivity was preserved under the proxy gate.

Consequence: the next milestone should test held-out paraphrases or a real local VLA adapter under a separate risk assessment. The current result is exploratory offline proxy evidence from a tiny NumPy surrogate over local LIBERO action chunks and official LIBERO-Para metadata. It is not standard success, rollout success, a real VLA checkpoint result, or paper-grade evidence.

Execution boundary: tiny CPU training happened and loss was computed. GPU jobs, simulator rollouts, heavy VLA imports, OpenVLA-OFT, token access, external upload, and paper-grade claims did not occur.

## Research Integrity Before Comparisons

Decision: Fix primary metrics, baselines, ablations, split/sample policy,
tuning budget, and kill/pivot criteria before any confirmatory ActionMap vs
TCA-Map, TCA-Select, LoRA, or QLoRA evaluation.

Reason: The project must test whether TCA-Map is actually valuable rather than
optimize experiments to make it look good.

Consequence: Codex must not cherry-pick tasks, samples, seeds, metrics,
baselines, visualizations, or rollout episodes. Failed runs and weak results
must be logged. Exploratory debugging must remain separate from confirmatory
evaluation. If ActionMap + LoRA or ActionMap + counterfactual augmentation
matches TCA-Map, the novelty is weak. If TCA-Select adds no measurable gain or
offline gains disappear in rollout, the report must say so. If TCA-Map fails,
produce a kill/pivot report.

## Bounded Autopilot Execution

Decision: Replace unbounded end-to-end research autopilot loops with bounded per-execution milestones.

Reason: Long autonomous chains can drift into planner expansion, large diffs, or multiple research milestones without a clear merge checkpoint.

Consequence: Each execution may complete at most one major research milestone, such as real candidate-generation smoke, research-integrity policy update, ActionMap vs TCA-Map tiny training/eval, LoRA tiny training/eval, rollout diagnostic, or paper-grade roadmap update. Codex must stop before commit if more than 50 files or more than 5,000 changed lines would be included, and must report changed-file count, line diff count, training/rollout/loss/scaffolding status, validation results, and merge justification before every merge.

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

## Init-State Recheck Metric Summary Result

Decision: Keep rollout scaling blocked.

Reason: `scripts\103_generate_init_state_recheck_metric_summary.ps1` compared reset-only 3-step, reset-only 10-step, and HDF5-init-state 3-step learned-policy diagnostics. All wrappers passed, but every scenario had diagnostic success `false` and reward sum `0.0`.

Consequence: The project should stop adding rollout variants for now and inspect checkpoint/task alignment, VLM loading policy, and offline demonstration-conditioned action decoding before another learned-policy rollout.

## SmolVLA/LIBERO Checkpoint-Task Alignment Audit

Decision: Add a report-only checkpoint/task alignment audit before any further learned-policy rollout.

Reason: The current diagnostics can execute reset-only and HDF5-init-state rollouts, but all reward and success signals remain zero. The next uncertainty is whether the local SmolVLA checkpoint, VLM loading policy, LIBERO task language, observation/action conventions, and demonstration-conditioned decoding are aligned.

Consequence: `scripts\104_audit_smolvla_libero_checkpoint_task_alignment.ps1` reads only local config/preprocessor files, local BDDL names, and existing reports. It keeps rollout scaling blocked and routes the next safe work toward a planning-only offline demonstration-conditioned action decoding gate.

## Offline Demonstration-Conditioned Action Decoding Planner

Decision: Add a planning-only gate for one-sample offline action decoding before any further learned-policy rollout.

Reason: The checkpoint-task alignment audit found that a non-rollout offline action-decoding check is the next informative step. It can test whether the local policy action decoder is at least consistent with a real LIBERO demonstration observation and expert action without simulator rollout.

Consequence: `scripts\105_plan_offline_demo_conditioned_action_decoding.ps1` checks local file/report prerequisites and writes a risk assessment for a future one-sample runner. It does not load models, infer, rollout, train, download, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

## Bounded Offline Demonstration Action Decoding Runner

Decision: Add a separately gated one-sample offline action-decoding runner before any additional learned-policy rollout.

Reason: The planner authorized a non-rollout check that can directly compare a local SmolVLA decoded action to a real LIBERO expert action from the same HDF5 observation.

Consequence: `scripts\106_bounded_offline_demo_action_decoding.ps1` requires `ALLOW_OFFLINE_DEMO_ACTION_DECODING=1`, loads SmolVLA on CPU, reads one HDF5 timestep, runs one `select_action` call, and writes diagnostic action error metrics. It does not create simulator environments, rollout, train, download, use GPU jobs, execute OpenVLA-OFT, or make paper claims.

## Offline Demonstration Action Decoding Summary

Decision: Summarize the one-sample offline action-decoding diagnostic before any further rollout decision.

Reason: The one-sample diagnostic can complete successfully while still producing a large action error to the expert action. That distinction should be explicit before adding rollout variants.

Consequence: `scripts\107_summarize_offline_demo_action_decoding.ps1` keeps rollout scaling blocked when offline alignment is weak and routes the next work toward VLM loading policy, checkpoint provenance, or action normalization analysis.

## VLM Loading Policy and Action-Normalization Audit

Decision: Add a report-only audit for VLM loading policy, action normalization, action clipping, and camera/action adapter conventions.

Reason: The offline action-decoding diagnostic produced finite actions but weak expert alignment. The checkpoint config requests `load_vlm_weights=true`, while the bounded local diagnostic used `load_vlm_weights=false`, and the 6D-to-7D adapter clipped at least one action component.

Consequence: `scripts\108_plan_vlm_loading_policy_action_normalization_audit.ps1` keeps rollout scaling blocked and routes the next safe work toward a tiny repeated offline HDF5 action-decoding diagnostic. It does not load models, run inference, rollout, train, download, use GPU jobs, execute OpenVLA-OFT, or make paper claims.

## Repeated Offline Demonstration Action-Decoding Plan

Decision: Add a planning-only gate for a tiny repeated offline action-decoding diagnostic.

Reason: One-sample weak alignment is informative but too brittle. The next non-rollout check should determine whether weak expert-action alignment persists across a few HDF5 timesteps while explicitly logging VLM load policy, action unnormalization, clipping, gripper strategy, and image aliases.

Consequence: `scripts\109_plan_repeated_offline_demo_action_decoding.ps1` inspects HDF5 metadata and writes a bounded future-runner risk assessment. It does not load models, infer, rollout, train, download, use GPU jobs, execute OpenVLA-OFT, or make paper claims.

## Bounded Repeated Offline Demonstration Action Decoding

Decision: Add a separately gated repeated offline decoding runner capped at three local HDF5 timesteps.

Reason: The planner found enough local HDF5 timesteps and the previous VLM/action audit authorized a repeated offline diagnostic. The runner can test whether weak action alignment persists without creating a simulator environment or rollout.

Consequence: `scripts\110_bounded_repeated_offline_demo_action_decoding.ps1` may load local SmolVLA on CPU under `ALLOW_REPEATED_OFFLINE_DEMO_DECODING=1` and run at most three `select_action` calls. It records diagnostic metrics only and keeps rollout scaling and paper claims blocked.

Current result: The bounded runner passed on three HDF5 timesteps, but alignment stayed weak (`mean_action_l1_to_expert=0.412322`, `mean_action_mse_to_expert=0.286972`, clipped values total `3`, `load_vlm_weights=false`). Rollout scaling remains blocked. The next decision should focus on VLM-enabled loading risk/provenance and action normalization, not another rollout variant.

## VLM-Enabled Loading Risk Plan

Decision: Add a metadata-only VLM-enabled loading risk/provenance planner before any full SmolVLM2 weight acquisition or VLM-enabled load.

Reason: Repeated offline alignment stayed weak while local diagnostics used `load_vlm_weights=false`. The next question is whether enabling the VLM path is official, token-free, size-bounded, disk-safe, and memory-safe.

Consequence: `scripts\111_plan_vlm_enabled_loading_risk.ps1` queries or reads Hugging Face metadata only. It does not download weights, load models, infer, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

Current result: The planner is green. The official repo is public/ungated with `apache-2.0`, required files are about `1.895GB`, and disk budget remains green. This authorizes only a future separately gated acquisition plan/runner, not immediate model loading.

## VLM Required Files Acquisition Result

Decision: Acquire the bounded required files from `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.

Reason: The metadata-only risk planner verified an official public ungated source, apache-2.0 license, known bounded size around `1.895GB`, no token/login/payment/license gate, and enough disk margin.

Consequence: The local dependency directory now contains root `model.safetensors` plus the config/tokenizer/processor files needed for a later VLM-enabled load-smoke plan. This does not authorize model loading, inference, rollout scaling, training, GPU jobs, OpenVLA-OFT, package changes, token access, or paper-grade claims. The next step must be a separate bounded VLM-enabled load-smoke planner.

## VLM-Enabled Load Smoke Planner

Decision: Add a planning-only gate before any `load_vlm_weights=true` SmolVLA construction.

Reason: The full VLM dependency files are now local, but actual VLM-enabled policy construction is a heavy model-load task with RAM/runtime risk and must be separated from acquisition.

Consequence: `scripts\113_plan_vlm_enabled_load_smoke.ps1` can authorize only a future separately gated load-only runner. It does not load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, access tokens, install packages, download files, or make paper-grade claims.

Current result: The planner reports `decision=proceed` and `ready_for_bounded_vlm_enabled_load_smoke_runner=true`. This authorizes implementation of the separate runner, not model-load execution inside the planner.

## Bounded VLM-Enabled Load Smoke Runner

Decision: Add a separately gated CPU-first runner for `load_vlm_weights=true` construction.

Reason: The planner is green and the full VLM dependency files are local, so the next informative compatibility test is whether SmolVLA construction succeeds with VLM weights enabled.

Consequence: `scripts\114_bounded_vlm_enabled_load_smoke.ps1` requires both `ALLOW_HEAVY_IMPORT=1` and `ALLOW_VLM_ENABLED_LOAD_SMOKE=1`. It is load-only and must not infer, train, rollout, use GPU jobs, download, install, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

Current result: The bounded runner passed on CPU with `load_vlm_weights=true`, CUDA max allocation `0MB`, and no inference/training/rollout/download/install/OpenVLA-OFT/token/paper-claim behavior. The next decision should test whether VLM-enabled loading improves repeated offline action-decoding alignment before any rollout scaling.

## VLM-Enabled Repeated Offline Decoding Planner

Decision: Add a planning-only gate for repeated offline action decoding with VLM weights enabled.

Reason: The prior repeated offline diagnostic was weak with `load_vlm_weights=false`, while VLM-enabled load-only construction now passes. A tiny offline recheck is the next informative step before any rollout scaling.

Consequence: `scripts\115_plan_vlm_enabled_repeated_offline_decoding.ps1` authorizes only a future separately gated CPU offline diagnostic capped at three local HDF5 timesteps. It does not authorize rollout scaling, simulator execution, training, GPU jobs, OpenVLA-OFT, or paper claims.

Current result: The planner reports `decision=proceed` and selects timesteps `0`, `136`, and `271` for a future VLM-enabled offline recheck against the previous weak no-VLM baseline.

## Bounded VLM-Enabled Repeated Offline Decoding Runner

Decision: Add a separately gated offline runner for the VLM-enabled action-decoding recheck.

Reason: The planner is green and the next uncertainty is whether enabling VLM weights changes expert-action alignment on the same local HDF5 timesteps.

Consequence: `scripts\116_bounded_vlm_enabled_repeated_offline_decoding.ps1` may run only under its task-local gates and remains offline diagnostic evidence. It does not authorize simulator rollout, training, GPU jobs, OpenVLA-OFT, or paper claims.

Current result: The bounded runner passed on CPU with `load_vlm_weights=true` and decoded three local HDF5 timesteps. Mean action L1/MSE improved versus the previous no-VLM repeated diagnostic (`0.301665` / `0.216188` versus `0.412322` / `0.286972`), but the offline alignment signal remains `weak` and clipped values remain present.

Consequence: VLM-enabled loading is behaviorally relevant, but still not enough to justify rollout scaling or paper claims. The next decision should summarize the VLM-on/off delta and inspect action normalization/provenance before any further learned-policy rollout.

## VLM-Enabled Offline Decoding Summary

Decision: Add a report-only VLM-on/off summary before any rollout decision.

Reason: The VLM-enabled repeated offline diagnostic improved action L1/MSE but kept the alignment signal weak. The project needs a compact comparison and explicit normalization/provenance blocker list before choosing another learned-policy rollout hypothesis.

Consequence: `scripts\117_summarize_vlm_enabled_offline_decoding.ps1` reads existing runtime reports and local config JSON only. It does not load models, infer, train, rollout, download, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims. If weak alignment and clipping remain, rollout scaling stays blocked.

Current result: The summary passed. VLM-enabled loading reduced mean action L1/MSE by `26.838%` / `24.666%`, but the alignment signal remained `weak`, clipped values persisted, ACTION `MEAN_STD` normalization is active, and the 6D policy action shape still requires provenance analysis against the 7D LIBERO action convention.

Consequence: Do not scale learned-policy rollout yet. The next decision should come from a report-only action-normalization/provenance audit.

## Action Normalization Provenance Audit

Decision: Add a report-only audit for action normalization statistics and action-convention provenance.

Reason: VLM-enabled loading improved offline action-distance metrics but did not clear weak alignment or clipping. Processor action stats, 6D policy action shape, 7D LIBERO action convention, and adapter clipping must be audited before another rollout hypothesis.

Consequence: `scripts\118_audit_action_normalization_provenance.ps1` reads local config/processor JSON, processor safetensors, and offline reports only. It does not load models, infer, train, rollout, download, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

Current result: The audit passed with `decision=no_go_rollout_scaling`. Processor action stats are keyed by `so100`, `so100-blue`, and `so100-red`; action mean/std magnitudes are much larger than local LIBERO expert-action previews; policy action shape is `[6]`; the adapter path remains 7D; decoded actions still clip.

Consequence: The next decision should be a planning-only action-stat mapping or checkpoint/task-provenance correction plan. Do not run another learned-policy rollout variant until this mismatch is addressed or explicitly ruled out.

## Action-Stat Provenance Correction Plan

Decision: Add a planning-only correction gate before changing action behavior or running another rollout.

Reason: The audit found strong action-stat provenance mismatch risk. The next safest informative step is to compute LIBERO action statistics from local HDF5 files and compare them directly to checkpoint processor stats.

Consequence: `scripts\119_plan_action_stat_provenance_correction.ps1` selects a report-only LIBERO action-stat subset audit as the next step. It does not authorize model changes, training, rollout, checkpoint downloads, GPU jobs, OpenVLA-OFT, or paper claims.

Current result: The plan passed with `decision=reduce_scope` and selected `libero_action_stat_subset_audit`.

Consequence: Implement a bounded HDF5 action-stat audit before any normalized-action probe, postprocessor bypass/replacement, checkpoint download, or learned-policy rollout.

## LIBERO Action-Stat Subset Audit

Decision: Add a report-only bounded HDF5 action-stat audit.

Reason: The correction plan selected direct measurement of local LIBERO action statistics as the safest next step before changing policy behavior or running rollouts.

Consequence: `scripts\120_audit_libero_action_stats.ps1` reads bounded local HDF5 action arrays and compares their scale/dimension to checkpoint action stats. It does not load models, infer, train, rollout, download, use GPU jobs, execute OpenVLA-OFT, alter policy behavior, or make paper claims.

Current result: The audit passed with `decision=no_go_rollout_scaling`. It sampled `2500` local LIBERO actions from `5` files and confirmed 7D unit-scale LIBERO actions versus 6D SO100 large-scale checkpoint action statistics.

Consequence: Do not treat current learned-policy rollout failures as paper-relevant policy performance. The next decision should plan a normalized-action-space probe or checkpoint/task provenance resolution.

## Normalized Action-Space Probe Plan

Decision: Add a planning-only gate for normalized-action-space probing versus checkpoint/task provenance resolution.

Reason: The LIBERO action-stat subset audit confirmed both scale and dimension mismatch, and the checkpoint processor stats are SO100-prefixed. That makes a checkpoint/task provenance audit safer and more informative than immediately changing postprocessing or running another learned-policy rollout.

Consequence: `scripts\121_plan_normalized_action_space_probe.ps1` selects `checkpoint_task_provenance_resolution` as the next safe step when SO100-prefixed stats, 7D unit-scale LIBERO actions, scale mismatch, and dimension mismatch are all present. It keeps normalized-action probing deferred to a later separately gated runner and does not authorize model loading, inference, training, rollout, downloads, GPU jobs, OpenVLA-OFT, policy behavior changes, or paper claims.

## Checkpoint / Task Provenance Resolution

Decision: Add a report-only provenance resolver for the current SmolVLA checkpoint versus local LIBERO action conventions.

Reason: The normalized-action plan selected provenance resolution before any action-space probe. Local checkpoint metadata and model-card text must be checked against LIBERO HDF5 action stats so that learned-policy rollout failures are not overinterpreted.

Consequence: `scripts\122_resolve_checkpoint_task_provenance.ps1` reads only local checkpoint metadata and existing runtime reports. It blocks learned-policy rollout scaling if the checkpoint remains 6D/SO100-like while local LIBERO demonstrations are 7D/unit-scale, and routes the next research path to offline/head TCA-Map plus required LoRA evidence or to a separate LIBERO-aligned checkpoint source plan.

## Offline TCA-Map / LoRA Pivot Plan

Decision: Add a report-only pivot plan after checkpoint/task provenance blocks learned-policy LIBERO rollout scaling.

Reason: The current base checkpoint is not valid LIBERO learned-policy rollout evidence, but the real-LIBERO offline head and LoRA proxy reports are available. The project needs a clean path that keeps paper work moving without overclaiming rollout evidence.

Consequence: `scripts\123_plan_offline_tca_map_lora_pivot.ps1` selects an offline evidence table and gap report when provenance no-go and offline reports are present. It keeps learned-policy rollout scaling, standard success, benchmark success, paper claims, OpenVLA-OFT, downloads, GPU jobs, and heavy imports blocked.

## Offline Evidence Gap Report

Decision: Add a report-only evidence table/gap report for the real-LIBERO offline proxy ladder.

Reason: The pivot plan selected offline evidence consolidation. The project needs one compact table for ActionMap, TCA-Map, Distributional TCA-Select, required LoRA arms, and remaining blockers before planning any scale-up.

Consequence: `scripts\124_generate_offline_evidence_gap_report.ps1` consolidates existing offline proxy reports and records that standard success, learned-policy rollout scaling, benchmark claims, and paper claims remain blocked. It does not run training, rollout, model loading, inference, GPU jobs, downloads, OpenVLA-OFT, or heavy imports.

## Bounded LoRA / Offline Proxy Scale-Up Plan

Decision: Add a planning-only gate for the next required LoRA/offline-proxy scale-up.

Reason: The evidence gap report says the offline evidence table is ready and LoRA scale-up planning is safe, while current-checkpoint learned-policy rollout scaling remains blocked.

Consequence: `scripts\125_plan_bounded_lora_offline_scaleup.ps1` authorizes only a future separately gated CPU-only offline LoRA runner with at most 16 pairs, 64 samples, 64 steps, LoRA rank 4, frozen base weights, no full fine-tuning, no rollout, no heavy imports, no model load, no GPU job, no OpenVLA-OFT, and no paper claim.

## Bounded LIBERO Offline LoRA Scale-Up Runner

Decision: Add the separately gated CPU-only offline LoRA scale-up runner selected by the planning gate.

Reason: The evidence gap report and scale-up plan show that the safest next informative step is a bounded required-LoRA track over local LIBERO HDF5 snippets, not learned-policy rollout scaling with the current checkpoint.

Consequence: `scripts\126_bounded_lora_offline_scaleup.ps1` may run only under task-local `ALLOW_TINY_TRAINING=1`, trains tiny NumPy LoRA adapter matrices only, and keeps SmolVLA loading, heavy imports, GPU jobs, rollouts, simulator execution, downloads, OpenVLA-OFT, full fine-tuning, token access, and paper claims blocked.

Current result: The bounded scale-up runner passed over 16 local LIBERO offline records with 64 update steps. It improved TCA-Map + LoRA over ActionMap + LoRA on the offline proxy wrong-target rate and action L1 deltas, while keeping paper and rollout readiness false.

## Scale-Up-Aware Offline Evidence Gap Refresh

Decision: Extend the offline evidence gap generator to include the bounded LoRA scale-up report when present.

Reason: The project now has a bounded required-LoRA proxy result that should be visible next to the earlier tiny LoRA comparison without being mistaken for rollout or paper-grade evidence.

Consequence: `scripts\124_generate_offline_evidence_gap_report.ps1` remains report-only, but now records whether bounded LoRA scale-up was included and adds separate bounded offline LoRA proxy rows. It still blocks learned-policy rollout scaling and paper claims.

Current result: The refreshed evidence gap report included the bounded LoRA scale-up rows and kept rollout, benchmark, and paper-claim readiness false. The next decision point should synthesize attribution gaps rather than claim benchmark progress.

## Scale-Up Attribution Gap Synthesis

Decision: Add a report-only synthesis of bounded LoRA scale-up attribution gaps.

Reason: The scale-up-aware evidence table shows useful TCA-Map + LoRA proxy gains, but Distributional TCA-Select has no additional LoRA proxy gain in the current runner. The project needs to record that gap before planning the next offline stress test.

Consequence: `scripts\127_synthesize_scaleup_attribution_gaps.ps1` reads existing reports only and keeps training, rollout, model loading, GPU jobs, OpenVLA-OFT, and paper claims blocked.

Current result: The synthesis passed and recorded that Distributional TCA-Select currently adds no extra LoRA proxy gain in the bounded runner. The next decision is to plan an offline candidate-ambiguity stress test before any selection-specific claim.

## TCA-Select Ambiguity Stress-Test Plan

Decision: Add a planning-only gate for an offline TCA-Select ambiguity stress test.

Reason: Distributional TCA-Select needs a stress test with ambiguous target/action candidates before claiming inference-time selection gain.

Consequence: `scripts\128_plan_tca_select_ambiguity_stress_test.ps1` defines CPU-only offline proxy metrics and pass/fail criteria while keeping training, rollout, model loading, GPU jobs, OpenVLA-OFT, privileged inference, external verifiers, and paper claims blocked.

Current result: The plan passed and authorized only a CPU-only offline stress-test runner over existing local counterfactual artifacts.

## Offline TCA-Select Ambiguity Stress-Test Runner

Decision: Add the CPU-only offline ambiguity stress-test runner.

Reason: The planning gate authorized a safe offline proxy runner to isolate Distributional TCA-Select gain without model loading, training, rollout, GPU jobs, or paper claims.

Consequence: `scripts\129_run_tca_select_ambiguity_stress_test.ps1` may compare TCA-Select against a top-heatmap baseline over synthetic ambiguous candidates generated from local HDF5 action snippets only.

Current result: The runner passed over 16 local offline counterfactual records. Distributional TCA-Select reduced wrong-target proxy rate from 1.0 for the top-heatmap baseline to 0.0, with action L1 delta -0.164299, while keeping model loading, training, rollout, GPU jobs, simulator execution, OpenVLA-OFT, and paper claims false.

Next decision: refresh the attribution synthesis/evidence table to include this selection-specific offline proxy evidence, without promoting it to standard success or paper-grade evidence.

## Stress-Aware Attribution Synthesis Refresh

Decision: Extend the scale-up attribution synthesis to read the offline TCA-Select ambiguity stress report when present.

Reason: The project needs to preserve both facts at once: the bounded LoRA runner shows zero additional selection delta, while the ambiguity stress test shows selection-specific proxy gain against a top-heatmap baseline.

Consequence: `scripts\127_synthesize_scaleup_attribution_gaps.ps1` remains report-only and now separates LoRA adaptation attribution from inference-time selection attribution without unlocking rollout, benchmark, or paper claims.

## Stress-Aware Offline Evidence Table Refresh

Decision: Extend the consolidated offline evidence table to include the TCA-Select ambiguity stress report when present.

Reason: The evidence table should show all current offline proxy arms in one place, including the selection-specific stress row, while keeping detailed top-heatmap deltas in JSON.

Consequence: `scripts\124_generate_offline_evidence_gap_report.ps1` remains report-only and adds `tca_select_ambiguity_stress_included` without changing rollout, training, model-loading, GPU, or paper-claim policy.

Current result: The refreshed evidence table contains 10 rows and includes the `Distributional TCA-Select ambiguity stress` row with wrong-target proxy delta -1.0 and action L1 delta -0.164299 versus the top-heatmap baseline. Learned-policy rollout scaling and paper claims remain blocked.

Next decision: plan a report-only candidate-generation readiness check before any real learned-policy inference or rollout.

## Candidate-Generation Readiness Plan

Decision: Add a report-only readiness plan before attempting learned-policy candidate action heatmap generation.

Reason: The project now has offline TCA-Select ambiguity evidence, but real candidate generation would require a separate model-inference gate. A contract checker can safely validate tensor shapes, metadata, and no-privileged-inference requirements first.

Consequence: `scripts\130_plan_candidate_generation_readiness.ps1` records future gates and routes the next safe task to synthetic-tensor contract checking, not real inference or rollout.

Current result: The plan passed and selected a synthetic-tensor candidate-generation contract checker as the next safe task. Real candidate-generation smoke execution remains blocked until a separate risk-gated model-inference task.

## Candidate-Generation Contract Checker

Decision: Add a synthetic-tensor candidate-generation contract checker before any real learned-policy candidate generation.

Reason: The project needs to validate heatmap/candidate/metadata/TCA-Select contracts without crossing into heavy model inference.

Consequence: `scripts\131_check_candidate_generation_contract.ps1` may run safely as a synthetic checker and keeps real candidate-generation smoke execution false.

Current result: The checker passed on synthetic tensors and validated the candidate/heatmap/metadata/TCA-Select contract without model loading, model inference, training, rollout, GPU jobs, simulator execution, OpenVLA-OFT, external verifiers, privileged inference, or paper claims.

Next decision: create a planning-only risk gate for a separately bounded real candidate-generation smoke.

## Real Candidate-Generation Smoke Plan

Decision: Add a planning-only risk gate for future bounded real candidate-generation smoke.

Reason: The synthetic contract is green, but real candidate generation would require heavy import and single-sample inference gates. Those must be explicit and task-local.

Consequence: `scripts\132_plan_real_candidate_generation_smoke.ps1` can authorize future implementation planning while keeping actual smoke execution false.

Current result: The plan is green for implementation planning and has no blockers. Actual real candidate-generation smoke execution remains false until a future script is run with all required task-local gates.

Next decision: implement the bounded real candidate-generation smoke script and tests, but keep default execution blocked.

## Bounded Real Candidate-Generation Smoke Scaffold

Decision: Add the separately gated real candidate-generation smoke scaffold while keeping default execution blocked.

Reason: The synthetic contract and planning gate are green, but real candidate generation crosses heavy-import and single-sample inference gates. The runner must therefore require all task-local gates and keep the normal safe stack free of heavy model execution.

Consequence: `scripts\133_bounded_real_candidate_generation_smoke.ps1` refuses to run unless `ALLOW_REAL_CANDIDATE_GENERATION_SMOKE=1`, `ALLOW_HEAVY_IMPORT=1`, and `ALLOW_SINGLE_SAMPLE_INFERENCE=1` are all set. When run after a green risk assessment, it may perform one local CPU SmolVLA synthetic action decode, build a low-resolution candidate heatmap, and run Distributional TCA-Select. It still forbids downloads, training, rollouts, simulator execution, OpenVLA-OFT, external verifiers, privileged state, token access, and paper claims.

## Bounded Real Candidate-Generation Smoke Result

Decision: Treat the bounded real candidate-generation smoke as passed engineering evidence.

Reason: After a green risk assessment, the task-local gates were set only for this execution. The runner loaded local SmolVLA on CPU, ran one synthetic `select_action` call, built four low-resolution candidates, and selected a target-consistent candidate with Distributional TCA-Select.

Consequence: The result supports the candidate-generation interface path but remains non-paper evidence. It does not unblock rollout, benchmark, or paper claims. The next step is report-only synthesis and a bounded offline candidate-generation comparison plan.

## Learned-Policy Rollout Diagnostic No-Go

Decision: Do not scale learned-policy LIBERO rollouts from the current SmolVLA checkpoint state.

Reason: Bounded diagnostics covering gripper strategy, action scale, prompt format, camera aliases, state adapters, and HDF5 demo init-state recheck all executed but produced zero reward and zero diagnostic success. The explicit action bridge reported policy action shape `[1, 6]`, environment action dimension `7`, no implicit padding, and no truncation, so the current blocker is not a simple action-shape adapter failure.

Consequence: Treat the current learned-policy rollout result as a concrete failure diagnosis, not benchmark evidence. The next execution-first milestone should move to fixed-integrity ActionMap vs TCA-Map tiny offline training/evaluation on real LIBERO HDF5 snippets, where loss curves and offline proxy metrics can test the method without depending on the current checkpoint's rollout competence.

Training happened: false. LoRA training happened: false. Loss was computed: false; no loss because this was not a training task. Rollout happened: true, bounded diagnostic only. Paper-grade claim: false.

## Tiny Offline ActionMap vs TCA-Map Training/Eval Result

Decision: Treat the first tiny offline ActionMap vs TCA-Map training/eval as weak evidence for TCA-Map on this exploratory split.

Reason: Both ActionMap and TCA-Map losses decreased under bounded CPU head-only training, so the milestone produced a valid loss/metric result. However, TCA-Map worsened target accuracy, wrong-target proxy rate, and standard proxy score on the deterministic holdout records. TCA-Map improved action L1 and counterfactual margin, but not enough to offset the failed target prediction. Distributional TCA-Select added no measurable gain in this run.

Consequence: Do not claim this supports TCA-Map. The next required LoRA comparison may proceed only as an attribution check, with ActionMap + LoRA vs TCA-Map + LoRA reported directly. If ActionMap + LoRA matches or beats TCA-Map + LoRA, report weak novelty rather than forcing a positive result.

Training happened: true. LoRA training happened: false. Loss was computed: true. Rollout happened: false. Paper-grade claim: false.

## Final Research Reset Decision

Decision: kill both current RA-L-stable routes and reset topic selection.

Reason: Target-Prior TCA-Map produced strong fixed-prior offline proxy evidence, but the online 7D action-quality gate failed and valid rollout-level support was not established. CSS-Shield produced promising controlled diagnostics, but Phase 2 native-action testing showed full vs safety-only wrong-target delta `0.0`, full vs clipping-only wrong-target delta `0.0`, full intervention rate `1.0`, and reward/success `0.0 / false`.

Consequence: neither route should continue as the main RA-L route. The next research topic must be rollout-first, baseline-first, and selected only after a short literature-driven screen with kill criteria defined before implementation.

Training happened: false. LoRA training happened: false. Loss was computed: false. Rollout happened: false for this reset package. Paper-grade claim: false.

## Tiny Offline LoRA Attribution Comparison Result

Decision: Treat the required tiny LoRA attribution comparison as weak evidence for the current TCA-Map formulation.

Reason: The LoRA comparison used the same deterministic split as the weak head-only result and passed the requested sanity checks. Both ActionMap + LoRA and TCA-Map + LoRA losses decreased. However, ActionMap + LoRA beat TCA-Map + LoRA on eval standard proxy, target top1 accuracy, wrong-target proxy, and action-target consistency. TCA-Map + LoRA improved action L1 and counterfactual margin only. Distributional TCA-Select candidate scores were non-degenerate but added no measured gain.

Consequence: Do not claim LoRA supports TCA-Map. The next milestone should debug TCA target labels/conditioning or revise the TCA-Map formulation before scaling. If future LoRA or larger-split results keep ActionMap stronger, the novelty claim should be killed or pivoted.

Training happened: true. LoRA training happened: true. Loss was computed: true. Rollout happened: false. Paper-grade claim: false.

## Publishability Gate Audit Result

Decision: Do not downgrade fixed-prior TCA as leakage under the current offline proxy interface, but do not proceed directly to limited rollout because the gain is target-concentrated. Kill TCA-Select as a core contribution for now.

Reason: The audit classified both instruction-text prior and fixed learned+text fusion as `A_valid_test_time_semantic_prior` under the explicit assumption that candidate/task natural-language text is available at test time. Neither prior uses BDDL metadata, eval labels, dataset target labels, filenames, task ids, or manifest target fields as inference-time target proxies. However, fixed-prior TCA + LoRA beat ActionMap + LoRA across all seeds on `8 / 9` eval task groups but only `1 / 2` target groups. Target `0` was approximately tied or slightly worse than ActionMap, while target `1` drove most of the standard-proxy and wrong-target improvement. The oracle selector upper bound had `0.0` standard-proxy delta over non-select fixed-prior TCA, and current TCA-Select had `0.0` turnover.

Consequence: The next execution-first milestone should redesign or calibrate the learned target head / target prior robustness while preserving the fixed metrics, ActionMap baseline, fixed-prior TCA, hard learned-target TCA, LoRA attribution, and oracle upper-bound arms. Limited rollout should wait until the target-concentration issue is understood. TCA-Select should be treated as killed as a core contribution unless a future targeted selector stress test demonstrates real headroom.

Training happened: true. LoRA training happened: true. Loss was computed: true. Rollout happened: false. Paper-grade claim: false.

## 64-Record Multi-Seed Fixed-Prior Offline Validation Result

Decision: Continue fixed-prior TCA-Map as viable exploratory offline-proxy evidence after the 64-record scale-up, but move the next method work to learned target-head redesign and demote TCA-Select as a core contribution.

Reason: The full deterministic scaled manifest was executed with `64` records, `48 / 16` train/eval records, `10` tasks, balanced target classes, and seeds `11, 23, 37`. Fixed-prior TCA + LoRA beat ActionMap + LoRA in `3 / 3` seeds with mean standard-proxy advantage `0.427353` and std `0.002126`, while improving wrong-target proxy in `3 / 3` seeds. Fixed-prior TCA head-only also beat ActionMap head-only with mean advantage `0.461798`. However, LoRA still hurt fixed-prior TCA relative to fixed-prior head-only, hard learned-target TCA remained weaker/unstable, and TCA-Select again had `0 / 3` nontrivial gains.

Consequence: The next execution-first milestone should redesign the learned target head or target-prior calibration under the same fixed split/metric discipline, not scale TCA-Select as a central claim. LoRA remains a required attribution/fairness arm, not a performance-improvement claim. The 64-record result is still exploratory offline proxy evidence only and must not be called standard success, rollout success, or paper-grade evidence.

Training happened: true. LoRA training happened: true. Loss was computed: true. Rollout happened: false. Paper-grade claim: false.

## 32-Record Multi-Seed Fixed-Prior Offline Validation Result

Decision: Continue fixed-prior TCA-Map cautiously after the 32-record offline proxy scale-up, but do not upgrade the evidence to paper-grade.

Reason: The deterministic scaled manifest contains `32` pairs / `64` records, and the milestone executed the first `32` records with `24 / 8` train/eval records, `10` tasks, balanced target classes, and five fixed seeds. Fixed-prior TCA + LoRA beat ActionMap + LoRA in `5 / 5` seeds with mean standard-proxy advantage `0.429379` and std `0.003737`, while improving wrong-target proxy in `5 / 5` seeds. Fixed-prior TCA head-only also beat ActionMap head-only with mean advantage `0.46886`. However, LoRA still hurt fixed-prior TCA relative to fixed-prior head-only, and TCA-Select again had `0 / 5` nontrivial gains.

Consequence: The next execution-first milestone should run the `64`-record split with `1` to `3` seeds if runtime remains bounded. Keep the learned target head as a separate redesign target. Treat LoRA as required attribution/fairness evidence, not a performance-improvement claim. De-emphasize or kill TCA-Select as a central contribution unless future scaled evidence shows a meaningful gain.

Training happened: true. LoRA training happened: true. Loss was computed: true. Rollout happened: false. Paper-grade claim: false.

## Multi-Seed Fixed-Prior Offline Validation Result

Decision: Treat fixed-prior TCA advantage as stable on the bounded 16-sample offline proxy split, but do not elevate the result to paper-grade evidence.

Reason: Across seeds `11, 23, 37, 53, 71`, fixed-prior TCA + LoRA beat ActionMap + LoRA in `5 / 5` seeds with mean standard-proxy advantage `0.426798` and std `0.004095`. Wrong-target proxy improved in `5 / 5` seeds. Fixed-prior head-only TCA also beat ActionMap head-only consistently. However, LoRA consistently hurt fixed-prior TCA relative to fixed-prior head-only, and TCA-Select had `0 / 5` nontrivial gains.

Consequence: The next execution-first milestone should be a larger offline split if more counterfactual pairs are available. If not, redesign the learned target head. Keep LoRA as a required attribution/robustness track, not a performance claim. De-emphasize or kill TCA-Select as a core contribution unless future evidence shows a meaningful gain.

Training happened: true. LoRA training happened: true. Loss was computed: true. Rollout happened: false. Paper-grade claim: false.

## Scaled Fixed-Prior Offline Comparison Result

Decision: Continue TCA-Map cautiously under the corrected target-prior formulation, while keeping learned-target-head redesign and TCA-Select de-emphasis as active conclusions.

Reason: On the smallest larger deterministic split (`16` records, `12 / 4` train/eval), fixed-prior TCA + LoRA reached standard proxy `0.854` and wrong-target proxy `0.0`, beating ActionMap + LoRA at standard proxy `0.427546` and wrong-target proxy `0.5`. Fixed-prior TCA head-only also beat ActionMap head-only. Hard learned-target TCA remained weak relative to fixed-prior TCA, and TCA-Select again added no measurable gain.

Consequence: The next execution-first milestone may be a larger offline split if more pairs are available, or multi-seed validation on the current scaled split. Do not claim paper-grade evidence. Keep TCA-Select secondary unless future results show meaningful selector gain.

Training happened: true. LoRA training happened: true. Loss was computed: true. Rollout happened: false. Paper-grade claim: false.

## TCA Label/Conditioning Debug Audit Result

Decision: Treat the weak TCA-Map result as a target-classifier/generalization failure rather than a confirmed label or metric bug.

Reason: The audit used the same deterministic 8-sample split and verified that target labels, action labels, candidate IDs, target-conditioned inputs, shape handling, wrong-target direction, and TCA-Select scores are internally consistent. TCA also overfit one sample. However, the learned target classifier predicted the wrong target for both eval records. Oracle-target TCA evaluation improved standard proxy from `0.0` to `0.86561`, showing that the action branch can work when the target is correct.

Consequence: No minimal source-code bug patch was applied. Do not scale training, LoRA, or rollout as a response to this result. The next milestone should revise or debug TCA target-conditioning design on the same split. If a revised target-conditioning design still loses to ActionMap, the current TCA-Map formulation should be killed or pivoted.

Training happened: true, diagnostic tiny heads only. LoRA training happened: false. Loss was computed: true. Rollout happened: false. Paper-grade claim: false.

## TCA Target-Prior Rescue Diagnostic Result

Decision: Keep the TCA target-conditioned action mechanism alive as a diagnostic path, but treat the current learned target head as failed on the tiny holdout split.

Reason: The target head fit the training records but inverted both eval targets. Target CE decreased from `0.693147` to `0.121261`, train target top1 was `1.0`, eval target top1 was `0.0`, and eval target top-k was `1.0`. Oracle-target TCA and instruction-text-prior TCA both reached standard proxy `0.86561` with wrong-target proxy `0.0`. Soft marginalization and soft target distributional selection did not improve over hard target prediction because the learned target probabilities still placed most mass on the wrong target.

Consequence: Do not scale sample count, LoRA, or rollout yet. The next execution-first step should implement the smallest target-prior/classifier fix and rerun the exact same head-only ActionMap vs TCA-Map comparison. If the target-prior-fixed TCA still loses to ActionMap, the current TCA target-head formulation should be redesigned or killed.

Training happened: true, diagnostic tiny heads only. LoRA training happened: false. Loss was computed: true. Rollout happened: false. Paper-grade claim: false.

## Target-Prior-Fixed Head Comparison Result

Decision: Keep the current TCA-Map target-conditioned action mechanism alive, but do not treat the learned target head as solved. Revise Distributional TCA-Select before scaling or rerunning LoRA attribution.

Reason: On the fixed 8-sample split, hard learned-target TCA remains weak with `standard_proxy_score=0.0` and `wrong_target_proxy_rate=1.0`, while instruction-text-prior TCA recovers to `standard_proxy_score=0.86561` and `wrong_target_proxy_rate=0.0`, matching the oracle-target upper-bound. ActionMap remains at `standard_proxy_score=0.434797`. Distributional TCA-Select adds no measurable gain over the best non-oracle target prior in this run.

Consequence: the next execution-first task should target TCA-Select redesign or a concrete target-prior/classifier improvement. Do not use this tiny offline proxy as paper-grade evidence.

Training happened: true, diagnostic tiny heads only. LoRA training happened: false. Loss was computed: true. Rollout happened: false. Paper-grade claim: false.

## TCA-Select Target-Uncertainty Audit Result

Decision: Use the fixed learned+text target prior for the next LoRA attribution rerun, and de-emphasize Distributional TCA-Select until it demonstrates a meaningful gain beyond target-prior correctness.

Reason: The audit found no fusion implementation bug or class-index mismatch. Equal learned+text fusion failed because the learned target prior was confidently wrong and overwrote the correct instruction-text prior on both eval records. A temperature-calibrated, conflict-aware fixed fusion recovered standard proxy `0.86561` and wrong-target proxy `0.0`, matching instruction-text prior and oracle-target TCA on this tiny split. The revised uncertainty-marginalized TCA-Select produced only a weak `+0.005128` top-k-uniform standard-proxy delta with no wrong-target improvement, and no gain over instruction-text or fixed-fusion priors.

Consequence: The next execution-first milestone should rerun LoRA attribution with the fixed target prior on the same split. Do not scale or claim paper-grade evidence. Keep TCA-Select as a secondary/uncertain contribution unless a future selector adds measurable value.

Training happened: true, diagnostic tiny heads only. LoRA training happened: false. Loss was computed: true. Rollout happened: false. Paper-grade claim: false.

## Fixed-Prior LoRA Attribution Result

Decision: Keep TCA-Map viable under a corrected target prior, but keep the learned target head as an unresolved bottleneck and de-emphasize TCA-Select as a core contribution.

Reason: On the same fixed 8-sample split, fixed learned+text fusion TCA + LoRA reached standard proxy `0.910293` and wrong-target proxy `0.0`, beating ActionMap + LoRA at standard proxy `0.454351` and wrong-target proxy `0.5`. Hard learned-target TCA + LoRA still failed with standard proxy `0.0` and wrong-target proxy `1.0`, so the improvement comes from target-prior correction rather than a solved learned target classifier. TCA-Select added `0.0` standard-proxy and wrong-target delta over fixed-fusion TCA + LoRA.

Consequence: The next execution-first milestone may cautiously scale the offline split, but must preserve ActionMap + LoRA and hard learned-target TCA + LoRA as baselines. Do not claim paper-grade evidence. TCA-Select should remain an ablation, not the main claim, unless future scaled diagnostics show a measurable contribution.

Training happened: true. LoRA training happened: true. Loss was computed: true. Rollout happened: false. Paper-grade claim: false.

## Representation Sensitivity Audit Result

Decision: Keep Target-Prior TCA-Map as the main method, but do not claim target-information collapse from this audit.

Reason: The audit used the fixed 64-record offline proxy split and did not extract full VLA hidden states. Cached proxy representations were target-sensitive under target swaps, while fixed semantic target-prior reinjection continued to improve ActionMap + LoRA from `0.429275` to `0.856612` standard proxy and reduced wrong-target proxy from `0.5` to `0.0`. This supports target-prior reinjection for action-pathway grounding/wrong-target correction, not a hidden-collapse claim.

Consequence: The next execution-first milestone may be a limited fixed-prior rollout diagnostic after a green rollout risk assessment. Learned target-head redesign remains a bottleneck. TCA-Select should be killed or de-emphasized as a core contribution because it again added `0.0` over fixed-prior TCA.

Training happened: true. LoRA training happened: true. Loss was computed: true. Rollout happened: false. Paper-grade claim: false.

## Fixed-Prior Rollout Readiness Gate Result

Decision: Do not run the limited fixed-prior rollout diagnostic yet.

Reason: The environment plumbing and non-leaking semantic target-prior source are ready, but the current fixed-prior offline proxy action path is not rollout-ready. Offline records use `ACTION_PREFIX_DIM=4`, while LIBERO env actions are `7D`; the validated adapter rejects this with `unsupported action dimension mapping: policy_dim=4, env_action_dim=7`. Gripper, rotation, and coordinate conventions are unresolved.

Consequence: The next execution-first milestone should be a narrow action-bridge/data-path fix that preserves `7D` LIBERO actions for fixed-prior ActionMap/TCA rollout candidates and validates the bridge on HDF5. After that, rerun the same readiness gate; run rollout only if it turns green.

Training happened: false. LoRA training happened: false. Loss was computed: false. Rollout happened: false. Paper-grade claim: false.

## Online 7D Diagnostic Head Result

Decision: treat the non-leaking online 7D ActionMap/TCA diagnostic head as implemented, but do not treat its rollout as method success.

Reason: the diagnostic trained CPU 7D linear heads from local LIBERO HDF5 training labels while filtering the rollout demo path out of training. In bounded matched-init rollout, ActionMap-7D, fixed-prior TCA-7D, and hard learned-target TCA-7D generated actions online from current observation/instruction and did not use same/future HDF5 actions at inference. However, reward and success stayed `0.0 / false` for every method and baseline variant. Fixed-prior TCA showed only a small partial target-movement advantage over ActionMap, not a reward/success improvement.

Consequence: fixed-prior TCA has `fixed_prior_tca_valid_rollout_support=false` and blocker classification `online_7d_head_partial_target_movement_no_success`. The next execution-first milestone should diagnose online action quality/head training before scaling rollout. Do not claim paper-grade success or rollout-level TCA support from this result.

Training happened: true. LoRA training happened: false. Loss was computed: true. Rollout happened: true, bounded diagnostic only. Paper-grade claim: false.

## Online 7D Action-Quality Diagnosis Result

Decision: do not scale rollout from the current online 7D heads; redesign target-prior conditioning/head features first.

Reason: the action-quality diagnosis showed that ActionMap-7D and fixed-prior TCA-7D actions are almost identical, with mean action L2 `0.00712081`. Fixed-prior TCA is slightly better than ActionMap on supervised 25-step 7D L2 (`0.988163728` vs `0.992624014`) and teacher-forced full-demo L2 delta (`-0.001041313`), but those gains are too small to expect rollout success. More importantly, the simple mean-action baseline has lower 25-step 7D L2 (`0.57299313`) than all learned heads, so the current head is not yet a strong action decoder. Full-demo gripper timing is also late: expert first open step `62`, fixed-prior TCA predicted step `100`.

Consequence: treat the current result as a concrete failure diagnosis that directly unblocks the next experiment. The next milestone should make the 7D head beat the mean-action baseline and make fixed-prior conditioning produce materially different actions before another method rollout. Fixed-prior TCA valid rollout-level support remains `false`.

Training happened: true. LoRA training happened: false. Loss was computed: true. Rollout happened: true, bounded diagnostic report regenerated only. Paper-grade claim: false.

## Bounded 7D Action-Head Redesign Gate Result

Decision: do not run another method rollout from the current redesigned 7D heads.

Reason: the bounded redesign gate evaluated current linear heads, normalized ridge heads, split translation/rotation/gripper heads, a small CPU MLP fixed-prior TCA head, a mean-residual fixed-prior TCA head, and a phase-aware fixed-prior TCA head on the same non-leaking split. The best redesigned method was `small_cpu_mlp_fixed_prior_tca_7d`, and it improved substantially over ActionMap (`0.669078005` vs `0.992624014` eval 7D L2). However, it still failed to beat the mean-action baseline (`0.57299313`) and teacher-forced evaluation also stayed worse than the mean-action baseline. The rollout gate is therefore red.

Consequence: fixed-prior TCA has no valid rollout-level support from this head family. The next step must either redesign target-prior conditioning/action features so a non-leaking head beats the mean baseline, or package the current evidence honestly as offline proxy plus bridge diagnostics with a rollout caveat. Do not revive TCA-Select as a core claim from these results.

Training happened: true. LoRA training happened: false. Loss was computed: true. Rollout happened: false. Paper-grade claim: false.

## ExecSpec-Repair State 0-1 Result

Decision: continue ExecSpec-Repair after bounded mismatch and exact-init replay diagnostics.

Reason: the first local LIBERO HDF5 demo reproduced substantial executable-spec mismatch, and exact-init replay showed that plausible mismatches degrade execution. Correct expert replay reached reward/success `1.0 / true`; gripper sign flip and translation-scale mismatch both produced reward/success `0.0 / false`.

Key metrics: strongest HDF5 mismatch was `gripper_sign_flip` with action L2 mean `2.0` and gripper mismatch rate `1.0`. Translation-scale mismatch had action L2 mean `0.363970119`. Supervised diagonal calibration beat identity, clipping-only, and naive global affine baselines on seven mismatch variants, but this remains calibration/evaluation evidence only.

Consequence: the next execution-first milestone is STATE 2 calibrated repair replay under the same exact-init boundary. Do not claim deployable repair or paper-grade evidence until a calibrated replay variant beats identity, clipping-only, and naive global-scale controls without future expert actions as method rollout actions.

Training happened: false. LoRA training happened: false. Loss was computed: false. Replay/rollout happened: true, bounded exact-init diagnostic only. GPU/download/OpenVLA-OFT happened: false. Paper-grade claim: false.

## ExecSpec-Repair Archive And Topic Tournament

Decision: kill and archive ExecSpec-Repair as a main RA-L-stable route, and start a fresh topic-selection tournament.

Reason: STATE 3.5 found that full ExecSpec-Repair recovered `17 / 19` degraded exact-init replay cases, but the best single simple baseline, `diagonal_affine_calibration`, also recovered `17 / 19`. Full-minus-best-simple recovery gain was `0.0`, and the mismatch-aware selector had `0.0` gain over diagonal affine. The broad claim that mismatch-aware executable-spec repair provides nontrivial value beyond simple baselines is therefore unsupported.

Consequence: Target-Prior TCA-Map, CSS-Shield, and ExecSpec-Repair are all archived as killed or reframed routes. The next route must be rollout/replay/control-first and baseline-first, with a metric within 48 hours and a simple-baseline gap within 72 hours. The recommended next candidate is Active Micro-Probe Goal Disambiguation, pending a pre-implementation kill-gated plan.

Execution boundary: this archive/tournament step was documentation-only. Experiments, training, LoRA training, replay, rollout, loss computation, downloads, GPU jobs, heavy VLA imports, OpenVLA-OFT execution, and paper-grade claims did not occur.

## AMP-GD State 0-1 Result

Decision: continue Active Micro-Probe Goal Disambiguation to State 2 scaling after the first diagnostic, while labeling the evidence as toy control evidence only.

Reason: State 1 produced a direct rollout/control metric on 60 seeded point-world trials. AMP-GD wrong-target rate was `0.0`, compared with no-probe `0.5`, random-probe `0.466666667`, safety-only/clipping-only `0.5`, and nearest-target `0.483333333`. AMP-GD success rate was `1.0`, unsafe/collision rate was `0.0`, probe cost was `0.12`, and extra path length versus no-probe was `0.318929988`.

Consequence: the route passed the first simple-baseline gate but is not RA-L-ready. State 2 should scale the diagnostic and begin the LIBERO/RoboSuite object-observable port. If random-probe, safety-only, nearest-target, or no-probe matches AMP-GD under scaling or in LIBERO/RoboSuite, kill or reframe the route immediately.

Execution boundary: toy rollout/control metric happened. LIBERO/RoboSuite rollout, training, LoRA training, loss computation, downloads, GPU jobs, heavy VLA imports, OpenVLA-OFT execution, and paper-grade claims did not occur.

## AMP-GD State 2 Result

Decision: kill or reframe AMP-GD as the current main RA-L route.

Reason: the State 2 audit found no utility-sign bug and no AMP-GD target-label leakage, but the toy result collapsed as main evidence because deterministic informative-probe and entropy-greedy probe heuristics matched AMP-GD. The LIBERO/RoboSuite port found real, non-leaking object observability and a computable wrong-target metric, but the tested scene had no active ambiguity signal. The tiny micro-probe diagnostic ran and AMP-GD did not beat safety-only; random-probe matched AMP-GD on wrong-target movement.

Consequence: do not scale AMP-GD toy diagnostics or present AMP-GD as RA-L-plausible from current evidence. Honest next options are a narrowly scoped active-ambiguity benchmark with demonstrated probe-revealed hidden state, or selecting a different rollout-first route.

Execution boundary: toy and tiny LIBERO/RoboSuite diagnostic metrics happened. Training, LoRA training, loss computation, downloads, GPU jobs, heavy VLA imports, OpenVLA-OFT execution, benchmark rollouts, and paper-grade claims did not occur.

## ResetSpec-Retarget State 0-1 Result

Decision: kill or reframe ResetSpec-Retarget as the current main RA-L route.

Reason: STATE 1 found the intended exact-init versus default-reset brittleness, and object-relative retargeting improved progress and shifted-trajectory tracking. However, it did not reach reward/success and did not beat the simple fixed global-scale baseline. The global-scale default-reset replay succeeded with reward/success `1.0 / true` and first done `257`, while both object-relative retarget variants stayed `0.0 / false`.

Key metrics: exact-init expert replay reached reward/success `1.0 / true` with first done `260`; default-reset raw, diagonal-affine, and clipping replay all stayed `0.0 / false`; object-relative translation reached EEF-object distance change `-0.232446` and object movement `0.231257`; object-relative translation plus gripper-phase reached distance change `-0.247037` but still failed success.

Consequence: do not scale ResetSpec-Retarget as a main route. Keep the runner as reusable reset/object-pose mismatch infrastructure, but require any future retargeting route to beat global scaling and other action-only baselines before broader replay or paper-readiness work.

Execution boundary: bounded LIBERO/RoboSuite replay/control metrics happened. Training, LoRA training, loss computation, downloads, GPU jobs, heavy VLA imports, OpenVLA-OFT execution, benchmark rollouts, and paper-grade claims did not occur.

## ResetSpec Archive And Next-Topic Pre-Screen

Decision: archive ResetSpec-Retarget and require a strict anti-baseline pre-screen before any new topic implementation.

Reason: five main-route candidates have now failed by the same pattern: they showed some positive evidence, then collapsed against a stronger trivial baseline or an online/replay gate. The next topic must be chosen by first identifying the baseline most likely to kill it.

Killed-route baseline map: Target-Prior TCA-Map by mean-action, CSS-Shield by safety-only, ExecSpec-Repair by diagonal affine, AMP-GD by simple probe/safety baselines, and ResetSpec-Retarget by global scale.

Recommended candidate: Phase-Locked Action Chunk Retiming. It is preferred only at the pre-screen level because it targets temporal phase mismatch, can produce a replay/control metric quickly with existing assets, and has a clear first table against fixed time shift, repeat-last, global scale, diagonal affine, gripper-only phase, and nearest-progress demo baselines.

Consequence: do not implement a new route until the next state explicitly starts and the anti-baseline fields are restated. Any candidate whose first result is offline-only, calibration-solvable, native-VLA-dependent, full-training-dependent, or lacks a 48-hour replay/control metric is invalid.

Execution boundary: documentation-only. Experiments, training, LoRA training, replay, rollout, loss computation, downloads, GPU jobs, heavy VLA imports, OpenVLA-OFT execution, and paper-grade claims did not occur.

## Phase-Locked Action Chunk Retiming State 0-1 Result

Decision: kill or reframe Phase-Locked Action Chunk Retiming as the current main RA-L route.

Reason: STATE 1 produced real bounded LIBERO/RoboSuite replay/control metrics and successfully created temporal phase failures: all nine perturbations degraded exact-init expert replay. However, event-locked retiming recovered over raw perturbed replay on `0 / 9` perturbations and beat the best simple baseline on `0 / 9`. Simple baselines recovered or matched key perturbations, including gripper-only correction for gripper timing, fixed time shift for chunk shift, linear time warp for time compression, and diagonal-affine/raw-equivalent behavior for boundary offset.

Consequence: do not scale Phase-Locked Retiming as a main route. Keep the perturbation and replay table as reusable phase-mismatch infrastructure, but require any future timing route to show recovery over raw replay and a win over fixed-shift, gripper-only, linear-warp, repeat-last, diagonal-affine, and nearest-progress baselines before broader replay or paper-readiness work.

Execution boundary: bounded LIBERO/RoboSuite replay/control metrics happened. Training, LoRA training, loss computation, downloads, GPU jobs, heavy VLA imports/model loading, OpenVLA-OFT execution, benchmark rollouts, and paper-grade claims did not occur.

## Phase-Locked Retiming Archive And V3 Topic Filter

Decision: archive Phase-Locked Retiming and require a stricter per-failure-mode simple-baseline screen before any new route implementation.

Reason: Phase-Locked Retiming produced real replay/control evidence and all nine perturbations degraded replay, but the method recovered over raw on `0 / 9` and beat best simple on `0 / 9`. Separate obvious baselines solved or matched their own slices: gripper-only for gripper timing, fixed shift for chunk shift, linear warp for time compression, and simple timing/action baselines for other perturbations.

Consequence: all killed routes are now summarized with the new failure pattern. Future topics must beat the best single simple baseline and the best per-failure-mode simple baseline. Candidate v3 recommends Post-Intervention Resume-Point Selection only as a pre-screened next candidate; it is not implemented.

Execution boundary: documentation-only. Experiments, training, LoRA training, replay, rollout, loss computation, downloads, GPU jobs, heavy VLA imports, OpenVLA-OFT execution, and paper-grade claims did not occur.

## TL-ChunkRepair State 0 Start

Decision: start a fresh TL-ChunkRepair route from main commit `1409dd1f737f9e70c4121dc01a1378ce16942a3b` on branch `codex/tl-chunkrepair-state0-state1`.

Reason: the new hypothesis targets temporal safety/property violations inside an already proposed action chunk, while explicitly requiring a win over both the best single simple baseline and per-failure-mode simple baselines.

Consequence: reuse exact-init HDF5 replay and safe runner infrastructure, but do not continue Target-Prior TCA-Map, CSS-Shield, ExecSpec-Repair, AMP-GD, ResetSpec-Retarget, or Phase-Locked Retiming as active routes. STATE 1 must produce a real replay/control metric or kill.

Execution boundary at STATE 0: documentation and implementation setup only. No training, loss computation, GPU job, download, OpenVLA-OFT execution, model loading, or paper-grade claim.

## TL-ChunkRepair State 1 Result

Decision: kill or reframe TL-ChunkRepair as the current main route.

Reason: STATE 1 produced real bounded exact-init LIBERO/RoboSuite replay/control metrics, and the finite-state monitor repaired symbolic temporal violations in all eight perturbations. However, TL-ChunkRepair recovered no reward/success/safe-success, did not beat the best single simple baseline, and did not beat the best per-failure-mode simple baseline.

Key metrics: exact-init expert replay succeeded; total simulator steps `19803`; variants `73`; perturbations tested `8`; perturbations that degraded replay `7 / 8`; TL violation reductions `8 / 8`; TL safe-success `0 / 8`; TL reward/success `0.0 / 0`; best single simple baseline `no_repair` reached reward/success `1.0 / 1`; TL beat best simple baseline on `0` degraded perturbations.

Consequence: do not proceed to STATE 2. The route demonstrates observable temporal-property repair, but it fails the hard gate because the improvement is symbolic rather than replay/control recovery and simple baselines remain at least as good on useful metrics.

Execution boundary: bounded exact-init replay happened. Training, LoRA training, loss computation, downloads, GPU jobs, heavy VLA imports/model loading, OpenVLA-OFT execution, benchmark rollouts, and paper-grade claims did not occur.

## TL-ChunkRepair Archive

Decision: archive TL-ChunkRepair as a hard-killed RA-L-stable route and update global killed-route summaries.

Reason: STATE 1 produced real exact-init LIBERO/RoboSuite replay/control metrics and reduced symbolic temporal violations on `8 / 8`, but TL safe-success was `0 / 8`, TL reward/success was `0.0 / 0`, and the best single simple baseline `no_repair` achieved reward/success `1.0 / 1`. TL failed both the best single simple baseline gate and the best per-failure-mode simple baseline gate.

Consequence: do not continue TL-ChunkRepair to STATE 2 or present it as RA-L-stable. Keep the temporal perturbation runner, exact-init replay diagnostic, temporal property monitor, violation metrics, and baseline suite as reusable diagnostic infrastructure only.

Global rule added: a method is invalid for RA-L-stable continuation if it improves symbolic/proxy constraints but degrades or fails real replay/control utility compared with a simple baseline.

Execution boundary: documentation-only archive. Experiments, training, LoRA training, replay/rollout, loss computation, downloads, GPU jobs, heavy VLA imports/model loading, OpenVLA-OFT execution, and paper-grade claims did not occur.

## ContactTube-Aug State 0-1 Result

Decision: kill or reframe ContactTube-Aug before STATE 2.

Reason: STATE 1 produced a real bounded LIBERO/RoboSuite replay/control metric and extracted contact-tube structure from runtime replay traces, but ContactTube-Aug failed the hard continuation gates. The exact-init no-op upper bound succeeded, runtime object pose was available, and random jitter/pose jitter were worse on tube preservation. However, ContactTube-Aug had controller-valid action rate `0.849265` with clip-step rate `0.150735`, below the predeclared validity gate, and simple object-relative translation retargeting beat it on contact-tube preservation error (`0.009154` versus `0.015226`).

Key metrics: total simulator steps `1621`; variants `6`; exact-init no-op reward/success `1.0 / true`; ContactTube-Aug reward/success `0.0 / false`; simple object-relative reward/success `0.0 / false`; HDF5 object pose available `false`; runtime object pose available `true`; ContactTube-Aug beats random action jitter `true`; ContactTube-Aug beats random pose jitter `true`; ContactTube-Aug beats simple object-relative retargeting `false`.

Consequence: do not run STATE 2 training for this branch. Preserve the ContactTube extraction/replay runner as reusable diagnostic infrastructure only. Any future contact-preserving augmentation route must predeclare a new mechanism that is controller-valid by construction and must beat simple object-relative translation before model training.

Execution boundary: bounded exact/default-reset LIBERO/RoboSuite replay happened. Training, LoRA training, loss computation, downloads, GPU jobs, heavy VLA imports/model loading, OpenVLA-OFT execution, benchmark rollouts, and paper-grade claims did not occur.

## ContactTube-Aug Archive And Global Killed-Route Update

Decision: archive ContactTube-Aug as a hard-killed route before training and update global killed-route summaries.

Reason: the STATE 1 replay/control diagnostic already triggered the hard gates: ContactTube-Aug action validity was too low (`0.849265` controller-valid action rate, `0.150735` clip-step rate), and simple object-relative translation retargeting beat ContactTube-Aug on contact-tube preservation (`0.009154` versus `0.015226`). Training after this would not test useful augmentation; it would train on invalid or inferior generated actions.

Consequence: do not proceed to STATE 2, do not claim ContactTube-Aug as RA-L-stable, and keep only the extraction/replay/validity infrastructure as reusable artifacts. Future data-augmentation topics must demonstrate controller-valid generated actions and beat simple object-relative retargeting before any BC/action-head or VLA training.

Global rule added: a data-augmentation method is invalid for continuation if generated actions are not controller-valid, or if simple object-relative retargeting preserves trajectory/contact metrics better than the proposed augmentation.

Execution boundary: documentation-only archive. No new experiments, replay/rollout, training, LoRA training, loss computation, downloads, GPU jobs, heavy VLA imports/model loading, OpenVLA-OFT execution, benchmark rollouts, or paper-grade claims occurred.
