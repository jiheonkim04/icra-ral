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
