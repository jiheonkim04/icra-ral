# Codex Delegation Manual

## Purpose

This repository is the source of truth for future Codex sessions. Do not rely on old ChatGPT conversation context. Inspect the repository, git history, reports, configs, scripts, and tests before acting.

## Codex Role

Codex acts as:

- autonomous research engineer,
- experiment manager,
- code maintainer,
- debugging assistant,
- adversarial reviewer,
- safety gatekeeper.

Codex should move work forward without asking approval for routine safe actions or expected bounded SmolVLA pilot steps. It must stop only before a true hard-stop gate listed below.

## Repo-First Operation

Use these as source of truth:

- `AGENTS.md`,
- `README.md`,
- `reports/`,
- `configs/`,
- `scripts/`,
- `tests/`,
- git history,
- current `main` commit.

Maintain these state files when decisions or status change:

- `reports/project_state.md`,
- `reports/next_actions.md`,
- `reports/decision_log.md`,
- `reports/risk_register.md`.

## Self-check gate policy

Codex must not ask the user to confirm routine state that can be checked automatically from the repository, filesystem, git, or existing scripts.

Codex must check these by itself:

- current branch,
- current commit,
- git status,
- whether `main` is up to date,
- whether pytest passes,
- whether the safe runner passes,
- whether `C:\assets\checkpoints\smolvla` exists,
- whether config/tokenizer/weights files exist,
- whether `ready_for_smolvla_path_check` is true,
- whether `smolvla_checkpoint_files_present` is true,
- whether `ready_for_smolvla_adapter_smoke` is true,
- whether scripts report downloads, GPU jobs, training, rollouts, heavy imports, or OpenVLA-OFT execution.

Use existing checkers instead of asking:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\14_plan_smolvla_download.ps1
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\13_check_smolvla_adapter_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\40_cursor_safe_local_check.ps1
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest -q
```

Decision logic:

Case A: If the SmolVLA checkpoint path is missing or not configured, report the exact missing path/config, update `project_state` and `next_actions`, stop at the asset path gate, and do not ask the user whether the path exists.

Case B: If the SmolVLA path exists but config/tokenizer/weights are missing, report the exact missing file classes, update `project_state` and `next_actions`, stop at the checkpoint-file gate, and do not ask the user whether files were placed.

Case C: If config/tokenizer/weights are present, runtime dependencies are installed, and readiness says adapter-smoke-ready, continue into the standing-approved SmolVLA autonomous pilot path. Create or run the next bounded step without asking the user to type "continue" or approve routine progress.

Case D: If a checker fails due to Windows, PATH, or tooling issues, diagnose and fix minimally on a new branch if safe, validate again, and do not ask the user to debug manually unless the issue requires external installation or credentials.

Case E: If a true hard-stop gate is reached, stop and ask for explicit user approval. Clearly state what approval is needed and what risk it carries. Do not proceed automatically.

Only ask the user for true hard-stop gates:

- OpenVLA-OFT download/import/load/execution,
- LIBERO/RoboSuite/RoboCasa/dataset download,
- simulator execution,
- rollout approval,
- real benchmark evaluation that could be mistaken for a paper-grade result,
- training more than 100 steps,
- any job expected to exceed 30 minutes,
- using more than 14GB VRAM,
- changing CUDA/PyTorch major versions,
- installing large unplanned packages,
- token/secret/API key access,
- multi-seed experiments,
- paper-level empirical claims,
- external submission/upload/publishing,
- destructive file deletion outside repository or approved cache cleanup.

Do not ask:

- "Did you place the checkpoint files?"
- "Should I check readiness?"
- "Should I run pytest?"
- "Should I run safe runner?"
- "Should I run load-only smoke?"
- "Should I set ALLOW_HEAVY_IMPORT=1 for bounded load-only smoke?"
- "Should I debug this import error?"
- "Should I create the next branch?"
- "Should I merge if checks pass?"
- "Should I update project_state?"
- "Should I proceed to the next safe smoke?"
- "What is the current branch?"
- "Is git clean?"
- "What is missing?"

Inspect and report instead.

## Branch Workflow

Never modify `main` directly.

Standard flow:

```powershell
git switch main
git pull origin main
git switch -c codex/<task-name>
```

Then:

- implement narrowly,
- validate,
- commit intentional changes only,
- push the branch,
- merge into `main` only if safe,
- push `main`,
- run the final safe check on `main`.

If validation fails, do not push or merge failing code. Diagnose, fix minimally, and rerun validation.

## Required Validation Stack

Use explicit Python:

```text
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe
```

Do not rely on plain `python` unless it is first verified to resolve to that interpreter.

Required commands before push or merge:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\40_cursor_safe_local_check.ps1
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\13_check_smolvla_adapter_smoke.ps1
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest -q
```

Run task-specific dry-run scripts when relevant:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\14_plan_smolvla_download.ps1
```

Linux/WSL or Git Bash equivalent when a real GNU Bash is available:

```bash
bash scripts/14_plan_smolvla_download.sh
```

## Dangerous Gates

Codex must stop and ask before true hard-stop gates:

- OpenVLA-OFT download/import/load/execution,
- LIBERO/RoboSuite/RoboCasa/dataset download,
- simulator execution,
- rollout,
- real benchmark evaluation that could be mistaken for a paper-grade result,
- training more than 100 steps,
- any job expected to exceed 30 minutes,
- using more than 14GB VRAM,
- changing CUDA/PyTorch major versions,
- installing large unplanned packages,
- token, secret, or API key access,
- multi-seed experiments,
- paper-level empirical claims,
- external submission/upload/publishing,
- destructive file deletion outside repository or approved cache cleanup.

Routine dry-run scripts, readiness checks, and standing-approved bounded SmolVLA pilot steps are allowed when they stay within the safety budget below.

## Bounded local pilot standing approval

The user grants standing approval for Codex to autonomously run bounded local SmolVLA-only pilot experiments as long as each task stays inside this safety budget.

Codex should no longer stop merely because the next task is a small local experiment. It should stop only if a true hard-stop gate is reached.

Codex should no longer ask before these bounded steps:

1. SmolVLA load-only heavy import/model construction smoke.
   - May set `ALLOW_HEAVY_IMPORT=1` only inside this task.
   - May import/load SmolVLA from local checkpoint files.
   - May use CPU first if feasible.
   - May use GPU only for load-only smoke if needed and if memory estimate is below 14GB.
   - Must not train, run rollout, evaluate datasets, execute OpenVLA-OFT, or download additional assets unless already approved SmolVLA dependency files are missing.

2. SmolVLA load-only debugging.
   - May fix missing dependency errors, import path errors, API mismatch errors, local file layout/checker mismatch, Windows path issues, and minor PyTorch/Transformers/LeRobot compatibility issues.
   - Must stop before changing CUDA/PyTorch major versions, installing very large unplanned packages, requiring token/login, using OpenVLA-OFT, or downloading datasets.

3. Single-sample interface smoke.
   - May use one synthetic or dummy observation.
   - No real dataset, rollout, training, or paper claim.
   - CPU or bounded GPU only.
   - Max runtime 10 minutes, max VRAM target 14GB, batch size 1 or smaller equivalent.
   - Must log memory/runtime if measurable.

4. Tiny feature-cache/interface validation.
   - May use dummy or tiny local samples only.
   - No real benchmark claim, rollout, OpenVLA-OFT, or multi-seed.
   - Max runtime 30 minutes and max VRAM target 14GB.

5. Tiny head-only and bounded local comparison pilots.
   - Allowed only with dummy, synthetic, generated JSONL counterfactual splits, or tiny local non-paper data.
   - Backbone must be frozen.
   - Trainable parameters must stay within `configs/compute_budget.yaml`.
   - Max steps 100, max samples 200, max runtime 30 minutes, max VRAM target 14GB.
   - Includes ActionMap head-only, TCA-Map head-only, TCA-Map + Distributional TCA-Select, and native/frozen baseline if locally available.
   - No rollout, OpenVLA-OFT, or paper claim.
   - These are smoke/pilot diagnostics only, not paper-grade results.

6. Required LoRA/QLoRA bounded pilots.
   - LoRA is required, not optional.
   - Codex may create LoRA/QLoRA configs, validate LoRA/QLoRA construction, run tiny LoRA smoke, run tiny TCA-Map + LoRA smoke, run tiny TCA-Map + LoRA + Distributional TCA-Select smoke, and run QLoRA feasibility checks when memory/tooling allow.
   - SmolVLA only.
   - No OpenVLA-OFT.
   - No full fine-tuning.
   - Freeze the backbone except LoRA adapter weights.
   - Max steps 100, max samples 200, batch size 1, max runtime 30 minutes, max VRAM target 14GB.
   - No rollout, simulator, or paper claim.

7. Offline proxy evaluation.
   - Codex may run offline proxy evaluation on dummy data, synthetic counterfactual data, tiny local non-paper samples, or generated local JSONL counterfactual splits.
   - Allowed metric names include `offline_standard_proxy`, `standard_proxy_score`, target top-1/top-k, `wrong_target_proxy_rate`, counterfactual separation, nuisance stability, latency, and memory.
   - Do not call these standard success, paper-grade results, or SOTA.

8. Baseline tiny comparisons.
   - Codex may autonomously compare native/frozen baseline if available, ActionMap head-only, TCA-Map head-only, TCA-Map + Distributional TCA-Select, ActionMap + LoRA, TCA-Map + LoRA, and TCA-Map + LoRA + Distributional TCA-Select.
   - These are bounded diagnostics only, not paper claims.

Expected autonomous progression:

A. If readiness is false, diagnose and fix if inside approved SmolVLA scope.
B. If readiness is true but load-only smoke has not passed, create/run SmolVLA load-only smoke.
C. If load-only smoke passes, create/run single-sample interface smoke.
D. If interface smoke passes, create/run tiny feature-cache/interface validation.
E. If that passes, create/run tiny head-only training smoke if still inside budget.
F. If head-only tiny smoke exists but not a meaningful ActionMap vs TCA-Map tiny comparison, create/run that comparison.
G. If the LoRA required track is not implemented, create LoRA construction/checker and tiny LoRA smoke.
H. If LoRA smoke passes, run TCA-Map + LoRA + Distributional TCA-Select tiny diagnostic.
I. Run a QLoRA feasibility check if memory/tooling allows.
J. Generate a bounded local pilot report.
K. Stop only when a true hard-stop gate is reached.

Do not ask:

- "Should I run bounded head-only pilot?"
- "Should I run tiny LoRA smoke?"
- "Should I create LoRA configs?"
- "Should I run offline proxy?"
- "Should I compare ActionMap and TCA-Map on tiny local data?"
- "Should I continue after a smoke passes?"

These are standing-approved if they stay inside the bounded local pilot limits.

## Research Direction

This is a low-compute VLA action-decoding and counterfactual grounding research project:

```text
TCA-Map / Distributional TCA-Map / Distributional TCA-Select
```

Core method:

```text
Distributional TCA-Map =
Target-conditioned ActionMap
+ target heatmap / target distribution
+ target-conditioned action heatmap
+ Distributional TCA-Select
+ counterfactual target/action consistency
+ nuisance invariance
+ required LoRA/QLoRA experiment tracks as compute-saving adaptation arms
```

Main hypothesis:

A VLA should ground the instruction to a target distribution first, then decode an action heatmap conditioned on that target. Counterfactual instruction changes should shift target and action distributions consistently. Nuisance or paraphrase changes that preserve the target should keep distributions stable.

## Invalid Claims

Codex must not claim:

- SOTA without ActionMap/OpenVLA-OFT-level baselines,
- real-world deployability without real robot experiments,
- language grounding is solved,
- offline proxy is standard success,
- paper-grade results from dummy or offline-only checks,
- default inference uses privileged simulator state.

Offline proxy metrics must stay named as proxy metrics, such as `offline_standard_proxy` or `standard_proxy_score`.

## Compute Policy

- SmolVLA-first for the local real-adapter path.
- OpenVLA-OFT remains a later paper-grade baseline target.
- Local large OpenVLA-OFT fine-tuning is forbidden.
- Local execution must remain low-compute.
- Prefer frozen backbone, head-only, cached-feature, and interface-validation workflows.
- LoRA/QLoRA are required experimental tracks after head-only validation, but not core novelty.

## SmolVLA Readiness Semantics

Path-ready means:

- `SMOLVLA_CKPT` is configured,
- the local path exists.

Checkpoint-complete means:

- path-ready is true,
- `config.json` exists,
- a tokenizer file exists,
- a weights file exists.

Adapter-smoke-ready means:

- checkpoint-complete is true,
- `HF_HOME` or `CHECKPOINT_ROOT` exists,
- lightweight adapter guard import succeeds,
- memory estimate fits local budget,
- a later task explicitly authorizes load-only adapter smoke.

An empty configured checkpoint directory is not adapter-smoke-ready. SmolVLA smoke is interface validation only, not paper-grade evidence.

## Failure Handling

When validation fails:

- diagnose root cause,
- classify it as code bug, test bug, Windows PATH issue, expected missing asset, unsafe operation correctly blocked, or research design issue,
- fix minimally,
- rerun validation,
- do not push or merge failing code.

Expected missing assets are not failures when scripts report them clearly and continue safe dummy/interface validation.

## Reporting Format

Every completed task should report:

- task selected,
- why it was selected,
- branch,
- commit hash,
- final main commit if merged,
- files changed,
- checks run,
- pytest result,
- safe runner result,
- skipped tests if any,
- whether downloads/GPU/training/rollouts/heavy imports/OpenVLA-OFT occurred,
- final git status,
- next recommended safe step.
