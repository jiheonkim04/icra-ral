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

Codex should move work forward without asking approval for routine safe actions, but must stop before any dangerous gate listed below.

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

Case C: If config/tokenizer/weights are present and readiness says adapter-smoke-ready, update `project_state` and `next_actions`, prepare the next safe load-only adapter smoke plan, do not perform heavy import or GPU execution, and stop for explicit approval before any heavy import or model load.

Case D: If a checker fails due to Windows, PATH, or tooling issues, diagnose and fix minimally on a new branch if safe, validate again, and do not ask the user to debug manually unless the issue requires external installation or credentials.

Case E: If a dangerous gate is reached, stop and ask for explicit user approval. Clearly state what approval is needed and what risk it carries. Do not proceed automatically.

Only ask the user for:

- actual download approval,
- `ALLOW_DOWNLOADS=1` approval,
- `ALLOW_HEAVY_IMPORT=1` approval,
- GPU inference approval,
- training approval,
- rollout approval,
- simulator execution approval,
- OpenVLA-OFT execution approval,
- token/secret/API key access,
- paper-level empirical claim approval.

Do not ask:

- "Did you place the checkpoint files?"
- "Should I check readiness?"
- "Should I run pytest?"
- "Should I run safe runner?"
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

Codex must stop and ask before:

- any actual download,
- setting `ALLOW_DOWNLOADS=1`,
- setting `ALLOW_HEAVY_IMPORT=1`,
- GPU inference,
- training,
- rollout,
- simulator execution,
- heavy SmolVLA/OpenVLA import,
- OpenVLA-OFT execution,
- token, secret, or API key access,
- paper-level empirical claims.

Routine dry-run scripts and readiness checks are allowed when they explicitly report no downloads, no GPU jobs, no training, no rollouts, no heavy imports, and no OpenVLA-OFT execution.

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
+ optional LoRA/QLoRA as compute-saving support
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
- LoRA/QLoRA are optional support, not core novelty.

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
