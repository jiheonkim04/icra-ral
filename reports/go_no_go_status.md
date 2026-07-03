# Go/No-Go Status

## Purpose

This document defines the safe interpretation of the completed local SmolVLA-first smoke stack.

The current status is no-go for the next larger experimental stage until the user explicitly approves exactly one true next gate. Required LoRA/QLoRA planning is still allowed because it is documentation/config/checker work, not training.

## Current Decision

```text
no_go_for_next_larger_experimental_stage
```

The project is go for:

- routine safe checks,
- documentation and checker maintenance,
- planning-only reports,
- required LoRA/QLoRA adapter construction and feasibility planning,
- LoRA/QLoRA planning interpretation and risk review,
- preparing a request for exactly one explicit next gate if the user wants to continue.

The project is no-go for:

- paper-grade empirical claims,
- real dataset training,
- LoRA or QLoRA execution without a bounded runner,
- simulator rollouts,
- OpenVLA-OFT execution,
- multi-seed experiments.

## Evidence Completed

The safe local stack has completed:

- SmolVLA checkpoint/dependency readiness,
- bounded SmolVLA load-only smoke,
- bounded single-sample interface smoke,
- dummy feature-cache/interface validation,
- eval-only cached-feature smoke,
- bounded tiny head-only smoke.
- required LoRA adapter construction plan,
- LoRA tiny-smoke scaffold,
- TCA-Map + LoRA comparison plan,
- QLoRA feasibility check.

These are engineering/interface validations only. They are not standard success, not rollout success, and not paper-grade evidence.

## Generator

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\31_generate_go_no_go_report.ps1
```

It writes ignored runtime reports:

```text
reports\go_no_go_status_report.json
reports\go_no_go_status_report.md
```

The generator reads local reports only. It does not download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims.

## Remaining Hard Gates

Stop before:

- OpenVLA-OFT download/import/load/execution,
- LIBERO/RoboSuite/RoboCasa/dataset download,
- simulator execution,
- rollout,
- real benchmark evaluation,
- LoRA/QLoRA execution without bounded runner and explicit tiny-training gate,
- training beyond the tiny smoke budget,
- package/CUDA/PyTorch changes for QLoRA,
- jobs expected over 30 minutes,
- more than 14GB VRAM,
- major CUDA/PyTorch changes,
- unplanned large package installs,
- token/secret/login requirements,
- multi-seed experiments,
- paper-level empirical claims,
- external submission/upload/publishing.
