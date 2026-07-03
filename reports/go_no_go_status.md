# Go/No-Go Status

## Purpose

This document defines the safe interpretation of the completed local SmolVLA-first smoke stack.

The current status is ready for bounded local SmolVLA pilot work inside the risk-assessed pilot envelope, while still no-go for the next larger paper-grade experimental stage until the relevant risk assessment passes and no external stop gate is reached.

## Current Decision

```text
no_go_for_next_larger_experimental_stage
```

Additional status:

```text
ready_for_bounded_local_pilot=true
blocked_for_larger_paper_grade_stage=true
```

The project is go for:

- routine safe checks,
- documentation and checker maintenance,
- planning-only reports,
- required LoRA/QLoRA adapter construction and feasibility planning,
- LoRA/QLoRA planning interpretation and risk review,
- risk-assessed bounded local SmolVLA pilot tasks inside budget.

The project is no-go for:

- paper-grade empirical claims,
- real benchmark evaluation that could be mistaken for paper-grade evidence,
- simulator rollouts without passing risk assessment,
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
- bounded cached-feature local pilot extension.
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
- LIBERO/RoboSuite/RoboCasa/dataset download without passing source/size/license/token/disk risk assessment,
- simulator execution without passing readiness risk assessment,
- rollout without passing bounded rollout risk assessment,
- real benchmark evaluation,
- training beyond 300 steps after stable smaller smoke,
- runtime expected over 30 minutes,
- more than 14GB VRAM,
- package/CUDA/PyTorch changes for QLoRA,
- major CUDA/PyTorch changes,
- unplanned large package installs,
- token/secret/login requirements,
- multi-seed experiments,
- paper-level empirical claims,
- external submission/upload/publishing.
