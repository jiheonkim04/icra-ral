# TG-VLA Autopilot State

Date: 2026-07-09 KST

## Branch

`codex/tg-vla-smolvla-lora-gate`

## Initial Git State

- starting branch: `main`
- starting commit: `fcd527f Scout official LIBERO-Safety feasibility`
- initial status: clean
- branch created: yes

## STATE 0

Status: complete.

Outputs:

- `reports/tg_vla_task_definition.md`
- `reports/tg_vla_related_work_matrix.md`
- `reports/tg_vla_experiment_plan.md`
- `reports/tg_vla_kill_criteria.md`
- `reports/tg_vla_autopilot_state.md`
- `reports/tg_vla_risk_register.md`
- `reports/tg_vla_decision_log.md`

## STATE 1

Status: complete.

Outputs:

- `reports/tg_vla_source_feasibility.md`
- `reports/tg_vla_model_feasibility.md`
- `reports/tg_vla_hardware_budget.md`
- `reports/tg_vla_state1_decision.md`

## STATE 2

Status: not run.

Reason: STATE 1 decision is `HIGH_KILL_RISK_DO_NOT_TRAIN`, driven by novelty and baseline risk rather than source/hardware absence.

## Actions Performed

- git status/branch/log checked before edits,
- requested branch created,
- official source pages checked,
- local SmolVLA/LIBERO/LIBERO-Para readiness checked,
- no downloads,
- no installs,
- no GPU jobs,
- no training,
- no rollouts,
- no OpenVLA-OFT execution.

## Current Decision

`KILL_TG_VLA_BASELINE_DOMINATED`

This is a bounded pre-training kill decision for the naive TG-VLA formulation, not an empirical negative result from a real TG-VLA adapter run.
