# Reusable Infrastructure

This repository should be treated as a useful research harness, not as evidence that either killed route is RA-L-stable.

## Simulator And Data

- LIBERO/RoboSuite WSL setup.
- OSMesa/offscreen rendering path.
- Local LIBERO HDF5 data readiness checks.
- Counterfactual split and scaled manifest tooling.

## Model And Runtime

- SmolVLA local checkpoint readiness checks.
- SmolVLA tokenizer/processor dependency checks.
- Native SmolVLA CPU load and inference path.
- Explicit no-download/no-GPU/no-OpenVLA guard conventions.

## Diagnostics

- Safe runner: `scripts/40_cursor_safe_local_check.ps1`.
- Preflight and asset readiness checks.
- Native rollout diagnostic scripts.
- Expert replay sanity checks.
- Online 7D action-quality diagnostics.
- Object/safety/wrong-target metric framework.

## Research Operations

- Branch, validate, commit, push, merge pattern.
- Tree check and compute-budget enforcement.
- Decision logs, risk registers, kill reports, and project-state summaries.
- Bounded autopilot pattern with explicit kill gates.

## Reuse Rule

Reuse this infrastructure for new topics only if the new topic is rollout-first, baseline-first, and has kill criteria before implementation.

