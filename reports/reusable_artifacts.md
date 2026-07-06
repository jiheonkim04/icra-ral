# Reusable Artifacts

The RA-L route is killed, but several artifacts remain useful.

## Repository Infrastructure

- local path policy and asset readiness checks
- Windows-safe preflight and safe runner
- compute budget enforcement
- no-large-OpenVLA guardrails
- ignored runtime report convention
- explicit local asset setup documentation

## SmolVLA / LIBERO Readiness

- SmolVLA checkpoint readiness checks
- tokenizer/processor dependency checks
- runtime dependency checks
- LIBERO/RoboSuite path readiness checks
- WSL simulator setup documentation

## Offline Evaluation Tools

- LIBERO HDF5 reader and metadata subset tooling
- counterfactual split construction
- ActionMap vs TCA offline comparison scripts
- fixed-prior target-prior audit logic
- LoRA attribution diagnostics
- multi-seed offline proxy validation utilities

## Rollout And Bridge Diagnostics

- action-interface metadata audit
- HDF5 rollout alignment audit
- matched-init expert replay sanity check
- zero-action and expert replay controls
- online action-generation bridge diagnostic
- online 7D diagnostic head scaffold
- 7D action-quality diagnosis
- bounded 7D action-head redesign gate

## Research Integrity Assets

- decision log
- risk register
- anti-p-hacking policy
- publishability criteria
- final kill/archive package

## Caution

These artifacts should not be reused to claim fixed-prior TCA rollout success. Any future project must keep the same distinction between offline proxy evidence, bridge diagnostics, expert replay, and valid closed-loop method rollout.

