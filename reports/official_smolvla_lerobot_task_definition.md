# Official SmolVLA / LeRobot Baseline Task Definition

Date: 2026-07-09 KST

Branch: `codex/official-smolvla-lerobot-baseline`

## Goal

Determine whether SmolVLA can be used as a reliable official-recipe research backbone on this machine before any new RA-L method work.

This pass is official-recipe-first. It must use official or official-compatible LeRobot/SmolVLA preprocessing, normalization, action conventions, and evaluation stack. It must not revive the archived custom SmolVLA-to-LIBERO 7D adapter route.

## Scope

- Inspect the local repository state, local checkpoint, installed Python environment, LeRobot package, and official documentation.
- Verify whether a local SmolVLA checkpoint can load with LeRobot's official loader and processor factory.
- Run only a bounded mini-reproduction if the preprocessing and action convention are clear.
- Stop before large downloads, full fine-tuning, full benchmarks, simulator rollouts, OpenVLA-OFT, or paper claims.

## Non-Goals

- No new method.
- No Target-Grounded ActionMap, PatchGuard, SafeLoRA, PRISM, ActionMap approximation, or OpenVLA-OFT.
- No use of the archived custom LIBERO 7D adapter route as method evidence.
- No full benchmark or long GPU job.
- No paper-level claim.

## Required Decision

The final decision must be exactly one of:

- `READY_FOR_OFFICIAL_SMOLVLA_MINI_REPRO`
- `NEEDS_OFFICIAL_DATASET_CONVERSION`
- `CHECKPOINT_OR_RECIPE_BLOCKED`
- `NO_OFFICIAL_RECIPE_PATH`
- `TOO_HEAVY_LOCAL`
- `SOURCE_BLOCKED`

## Current Finding

The local checkpoint at `C:\assets\checkpoints\smolvla` is a SmolVLA base-style checkpoint with a 6D state/action schema and SO100 normalizer tensors. LeRobot's official LIBERO path expects 8D state and 7D actions. Therefore, the local checkpoint is suitable for a bounded official SmolVLA base loader/processor smoke, but it is not by itself an official LIBERO baseline.
