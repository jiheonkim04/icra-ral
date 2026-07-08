# TG-VLA Experiment Plan

Date: 2026-07-09 KST

## STATE 0-1 Scope

This run performs research alignment and source/model/hardware feasibility only. It does not execute STATE 2 training.

## Candidate STATE 2 Smoke

Allowed only after a green STATE 1 decision and a predeclared implementation plan.

Smallest real smoke:

- model: local `lerobot/smolvla_base` through LeRobot SmolVLA,
- dataset: local official LIBERO HDF5 demos plus local LIBERO-Para metadata,
- split: deterministic group split, with paraphrase group IDs held out and no original/paraphrase group leakage,
- batch size: 1,
- rank: 4 first,
- max steps: <= 100 for first smoke,
- runtime target: <= 30 minutes for a first real diagnostic, hard stop before 4 hours without user approval,
- no OpenVLA-OFT,
- no full fine-tuning,
- no rollout in first smoke.

## Required Arms

1. Frozen/base SmolVLA action metric.
2. Standard LoRA or action imitation adapter without target grounding.
3. Canonicalization-only baseline with deterministic lexical normalization and no metadata leakage.
4. Simple paraphrase augmentation baseline.
5. TG-VLA target-grounded action-path adapter.

Optional only if cheap:

- oracle target upper bound, explicitly labeled oracle,
- prompt-only target insertion.

## TG-VLA Training Objective

Loss components:

- supervised action imitation loss on clean demonstrations,
- target-preserving paraphrase consistency loss,
- counterfactual target sensitivity loss,
- clean-retention loss,
- optional target classification/contrastive loss only if target labels come from training data or non-leaking instruction/object candidates.

## Metrics

Primary:

- clean action L2 / translation L2 / gripper error,
- held-out paraphrase action metric,
- object lexical variation metric,
- wrong-target proxy rate,
- target consistency,
- counterfactual target sensitivity,
- clean retention.

Engineering:

- peak VRAM,
- runtime,
- trainable parameter count,
- loss curve,
- model/load path,
- dataset/split manifest.

## Continue Criteria

Continue only if TG-VLA:

- beats standard LoRA/action imitation on target/paraphrase/object lexical robustness,
- beats canonicalization-only,
- beats simple paraphrase augmentation,
- preserves counterfactual target sensitivity,
- preserves clean behavior,
- uses a real SmolVLA/adapter path, not only NumPy proxy evidence,
- avoids eval metadata leakage.

## Current Blockers Before STATE 2

- No repo-integrated real TG-VLA SmolVLA action-path adapter runner exists yet.
- `peft` is not installed, so off-the-shelf standard LoRA is unavailable in the current Python environment.
- `bitsandbytes` is not installed, so QLoRA is unavailable.
- A naive language-consistency objective is high risk because prior PRISM-VLA was killed by canonicalization-only and weakened counterfactual sensitivity.
- The closest 2026 anchor already injects grounded 3D target information into the action head; TG-VLA must explicitly beat single-point injection, not merely resemble it.

## No-Claim Boundary

A future STATE 2 smoke would be an engineering diagnostic only. It would not establish RA-L readiness, SOTA, standard success, or rollout success.
