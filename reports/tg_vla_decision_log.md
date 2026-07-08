# TG-VLA Decision Log

Date: 2026-07-09 KST

## 2026-07-09 - Goal Context Loaded

Read the referenced pasted text file:

`C:\Users\jiheo\.codex\attachments\61688635-d4bf-461e-88a2-b78f89f2a2e4\pasted-text-1.txt`

Key instruction: start a new TG-VLA direction, do not revive killed proxy routes, make target/object grounding in the VLA action pathway the novelty, and run bounded STATE 0-1 before any STATE 2 training.

## 2026-07-09 - Git State Checked

- `git status --short --branch`: `## main...origin/main`
- `git branch --show-current`: `main`
- `git log -1 --oneline`: `fcd527f Scout official LIBERO-Safety feasibility`

No reset, stash pop, or discard operation was performed.

## 2026-07-09 - Branch Created

Created and switched to:

`codex/tg-vla-smolvla-lora-gate`

## 2026-07-09 - Official Literature Gate

Primary finding: TG-VLA is plausible only in a narrow gap. ActionMap validates action representation as a strong lever. Direct grounded 3D point action-head injection is the closest novelty threat because it already injects target grounding into the action head. LIBERO-Para supplies a strong paraphrase/object lexical benchmark but is not itself a method. OpenVLA-OFT is too heavy locally and remains a strong nonlocal baseline. SmolVLA is the feasible local backbone.

## 2026-07-09 - Local Source Gate

Local checks found:

- SmolVLA checkpoint path configured and complete,
- SmolVLA load-only and single-sample interface reports already passed,
- runtime dependencies present for SmolVLA loading,
- GPU visible: NVIDIA GeForce RTX 5080, 16,303 MiB total,
- local LIBERO HDF5 demos present,
- local LIBERO-Para metadata present with 4,092 rows,
- all 10 LIBERO-Para original instructions have matching local LIBERO-Goal HDF5 demos.

## 2026-07-09 - Tooling Gate

Current Python environment:

- `peft`: missing,
- `bitsandbytes`: missing,
- `torch`, `lerobot`, `transformers`, `h5py`, `pandas`: present.

Conclusion: off-the-shelf PEFT LoRA/QLoRA is not immediately runnable. A custom small adapter is possible, but a repo-integrated real TG-VLA SmolVLA adapter runner is not present.

## 2026-07-09 - STATE 1 Decision

Decision: `HIGH_KILL_RISK_DO_NOT_TRAIN`

Reason: source/model/hardware are mostly green, but the novelty and baseline gate is not green. Direct 3D target action-head injection, canonicalization-only, and standard/adaptive LoRA are strong enough that naive TG-VLA training would likely produce another baseline-dominated or proxy-only route.

Mapped final run decision: `KILL_TG_VLA_BASELINE_DOMINATED`
