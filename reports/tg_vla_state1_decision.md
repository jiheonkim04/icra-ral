# TG-VLA STATE 1 Decision

Date: 2026-07-09 KST

STATE 1 decision: `HIGH_KILL_RISK_DO_NOT_TRAIN`

Final decision: `KILL_TG_VLA_BASELINE_DOMINATED`

This is a pre-training bounded decision. It is not an empirical claim that a fully implemented TG-VLA adapter was trained and failed.

## Required Final Report Fields

| Field | Value |
| --- | --- |
| final main commit | not merged in this run |
| branch | `codex/tg-vla-smolvla-lora-gate` |
| git status | see final response after validation |
| experiments happened | no |
| training happened | no |
| loss computed | no |
| LoRA/adapter used | no real adapter training; only feasibility checks |
| model used | none for new training; local SmolVLA readiness checked |
| dataset/split used | none for training; local LIBERO HDF5 and LIBERO-Para availability checked |
| GPU/download/OpenVLA-OFT happened | no / no / no |
| VRAM peak | not measured in this run; checker reports RTX 5080 16,303 MiB total and 12,603 MiB free at check |
| runtime | STATE 0-1 feasibility/reporting only |
| standard LoRA metric | not run |
| canonicalization metric | not run |
| TG-VLA metric | not run |
| counterfactual sensitivity | not run |
| clean retention | not run |
| whether TG-VLA beats strong baselines | unknown; not tested |
| final decision | `KILL_TG_VLA_BASELINE_DOMINATED` |
| exact next step if GO_TG_VLA_SCALE_UP | not applicable |

## Why Not READY

Local source/model/hardware feasibility is mostly green:

- local SmolVLA assets are complete,
- SmolVLA runtime dependencies are present,
- local LIBERO HDF5 data is available,
- local LIBERO-Para metadata is available,
- all 10 LIBERO-Para original instructions have local LIBERO-Goal HDF5 matches,
- no download is required for a tiny future smoke.

The blocking issue is the novelty and baseline gate:

- direct grounded 3D point action-head injection already covers the closest "target grounding into action head" idea,
- prior PRISM-VLA was killed because canonicalization-only beat the proposed paraphrase consistency method,
- standard/adaptive LoRA remains a strong explanation for any low-resource adaptation gain,
- off-the-shelf `peft` and `bitsandbytes` are absent in the current environment,
- no leakage-safe TG-VLA target resolver or real SmolVLA action-path adapter runner exists in the repo yet.

## What Would Reopen The Route

The route can be reconsidered only with a predeclared implementation that:

- compares against direct single-point target injection,
- compares against canonicalization-only and standard LoRA,
- derives target/object priors without eval metadata leakage,
- uses a real SmolVLA forward/action-path adapter rather than a NumPy proxy,
- preserves counterfactual target sensitivity.

Until then, STATE 2 training would be a high-risk spend toward another baseline-dominated result.
