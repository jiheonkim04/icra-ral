# SafeLoRA-VLA Decision Log

Date: 2026-07-08

## Initial Repo Gate

- `git status`: clean on `main`
- `git branch --show-current`: `main`
- `git log -1 --oneline`: `7b1b0f7 Add SafeManip feasibility scout`
- Created branch: `codex/safelora-vla-state0-state2`

## Literature And Source Alignment

Decision: SafeLoRA is conceptually different from pure benchmark diagnostics
only if property-conditioned LoRA changes the trained update and utility
retention is optimized jointly.

Reason: SafeManip, LIBERO-Safety, ForesightSafety-VLA, and SafeVLA-Bench expose
success-safety gaps. They do not by themselves provide a low-compute
property-conditioned LoRA method.

## Source Feasibility

Decision: no official source is green for immediate LoRA training.

Reason:

- SafeManip has the strongest temporal property semantics but is too heavy
  locally.
- LIBERO-Safety is public and relevant, but its accessible training corpus and
  code do not provide a clear official property-conditioned safe/unsafe LoRA
  path.
- ForesightSafety-VLA has strong CC/RET metrics on paper but no official
  source path found.
- Local standard LIBERO is not acceptable RA-L evidence for this route.

## Local LoRA Feasibility

Decision: local SmolVLA assets are ready for guarded engineering checks, but
not for official SafeLoRA training.

Reason:

- The SmolVLA readiness guard is green for local assets and prior interface
  smoke status.
- QLoRA is blocked by missing `peft` and `bitsandbytes`.
- Existing local LoRA runners are proxy/tiny NumPy or cached-feature scaffolds,
  not official benchmark SafeLoRA.

## Final Gate Decision

`NO_CLEAR_LORA_PATH`

Do not execute STATE 2. Do not run LoRA/QLoRA training until a later gate shows
an official property-label path, bounded data path, and real adapter
implementation path.
