# SafeLoRA-VLA STATE 1 Decision

Date: 2026-07-08

Final decision: `NO_CLEAR_LORA_PATH`

## Required Final Report Fields

| Field | Value |
| --- | --- |
| final main commit | `7b1b0f7 Add SafeManip feasibility scout` |
| branch | `codex/safelora-vla-state0-state2` |
| experiments happened | no |
| training happened | no |
| download/GPU/OpenVLA-OFT happened | no / no / no |
| selected official benchmark | none for immediate training; LIBERO-Safety is the best future candidate |
| selected model | none for training; SmolVLA is the preferred future candidate |
| exact LoRA/QLoRA feasibility | LoRA plausible as custom adapter work, but not green on an official safety source; QLoRA blocked locally by missing `peft` and `bitsandbytes` |
| expected memory | local SmolVLA checker estimates 12000 MB load plus 2048 MB headroom on RTX 5080 16 GB |
| expected runtime | no training estimated as executable; official SmolVLA docs cite about 4 hours on one A100 for 20k ordinary fine-tuning steps; OpenVLA-OFT cites 1-2 days on 8 A100/H100 80 GB GPUs |
| expected dataset size | LIBERO-Safety dataset about 19.1 GB page total / about 24.6 GB API used storage; assets about 10.7 GB; released pi0.5 model about 12.4 GB |
| required baselines | base, standard imitation/L1 LoRA, safety-only/stop-on-risk, generic DPO/ORPO LoRA, SafeLoRA property-conditioned LoRA |
| strongest baseline likely to kill the idea | generic DPO/ORPO LoRA on the same official pair set, followed by safety-only/stop-on-risk |
| estimated kill probability | `0.70` before blocker resolution |
| base metric | not run |
| standard LoRA metric | not run |
| safety-only/stop metric | not run |
| generic DPO/ORPO metric | not run |
| SafeLoRA metric | not run |
| SafeLoRA beats simple/research baselines | unknown; not tested |
| continue/kill/source-block decision | `NO_CLEAR_LORA_PATH` |
| exact next state | blocker resolution and rerun STATE 1; do not execute STATE 2 |

## Why This Is Not READY

The route fails the hard source plus LoRA gate. At least one official source
must support temporal/process safety metrics, a bounded official subset/sample,
and a realistic LoRA/QLoRA training path. The inspected candidates do not yet
satisfy all three:

- SafeManip has strong temporal properties but is too heavy locally.
- LIBERO-Safety is public and relevant, but property-conditioned unsafe labels
  and an official bounded LoRA path are not clear.
- ForesightSafety-VLA has strong metrics on paper, but no official source path
  was found.
- Local standard LIBERO is disallowed as evidence for this route.

## Exact Next Command

None.

The instruction says to provide an exact next command only if the decision is
`READY_FOR_USER_APPROVAL_TO_RUN_LORA`. This decision is not ready.
