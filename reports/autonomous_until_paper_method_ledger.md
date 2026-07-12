# Autonomous Until Paper Method Ledger

## 2026-07-12 KST Continuity Update

The active governed campaign has completed two valid method cycles:

| Epoch | Cycle | Method | Status | Key evidence |
| --- | ---: | --- | --- | --- |
| 1 | 1 | `DICD-VLA` | `KILLED_VALID_PROTOTYPE` | full `1 / 10`; direct chunk-index delay `2 / 10`; no-history ablation `1 / 10`; zero exceptions |
| 1 | 2 | `FEDO-VLA` | `KILLED_VALID_PROTOTYPE` | faulted full `1 / 10`; static/APEX/no-feedback baselines `2 / 10`; clean frozen `4 / 10`; clean FEDO `0 / 10`; zero exceptions |

Cycle 3 is selection-pending and is the final permitted distinct method cycle. It must not revive DICD, FEDO, ECHO, candidate ranking, adaptive-chunk selection, or residual command compensation through cosmetic changes.

Cycle 3 selected method: `GCAP-VLA`, status `STAGE_A_READY_FINAL_ALLOWED_CYCLE`. It targets controlled visual occlusion through patchwise temporal geometric repair at the camera tensor boundary.

## Historical Closed Methods

These methods are inherited as closed evidence and must not be cosmetically revived.

| Method | Final local status | Key evidence | Reopen rule |
| --- | --- | --- | --- |
| `ECHO-VLA` | `NO_ECHO_HEADROOM_CONFIRMED` | official and structured candidate oracle improvement `0.0` pp; recoverable default failures `0/2`; restoration determinism passed | only with a new candidate generator or representation and a new headroom gate |
| `PhaseBarrier-VLA` | `PHASEBARRIER_COMPONENT_NOT_USEFUL` | full `0/20`, no-phase ablation `9/20`, frozen `8/20`; full changed actions in `20/20` episodes | closed under current formulation |
| `CensorCredit-VLA` | `CENSORCREDIT_NO_VALID_REPAIR` | censored/uncensored labels matched in `24/24` rows; learned heads identical | only as a materially new method with new supervision, not a repair |
| `ISAC-VLA` | `FINAL_METHOD_KILLED_BEFORE_IMPLEMENTATION` | near-overlap with SDP/TORL-VLA/ConRFT; unavailable paired intervention chunks | closed in proposed paired intervention-chunk form |

## Current Campaign

| Epoch | Cycle | Method | Status | Next |
| --- | ---: | --- | --- | --- |
| 1 | 1 | `DICD-VLA` | `SYNTHETIC_MECHANISM_SMOKE_PASSED_REAL_SMOLVLA_SMOKE_PENDING` | run real SmolVLA chunk smoke, then real trace training and Stage A |
