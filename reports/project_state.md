# Project State

Date: 2026-07-10 KST

Target branch: `main`

Implementation branch: `codex/audit-smolvla-lora-regen-drift`

Current decision: `PROTOCOL_DRIFT_FOUND`

## Current Route

The valid route remains official SmolVLA/LeRobot reproduction first. No closed-loop rollout is safe from the current evidence.

This audit diagnosed the old-vs-regenerated LoRA metric drift after commit `15649d6`. It did not retrain any seed, install simulator dependencies, run rollout, download assets, run OpenVLA-OFT, revive FCAR, design a new method, relax the frozen `0.002` tolerance, or overwrite historical metrics.

## Locked Inputs

- model: `lerobot/smolvla_libero`
  - revision: `31d453f7edd78c839a8bbc39744a292686daf0de`
  - local path: `C:\assets\checkpoints\smolvla_libero`
- dataset: `lerobot/libero`
  - revision: `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
  - local path: `C:\assets\datasets\lerobot_libero`
- split manifest: unchanged across `5d48b1e` and `15649d6`
- metric protocol: unchanged across `5d48b1e` and `15649d6`

## Checkpoint Status

All three regenerated persisted adapter bundles remain complete and checksum verified:

| Seed | Checkpoint path | Status |
| ---: | --- | --- |
| `11` | `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_11` | `CHECKPOINT_COMPLETE_VERIFIED` |
| `22` | `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_22` | `CHECKPOINT_COMPLETE_VERIFIED` |
| `33` | `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_33` | `CHECKPOINT_COMPLETE_VERIFIED` |

## Drift Audit Result

Frame/label/protocol alignment:

- test frame IDs: identical
- task and episode IDs: identical
- ground-truth actions: identical
- split membership: identical
- frozen/base predictions: identical
- metric protocol file: identical
- static-alpha grid: identical, validation-only selection

Repeated disk evaluation:

| Seed | max action L2 repeat diff | rank4 metric diff | static metric diff | selected alpha identical |
| ---: | ---: | ---: | ---: | --- |
| `11` | `0.0` | `0.0` | `0.0` | `True` |
| `22` | `0.0` | `0.0` | `0.0` | `True` |
| `33` | `0.0` | `0.0` | `0.0` | `True` |

The current persisted checkpoints are internally stable under disk re-evaluation. The drift is not evaluation nondeterminism.

However, the saved regenerated artifact metrics do not exactly match the fixed-seed disk re-evaluation metrics. This shows that the evaluation RNG state was also an unpinned part of the protocol identity.

## Root Cause

The old `5d48b1e` run evaluated the trained in-memory policy and did not persist/reload adapter weights. The regenerated `15649d6` run assigns the PEFT wrapper return, saves adapter bundles, reloads them with `PeftModel.from_pretrained`, and evaluates that disk identity. The audit's fixed-seed disk re-evaluation is repeatable but differs from the saved regenerated artifact metrics, so evaluation RNG state must also be pinned in any future protocol.

Historical adapter weights, complete RNG state, and exact training sample order were not preserved, so the historical learned policy identity cannot be reconstructed exactly. The regenerated persisted checkpoints must not be described as identical historical models.

## Canonical Baseline Status

Current regenerated checkpoints are **not accepted as canonical** in this audit because real LoRA prediction-protocol differences were found. They can only become canonical after the PEFT in-memory vs persisted-reload difference and the evaluation RNG-state policy are fixed or explicitly adjudicated as a new re-baselining decision that preserves old metrics as historical.

## Key Reports

- `reports/official_smolvla_lora_drift_audit.md`
- `reports/official_smolvla_old_vs_regen_config_diff.md`
- `reports/official_smolvla_artifact_alignment_audit.md`
- `reports/official_smolvla_eval_determinism_check.md`
- `reports/official_smolvla_training_determinism_status.md`
- `reports/official_smolvla_canonical_checkpoint_proposal.md`
- `reports/official_smolvla_lora_drift_decision.md`

## Exact Next Step

Fix or explicitly adjudicate the PEFT in-memory versus persisted-reload protocol difference and evaluation RNG-state policy before canonicalizing or rolling out.

## 2026-07-10 Canonicalization Update

Current decision: `NEEDS_WSL_OR_LINUX_OFFICIAL_ROLLOUT`

Canonical persisted-checkpoint offline evaluation passed with intermediate decision `CANONICAL_BASELINES_READY_FOR_ROLLOUT`. The run evaluated frozen base plus rank-4 LoRA seeds 11/22/33 on the fixed val/test manifest under action-generation eval seeds `[101, 202, 303, 404, 505]`; it did not train, regenerate checkpoints, download dependencies, run rollout, revive FCAR, or use the old custom LIBERO_7D route.

Native Windows rollout remains blocked because `hf-libero`, `libero`, and `robosuite` are not installed in the active env. The next step is WSL/Linux official LeRobot LIBERO smoke using the canonical artifacts.

## 2026-07-10 WSL Official Rollout Pilot

Current decision: `OFFICIAL_ROLLOUT_BASELINE_READY`

The official WSL/LeRobot/LIBERO path is now working. Smoke completed `4/4` episodes and the bounded pilot completed `48/48` episodes across frozen base plus rank-4 LoRA seeds 11/22/33. All policy audits showed parameters and inputs on `cuda:0`; no CPU fallback, schema/action mismatch, or old custom `LIBERO_7D` route was used.

Pilot overall success:

- `frozen_base`: `75.0%`
- `rank4_lora_seed_11`: `83.3%`
- `rank4_lora_seed_22`: `66.7%`
- `rank4_lora_seed_33`: `75.0%`

Lower offline action L2 did not predict higher closed-loop success in the pilot. This result is enough to unlock larger official baseline rollout/failure mining, but not enough for method selection or best-seed selection.
