# Autonomous Until Paper State

## 2026-07-12 KST Continuity Update

The active governed campaign state is now tracked in `reports/autonomous_ral_campaign_state.json`.

Cycle 1 `DICD-VLA` is closed with valid kill `SIMPLE_BASELINE_EXPLAINS_METHOD`.

Cycle 2 `FEDO-VLA` is closed with valid kill `CLEAN_RETENTION_FAILURE`: Stage A completed `70 / 70` episodes with zero exceptions; faulted full FEDO reached `1 / 10`, while static inverse gain, APEX-style feedback proxy, and no-feedback ablation each reached `2 / 10`; clean frozen SmolVLA reached `4 / 10`, while clean FEDO reached `0 / 10`.

Current decision: `CYCLE_2_KILLED_PIVOT_TO_CYCLE_3`.

Next automatic stage: start Cycle 3, the final permitted distinct method cycle.

Date: 2026-07-12 KST

Campaign branch: `codex/autonomous-until-ral-evidence-ready`

Current method branch: `codex/auto-method-20260712-01-dicd-vla`

Base research branch: `codex/censorcredit-one-repair-and-final-method`

Base commit: `06cf915aefa57eb0c86160fb991a763b3ed323b2`

Normal success state: `READY_TO_DRAFT_RAL_PAPER_PACKAGE`

Current decision: `DICD_REAL_TRACE_TRAINING_PASSED_STAGE_A_PENDING`

## Current Stage

- epoch: `1`
- cycle: `1`
- topic: `delay_indexed_action_chunk_deployment`
- stage: `cycle_1_stage_a_rollout_pending`
- active method: `DICD-VLA`
- proposal hash: `B3D53F728974517A21DD91E45444C0611137AF1B10E15E46298F43FF5D150CC1`

Completed startup stages:

- verified current branch, commit, main commit, and recent git log;
- inspected `project_state.md`, `next_actions.md`, and `decision_log.md`;
- inspected recent ECHO, PhaseBarrier, CensorCredit, ISAC, OpenVLA-OFT, and official SmolVLA reports;
- created the campaign branch from the latest research commit.
- generated exactly three epoch-1 candidates and selected `DICD-VLA`.
- created the child method branch;
- froze researcher proposal, reviewer attack, rebuttal, preregistration, and prototype protocol.
- implemented the DICD core adapter and prototype smoke runner;
- passed unit tests and synthetic mechanism smoke;
- wrote `reports/dicd_vla/mechanism_smoke_result.json`;
- persisted checkpoint `reports/dicd_vla/checkpoints/dicd_synthetic_smoke.pt`.
- passed real SmolVLA action-chunk smoke with official `SmolVLAPolicy`, CUDA tensors, raw chunk `[1, 50, 7]`, finite postprocessed `[8, 7]` chunks, and real delay contrast.
- passed real trace training on identity `20260711` with `312` full examples and `312` no-history examples.
- saved real full checkpoint `reports/dicd_vla/checkpoints/dicd_real_full.pt`.
- saved real no-history checkpoint `reports/dicd_vla/checkpoints/dicd_real_no_history.pt`.

## Evidence Boundary

The prior decision `NO_VALID_CENSORCREDIT_REPAIR_FINAL_METHOD_KILLED` is narrow. It closes CensorCredit repair and the specific ISAC-VLA proposal, but it does not terminate the autonomous RA-L campaign.

Epoch 1 must not revive ECHO candidate ranking, PhaseBarrier action projection, CensorCredit temporal hold heads, or ISAC intervention-chunk fine-tuning in their previous forms.

Cycle 1 proceeds with `DICD-VLA`, a delay-indexed action-chunk deployment method. The next automatic stage is Stage A closed-loop rollout.

## Resume

`cd /d C:\Users\jiheo\tca_map && git switch codex/auto-method-20260712-01-dicd-vla && type reports\autonomous_until_paper_state.json`
