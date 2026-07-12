# Autonomous RA-L Campaign State

Date: 2026-07-12 KST

Target terminal state: `PAPER_READY_EXPERIMENTAL_PACKAGE`

Governance correction applied:

- maximum distinct method cycles: `3`
- maximum total GPU time: `24 h`
- maximum wall time per method cycle: `12 h`
- maximum single uncheckpointed command: `4 h`
- no routine user approvals for bounded local research actions

Current cycle: `1`

Current method: `DICD-VLA`

Current branch: `codex/auto-method-20260712-01-dicd-vla`

Prompt branch alias to create/preserve after commit: `codex/ral-cycle-01-dicd-vla`

Current stage: `cycle_1_valid_kill_archived_cycle_2_pending`

Cycle 1 Stage A result:

- result JSON: `reports/dicd_vla/stage_a_result.json`
- result markdown: `reports/dicd_vla/stage_a_result.md`
- completed episodes: `50 / 50`
- exceptions: `0`
- elapsed rollout time: `5637.278 s`
- final decision: `SIMPLE_BASELINE_EXPLAINS_METHOD`

Summary:

- frozen SmolVLA clean: `5 / 10`, task-balanced rate `0.50`
- frozen SmolVLA delay: `2 / 10`, task-balanced rate `0.20`
- direct chunk-index delay: `2 / 10`, task-balanced rate `0.20`
- DICD no-history ablation: `1 / 10`, task-balanced rate `0.10`
- DICD full: `1 / 10`, task-balanced rate `0.10`

Adjudication:

The Stage A runner compiled and `tests/test_dicd_vla.py` passed before launch. The first Stage A launch was stopped after about one hour because it had no episode-level checkpointing and had not written `stage_a_result.json`; this was an infrastructure/resumability stop before any scientific result existed.

The checkpointed Stage A launch completed the preregistered 50 episodes with no exceptions. The full method changed actions, but it did not improve closed-loop success: the direct chunk-index delay baseline exceeded the full method, and the no-history ablation matched it. DICD-VLA is therefore a valid Cycle 1 kill, not a candidate for rescue or repeat.

Next action: start a genuinely distinct Cycle 2 method family under the remaining governance budget.
