# Autonomous RA-L Topic Ledger

## Cycle 1

Topic: delay-indexed action-chunk deployment

Method: `DICD-VLA`

Status: `KILLED_VALID_PROTOTYPE`

Distinctness:

- core problem: controlled execution delay
- representation: delay-indexed chunk features plus executed-action history
- action-generation mechanism: adapter changes deployed action instead of ranking existing candidates

Forbidden rescues still excluded:

- ECHO candidate ranking or verifier rescue
- PhaseBarrier threshold tuning or rename
- CensorCredit relabeling or hold-strength tuning
- ISAC intervention-set fine-tuning without real paired intervention/correction data

Cycle 1 Stage A kill:

- decision: `SIMPLE_BASELINE_EXPLAINS_METHOD`
- result file: `reports/dicd_vla/stage_a_result.json`
- full DICD: `1 / 10`
- direct chunk-index delay: `2 / 10`
- frozen delay-only baseline: `2 / 10`
- no-history ablation: `1 / 10`
- exceptions: `0`
- mechanism active: `true`

The delay-indexed history-conditioned adapter must not be revived through threshold tuning, longer training, or cosmetic changes. A new cycle must change at least two of the core problem, representation, training signal, objective, action-generation mechanism, and closed-loop intervention.

## Cycle 2

Status: `PENDING_SELECTION`
