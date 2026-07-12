# Governance Migration 20260712

Date: 2026-07-12 KST

Branch: `codex/autonomous-until-paper-governance-v2`

Starting pushed commit: `e24a6a11db49054aaf7a9d6787449f671b5035b3`

## Migration Decision

The previous terminal decision `NO_METHOD_AFTER_3_VALID_CYCLES` is procedurally invalid under the active goal because it retained an older three-cycle maximum and an obsolete no-method terminal state. It is reclassified as:

`EPOCH_1_COMPLETED_PIVOT_REQUIRED`

The campaign continues into Epoch 2.

## Governance Replacements

Active governance now lives in:

`reports/current_research_governance.md`

`AGENTS.md` now points to the active governance file and explicitly deprecates older TCA-first, one-major-milestone, obsolete missing-asset, obsolete global 30-minute, obsolete OpenVLA INT4 block, and fixed three-cycle terminal instructions.

`reports/codex_delegation_manual.md` now permits multi-stage autonomous research inside one Goal execution, including literature, method selection, implementation, prototype, repair or kill, automatic pivot, and scale-up.

## Active State Corrections

The active campaign state is corrected to:

- current epoch: `2`
- current cycle: `0`
- current decision: `EPOCH_1_COMPLETED_PIVOT_REQUIRED`
- branch: `codex/autonomous-until-paper-governance-v2`
- finite global method-cycle maximum: none
- global no-method terminal allowed: false

Allowed final states are exactly:

1. `READY_TO_DRAFT_RAL_PAPER_PACKAGE`
2. `AUTONOMOUS_CAMPAIGN_PAUSED_RESUMABLE`
3. `HARD_EXTERNAL_BLOCKER`
4. `SAFETY_RESOURCE_STOP`

## Validation Added

The migration adds:

- `scripts/check_current_research_governance.py`
- `tests/test_current_research_governance.py`

The validator checks active governance and active state for obsolete finite-cycle/no-method terminal semantics, contradictory active decisions, obsolete one-milestone rules, mandatory TCA-final-method rules, universal OpenVLA-OFT INT4 prohibition, and invalid final states. Historical archived reports are excluded.
