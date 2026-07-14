# FANG-VLA Prototype Protocol

Date: 2026-07-14 KST

Proposal hash: `6837DBA2A1307F7C9938FA9F5463ED483907AF3C168F1C0514F6E281804E859B`

## Planned Artifacts

Design and audit:

- `reports/fang_vla/researcher_proposal.md`
- `reports/fang_vla/proposal_hash.txt`
- `reports/fang_vla/reviewer_attack.md`
- `reports/fang_vla/researcher_rebuttal.md`
- `reports/fang_vla/mathematical_mechanism_audit.md`
- `reports/fang_vla/preregistration.md`
- `reports/fang_vla/prototype_protocol.md`
- `reports/fang_vla/development_audit.json`
- `reports/fang_vla/development_audit.md`

Implementation:

- `tca_map/smolvla/fang_vla.py`
- `scripts/run_fang_vla_development.py`
- `scripts/run_fang_vla_prototype.py`
- `tests/test_fang_vla.py`

Training/checkpoints:

- `reports/fang_vla/checkpoints/`
- `reports/fang_vla/validation_search.json`
- `reports/fang_vla/validation_search.md`
- `reports/fang_vla/validation_search_uncalibrated_gate_failure.json`
- `reports/fang_vla/validation_search_uncalibrated_gate_failure.md`
- `reports/fang_vla/selected_config.json`

Rollout results, only after audit and validation pass:

- `reports/fang_vla/stage_a_result.json`
- `reports/fang_vla/stage_a_result.md`
- `reports/fang_vla/stage_b_result.json`
- `reports/fang_vla/stage_b_result.md`

## Development Audit Command

Planned Windows command:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_fang_vla_development.py --mode audit
```

The audit uses existing non-confirmatory trace records only and must not launch simulator rollout.

## Validation Search Command

Only if audit passes:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_fang_vla_development.py --mode train-validate
```

This command trains only lightweight heads. It must save every tried configuration, checkpoint, validation metrics, and the selected frozen configuration.

## Closed-Loop Commands

Only after audit, validation, checkpoint reload, gradient checks, action validity checks, and clean-retention checks pass.

Stage A:

```bash
/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_fang_vla_prototype.py --mode stage-a
```

Stage B:

```bash
/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_fang_vla_prototype.py --mode stage-b
```

## Resource Rules

- Use existing SmolVLA/LeRobot/LIBERO assets.
- No OpenVLA-OFT training during first prototype.
- No full-model SmolVLA fine-tuning.
- No CPU or disk offload.
- One SmolVLA instance loaded for rollout.
- FANG head training must run on local CPU or single GPU with tiny memory footprint.
- Stop on repeated CUDA OOM, rising RAM above active safety limits, or invalid action outputs.

## Required Checks

Before committing implementation:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests\test_fang_vla.py tests\test_current_research_governance.py tests\test_autonomous_campaign_final_decision.py
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\check_current_research_governance.py
git diff --check
powershell -ExecutionPolicy Bypass -File scripts\99_tree_check.ps1
```

## Decision Discipline

Do not modify FANG tasks, identity partitions, five-policy list, validation search budget, validation score, GO criteria, or kill criteria after seeing confirmatory outcomes.

Do not rescue FANG if the AFIL proxy, no-failure ablation, nearest-success replay, or Base explains the result.
