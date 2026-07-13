# CAVM-VLA Prototype Protocol

Date: 2026-07-13 KST

## Artifacts

Implementation artifacts:

- `tca_map/smolvla/cavm_vla.py`
- `scripts/run_cavm_vla_prototype.py`
- `tests/test_cavm_vla.py`

Generated artifacts:

- `reports/cavm_vla/acquisition_records.jsonl`
- `reports/cavm_vla/acquisition_summary.json`
- `reports/cavm_vla/memory_config.json`
- `reports/cavm_vla/stage_1_result.json`
- `reports/cavm_vla/stage_1_result.md`
- `reports/cavm_vla/stage_2a_partial_result.json`
- `reports/cavm_vla/stage_2a_result.json`
- `reports/cavm_vla/stage_2a_result.md`
- `reports/cavm_vla/stage_2b_partial_result.json`
- `reports/cavm_vla/stage_2b_result.json`
- `reports/cavm_vla/stage_2b_result.md`

## Commands

Stage 0/1 acquisition and calibration:

```bash
/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_cavm_vla_prototype.py --mode acquire-calibrate
```

Stage 2A:

```bash
/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_cavm_vla_prototype.py --mode stage-2a
```

Stage 2B, only if Stage 2A is non-catastrophic:

```bash
/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_cavm_vla_prototype.py --mode stage-2b
```

## Resource Rules

- Use official SmolVLA/LeRobot/LIBERO WSL environment.
- Load one SmolVLA policy instance.
- No OpenVLA-OFT training.
- No full-model fine-tuning.
- No CPU/disk offload.
- Stop on repeated CUDA OOM or resource-safety violation.
- Stage 2 partial files are resumable; relaunch the same command without rerun flags to resume.

## Validation

Before committing each major stage, run:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests/test_cavm_vla.py tests/test_current_research_governance.py tests/test_autonomous_campaign_final_decision.py
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts/check_current_research_governance.py
git diff --check
powershell -ExecutionPolicy Bypass -File scripts\99_tree_check.ps1
```

## Decision Discipline

Do not alter tasks, identities, variants, thresholds, or Stage 0/2 gates after observing any CAVM result.

Do not rescue CAVM if the no-contrast ablation, success-only proxy, or nearest-success replay explains the result.
