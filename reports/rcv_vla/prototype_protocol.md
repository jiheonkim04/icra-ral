# RCV-VLA Prototype Protocol

Date: 2026-07-13 KST

Decision: `IMPLEMENTATION_PROTOCOL_READY`

## Artifacts

Expected implementation files:

- `tca_map/smolvla/rcv_vla.py`
- `scripts/run_rcv_vla_prototype.py`
- `tests/test_rcv_vla.py`

Expected generated artifacts:

- `reports/rcv_vla/stage_0_result.json`
- `reports/rcv_vla/stage_0_result.md`
- `reports/rcv_vla/acquisition_records.jsonl`
- `reports/rcv_vla/verifier_full.json`
- `reports/rcv_vla/verifier_no_context.json`
- `reports/rcv_vla/stage_1_train_result.json`
- `reports/rcv_vla/stage_2a_result.json`
- `reports/rcv_vla/stage_2a_result.md`

Stage 2B artifacts are generated only if Stage 2A is not catastrophically killed.

## Runner Modes

`stage-0`

- runs the diagnostic manifest;
- evaluates queued and stateless variants;
- records queued-vs-fresh disagreement.

`acquire-train`

- runs normal queued frozen SmolVLA on acquisition identities;
- records train/calibration rows;
- computes `tau_train`;
- trains and saves full and no-context logistic verifier checkpoints;
- selects `theta_train` from calibration identities.

`stage-2a`

- runs exactly the five preregistered variants on the matched Stage 2A manifest;
- writes JSON/Markdown summaries and paired comparisons.

`stage-2b`

- same policies and metrics on the Stage 2B manifest;
- allowed only after a non-catastrophic Stage 2A result.

## Policy-Call Accounting

`queued_frozen_smolvla` uses normal chunked execution.

`sv_deviation_proxy` calls the frozen policy every step to compute the fresh reference, then replans by threshold. Its high call count is expected.

`stateless_first_action` calls the frozen policy every step and executes the first action.

`rcv_full` and `rcv_no_context_ablation` must not call the frozen policy every step. They may call it only at planning boundaries and when the verifier triggers a replan.

## Result Adjudication

Every result file must report:

- exact manifest;
- proposal hash;
- checkpoint paths and hashes;
- `tau_train`;
- `theta_train`;
- total rows;
- exception count;
- per-variant and per-task success;
- task-balanced success;
- replan rate;
- heavy-policy calls per step;
- latency;
- peak CUDA memory.

Before any terminal decision, run:

`C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests/test_rcv_vla.py tests/test_current_research_governance.py tests/test_autonomous_campaign_final_decision.py`

`C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts/check_current_research_governance.py`
