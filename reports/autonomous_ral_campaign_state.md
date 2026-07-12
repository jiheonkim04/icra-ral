# Autonomous RA-L Campaign State

Date: 2026-07-12 KST

Target terminal state: `PAPER_READY_EXPERIMENTAL_PACKAGE`

Governance correction applied:

- maximum distinct method cycles: `3`
- maximum total GPU time: `24 h`
- maximum wall time per method cycle: `12 h`
- maximum single uncheckpointed command: `4 h`
- no routine user approvals for bounded local research actions

Current cycle: `2`

Current method: `FEDO-VLA`

Current branch: `codex/ral-cycle-02-fedo-vla`

Prompt branch alias to create/preserve after commit: `codex/ral-cycle-02-fedo-vla`

Current stage: `cycle_2_real_trace_training_passed_stage_a_ready`

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

Cycle 2 current method:

- topic: feedback execution-disturbance observer
- method: `FEDO-VLA`
- proposal: `reports/fedo_vla/researcher_proposal.md`
- proposal hash: `F9098041A471641E1506BC9AEE2E2CDE205170BAEF2F9E281077724BC239D073`
- reviewer attack: `reports/fedo_vla/reviewer_attack.md`
- preregistration: `reports/fedo_vla/preregistration.md`
- synthetic result: `SYNTHETIC_MECHANISM_PASS`
- real trace training result: `REAL_TRACE_TRAIN_PASS`
- full checkpoint: `reports/fedo_vla/checkpoints/fedo_full.pt`
- no-feedback checkpoint: `reports/fedo_vla/checkpoints/fedo_no_feedback.pt`

Next command:

```powershell
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_fedo_vla_prototype.py --mode stage-a
```

FEDO Stage A must follow the preregistered APEX-style feedback, static inverse-gain, and no-feedback ablation kill gates.
