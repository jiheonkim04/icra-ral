# PRISM-VLA Autopilot State

## Current State

- Branch: `codex/prism-vla-state1`
- Base: `main` at `cad512e` before PRISM edits
- Milestone: State 0 docs plus State 1 CPU diagnostic
- Heavy training: not allowed
- OpenVLA-OFT: blocked
- Simulator rollouts: not part of this milestone
- Evidence label: exploratory offline proxy

## State 1 Result

- Diagnostic report: `reports/prism_vla_diagnostic_report.md` and `.json` (runtime-generated, gitignored)
- Decision: `continue`
- Training happened: yes, tiny CPU NumPy surrogate training
- Loss computed: yes
- Rollout happened: no
- Model: `tiny_numpy_semantic_action_distribution_policy`
- Dataset: official LIBERO-Para metadata plus local LIBERO HDF5 action chunks
- Selected tasks/paraphrases: `5 / 90`
- Base clean proxy: `0.519538`
- Base paraphrase proxy: `0.439197`
- Simple augmentation paraphrase proxy: `0.440673`
- PRISM paraphrase proxy: `0.440992`
- PRISM vs simple augmentation best robustness delta: `0.013981`
- Clean retained: true
- Counterfactual sensitivity preserved: true

Interpretation: PRISM passes the tiny exploratory proxy gate, mainly through difficulty-weighted robustness/PRIDE and consistency improvements over simple paraphrase augmentation. This is not a real VLA checkpoint result and not rollout evidence.

## Local Assets Observed

- LIBERO source root: `C:/assets/repos/LIBERO`
- LIBERO data root: `C:/assets/data/libero`
- Official LIBERO-Para metadata CSV: `C:/assets/data/libero_para/libero_para_metadata.csv`
- Metadata acquisition risk assessment: green, official GitHub raw CSV, about 708 KB

## Executable Added

Safe runner:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\190_prism_vla_paraphrase_diagnostic.ps1
```

The runner refuses download, GPU, rollout, simulator, runtime-install, heavy-import, and OpenVLA gates. It trains only tiny CPU NumPy surrogate policies.

## Validation Completed

- Targeted tests: `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests\test_prism_vla_paraphrase_diagnostic.py -q` passed, `5 passed`.
- Diagnostic script: `ALLOW_TINY_TRAINING=1` plus `scripts\190_prism_vla_paraphrase_diagnostic.ps1` passed.
- Full pytest: `495 passed`.
- Safe runner: `scripts\40_cursor_safe_local_check.ps1` passed.
- Readiness checks: `scripts\11_check_real_assets.ps1`, `scripts\13_check_smolvla_adapter_smoke.ps1`, and `scripts\17_check_smolvla_runtime_deps.ps1` passed.

## Required Final Report Fields

- final main commit,
- training happened yes/no,
- loss computed yes/no,
- model/dataset used,
- clean metric,
- paraphrase metric,
- simple augmentation metric,
- PRISM metric,
- whether PRISM beats simple baselines,
- continue/kill decision.
