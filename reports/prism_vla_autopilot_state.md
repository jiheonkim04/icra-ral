# PRISM-VLA Autopilot State

## Current State

- Branch: `codex/prism-vla-state2`
- Base: `main` at `141fcea` before State 2 edits
- Milestone: State 2 held-out paraphrase split and canonicalization dominance gate
- Heavy training: not allowed
- OpenVLA-OFT: blocked
- Simulator rollouts: not part of this milestone
- Evidence label: exploratory offline proxy

## State 2 Result

- Diagnostic report: `reports/prism_vla_diagnostic_report.md` and `.json` (runtime-generated, gitignored)
- Decision: `kill`
- Training happened: yes, tiny CPU NumPy surrogate training only
- Loss computed: yes
- Rollout/GPU/heavy VLA/OpenVLA-OFT happened: no
- Real VLA adapter diagnostic happened: no
- Model: `tiny_numpy_semantic_action_distribution_policy`
- Dataset/split: official LIBERO-Para metadata plus local LIBERO HDF5 action chunks; deterministic held-out paraphrase group split
- Selected tasks/paraphrases: `5 / 90`
- Train/held-out paraphrases: `51 / 39`
- Paraphrase groups train/held-out: `20 / 13`
- Group leakage detected: false
- Held-out object/syntactic paraphrases: `30 / 9`
- Base clean proxy: `0.519538`
- Base held-out paraphrase proxy: `0.457110`
- Base held-out paraphrase drop: `0.062428`
- Simple augmentation held-out paraphrase proxy: `0.417930`
- Canonicalization-only held-out paraphrase proxy: `0.474066`
- Canonicalization-only PRIDE: `46.686731`
- Best PRISM variant: `prism_vla_plus_canonicalization`
- Best PRISM held-out paraphrase proxy: `0.436356`
- Best PRISM PRIDE: `31.985592`
- Best PRISM primary held-out delta vs canonicalization: `-0.030420`
- Best PRISM primary held-out delta vs simple augmentation: `+0.055205`
- Clean retention for best PRISM: `0.870968`
- Counterfactual sensitivity preserved versus canonicalization: false

Interpretation: State 2 kills PRISM-VLA as the current main route. The method beats simple augmentation, but canonicalization-only remains stronger on primary held-out paraphrase and PRIDE metrics. The auxiliary consistency/syntactic gains are not enough because counterfactual sensitivity weakens.

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
