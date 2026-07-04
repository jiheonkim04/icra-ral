# Prompt-Format Diagnostic

This report documents the bounded prompt-format diagnostic after adapter-strategy and action-scale diagnostics.

Planning command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\86_plan_prompt_format_diagnostic.ps1
```

Bounded runner command:

```powershell
$env:ALLOW_PROMPT_FORMAT_DIAGNOSTIC="1"; powershell -ExecutionPolicy Bypass -File scripts\87_bounded_prompt_format_diagnostic.ps1; Remove-Item Env:\ALLOW_PROMPT_FORMAT_DIAGNOSTIC -ErrorAction SilentlyContinue
```

The planner is read-only. The runner is bounded to one task, at most 10 steps per prompt variant, CPU execution, no downloads, no installs, no training, no GPU job, no OpenVLA-OFT, no multi-seed evaluation, no benchmark claim, and no paper-grade claim.

## Current Local Result

Latest bounded runner result: `passed` as diagnostic execution only.

The runner executed one `libero_10` task for up to 10 steps under:

- action adapter strategy: `policy_6d_delta_pose_plus_gripper_zero_hold`,
- action scale: `1.0`,
- prompt strategies:
  - `stem_spaces`,
  - `bddl_language`,
  - `bddl_language_period`.

Observed prompts:

- `stem_spaces`: `KITCHEN SCENE3 turn on the stove and put the moka pot on it`,
- `bddl_language`: `turn on the stove and put the moka pot on it`,
- `bddl_language_period`: `turn on the stove and put the moka pot on it.`

Observed result:

- variants completed: 3,
- wrapper/execution passed for all variants,
- diagnostic success rate: 0.0 for all variants,
- reward sum: 0.0 for all variants,
- prompt changes produced different continuous action previews,
- rollout scaling ready: false,
- benchmark claim: false,
- paper-grade claim: false.

Interpretation: prompt-format wiring is working and the BDDL `(:language ...)` prompt is cleaner than the previous stem-derived prompt. However, changing only the prompt format did not produce reward or task success on the selected diagnostic task. The next safe rung is a bounded camera-source or state-sufficiency diagnostic. This remains diagnostic/local-pilot evidence only.
