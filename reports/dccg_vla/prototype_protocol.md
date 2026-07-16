# DCCG-VLA Executable Prototype Protocol

Date: 2026-07-16 KST

Decision: `DCCG_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING`

Method: `DCCG-VLA`, Demonstration-Calibrated Coherence Guidance for SmolVLA.

Proposal SHA-256:
`AE5DBB13F0B4C19E3DD8BD054433DCFBCC301F4C4293D7B98883D76CA4A1390E`

Frozen inputs:

- proposal: `reports/dccg_vla/researcher_proposal.md`
- Reviewer B attack: `reports/dccg_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/dccg_vla/researcher_rebuttal.md`
- mathematical audit: `reports/dccg_vla/mathematical_mechanism_audit.md`
- preregistration: `reports/dccg_vla/preregistration.md`

No DCCG implementation, validation search, rollout, simulator evaluation, or
confirmatory-test tuning has happened before this protocol.

## Stage 0 Purpose

Stage 0 is a development-only implementation, data, and mechanism audit. It
decides only whether DCCG may proceed to bounded validation search.

It is not a closed-loop scientific result and cannot be interpreted as a paper
claim or confirmatory test.

## Required Command Contract

Implement Stage 0 with:

- helper module: `tca_map/smolvla/dccg_vla.py`;
- runner: `scripts/run_dccg_vla_stage0.py`;
- focused tests: `tests/test_dccg_vla.py`;
- serializer/preflight artifact:
  `reports/dccg_vla/stage_0_serializer_preflight.json`;
- primary result artifact: `reports/dccg_vla/stage_0_result.json`.

The runner must support the repository's WSL execution pattern:

```powershell
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e ./.venv/bin/python scripts/run_dccg_vla_stage0.py
```

The runner may support explicit `--data-root`, `--output-dir`, `--resume`,
`--max-rows`, and `--serializer-preflight` arguments. Defaults must use
validated local SmolVLA/LIBERO paths discovered by existing repository helpers.

## Worker Safety And Resume

Before launching a worker, check existing PID, heartbeat/status, partial,
result, logs, and exit-code files.

- If an existing DCCG worker is alive, monitor it only.
- If a final result already exists, adjudicate that result and refuse duplicate
  execution.
- If a worker died and `stage_0_partial.json` parses, resume only missing row
  keys.
- If heartbeat is stale, verify PID, status, logs, partial JSON parseability,
  and exit-code file before deciding it is dead.

Resume may add only missing manifest keys and may not repeat completed keys.
Duplicate manifest keys, duplicate partial keys, missing keys, extra keys, and
split-overlap keys must all be zero before accepting a final result.

## Required Helper API

The helper module must provide deterministic utilities for:

- protocol constants for `H=50`, `D=7`, proposal hash, smoothing constants,
  clipping constants, policy names, and stop decisions;
- canonical JSON serialization helpers;
- row-key construction and duplicate/missing/extra/split-overlap checks;
- legal split/task/demo/window enumeration;
- action chunk shape and finite checks;
- official action postprocessor validity checks;
- ACG official/proxy asset inspection and mismatch reporting;
- differentiable DCCG feature computation;
- demonstration robust center/scale computation;
- legal deployment-bin construction without privileged inputs;
- coherence energy computation;
- `grad_A E(A,b)` and group clipping;
- exact Base passthrough at `gamma = 0`;
- no-demo-calibration ablation;
- ACG proxy diagnostic;
- action-smoothing simple killer with gripper-event preservation when it is the
  strongest simple baseline;
- normalized and postprocessed action-delta summaries;
- hard gripper transition and reversal diagnostics;
- clean-retention and gate-activation diagnostics;
- Stage 0 decision taxonomy.

The helper must not import simulator environments, read reward/success/done
fields, use object poses, use future observations, or access confirmatory
identities.

## Required Artifacts

Stage 0 writes under `reports/dccg_vla/`:

- `stage_0_preflight.json`;
- `stage_0_manifest.json`;
- `stage_0_partial.json`;
- `stage_0_status.json`;
- `stage_0_heartbeat.json`;
- `stage_0_result.json`;
- `stage_0_result.md`;
- `stage_0_adjudication.md`;
- `stage_0_action_semantics.json`;
- `stage_0_official_prior_asset_check.json`;
- `stage_0_serializer_preflight.json`;
- `stage_0_implementation_blocker.json` on exception;
- `stage_0_pid.txt`;
- `stage_0_exit_code.txt`;
- `stage_0_stdout.log` and `stage_0_stderr.log` when launched detached.

Stage 0 writes caches under `runs/dccg_vla/stage0/` only if needed.

## Data Sources

Stage 0A source smoke:

- `libero_10/task_1`;
- `libero_goal/task_1`;
- demonstrations `0..1`;
- at most `32` windows per demonstration.

Stage 0B development audit:

- discovery tasks:
  - `libero_10/task_1`
  - `libero_10/task_3`
  - `libero_goal/task_1`
  - `libero_goal/task_3`
  - `libero_object/task_1`
  - `libero_spatial/task_1`
- validation tasks:
  - `libero_10/task_5`
  - `libero_goal/task_5`
  - `libero_object/task_3`
  - `libero_spatial/task_3`
- discovery demonstrations `0..29`;
- validation demonstrations `30..39`.

Confirmatory task/reset identities, rewards, success flags, done flags, object
poses, future observations, rollout outcomes, and confirmatory policy actions
are forbidden.

Minimum accepted final Stage 0 manifest:

- at least `384` discovery windows;
- at least `128` validation windows;
- every validation task has rows;
- no validation task fraction exceeds `0.40`;
- duplicate manifest keys `0`;
- duplicate partial keys `0`;
- missing manifest keys `0`;
- extra partial keys `0`;
- split-overlap keys `0`.

## Required Row Key

Every manifest and partial row must include a stable key containing:

`split | task_suite | task_id | demo_id | window_start | bin_key |
policy | config_label | probe_label`

Completed keys may not be repeated during resume.

## Required Preflight

Before model-row work:

1. verify proposal hash equals
   `AE5DBB13F0B4C19E3DD8BD054433DCFBCC301F4C4293D7B98883D76CA4A1390E`;
2. verify required source documents exist;
3. persist official ACG asset/code status and whether policy 2 is official ACG
   or the transparent `acg_official_proxy`;
4. persist official SmolVLA/LIBERO action semantics;
5. verify JSON serialization of manifest rows, action chunks, feature vectors,
   bin statistics, gradients, gate values, booleans, paths, and nested metric
   dictionaries;
6. verify CUDA and official SmolVLA checkpoint availability when Base chunk
   decoding is required;
7. persist preflight failures as implementation blockers without fabricating
   partial rows.

## Required Action Semantics

`stage_0_action_semantics.json` must include:

- model-native action shape;
- postprocessor/unnormalizer class and parameters;
- environment action shape;
- environment action-space low/high if exposed;
- gripper convention;
- finite checks;
- action-space or equivalent official environment validation result for Base;
- the final boolean action-validity definition applied to every policy/probe.

No ad hoc normalized `[-1,1]` validity-only rule is allowed as the hard gate.

## Fixed Policies And Diagnostic Rows

Stage 0 must include rows for:

1. `smolvla_base`;
2. `acg_official_proxy`;
3. `dccg_full`;
4. `dccg_no_demo_calibration_ablation`;
5. `action_smoothing_simple_killer`;
6. `expert_demo_coherence_diagnostic`;
7. `synthetic_jitter_diagnostic`;
8. `synthetic_pause_diagnostic`;
9. `synthetic_gripper_corruption_diagnostic`.

Only the first five are policy comparisons. Diagnostic rows cannot replace ACG
as policy 2 and cannot be reported as inference methods.

## Required Metrics

Each result must report:

- planned and completed row counts;
- exception count and last exception;
- duplicate/missing/extra/split-overlap key counts;
- source hashes and manifest hash;
- policy row counts;
- official/proxy ACG status and mismatch list;
- action shape and finite fraction;
- feature variance and bin counts;
- gate activation fraction;
- coherence energy p50/p95/max by policy and split;
- finite nonzero gradient rate;
- gradient norm by translation, rotation, and gripper group;
- exact Base passthrough max error at `gamma = 0`;
- DCCG action deltas from Base by action group;
- hard gripper transition/reversal/sign-change metrics;
- normalized and postprocessed action validity;
- clean-retention proxy;
- DCCG distinction from ACG, no-demo-calibration, and smoothing;
- Stage 0 stop decision and reason.

## Required Decisions

The runner must emit exactly one final decision:

- `DCCG_STAGE_0_PASS_TO_VALIDATION_SEARCH`;
- `DCCG_STAGE_0_DATA_FAILURE`;
- `DCCG_STAGE_0_NO_HEADROOM`;
- `DCCG_STAGE_0_IMPLEMENTATION_FAILURE`;
- `DCCG_STAGE_0_DESIGN_FAILURE`.

Any implementation/data/no-headroom/design failure is development-only and may
not be reported as a closed-loop scientific result.

## Validation And Commit Gate

Before committing an implementation or result:

- `python -m py_compile tca_map/smolvla/dccg_vla.py scripts/run_dccg_vla_stage0.py`;
- focused tests in `tests/test_dccg_vla.py`;
- current governance tests;
- JSON parse checks for every DCCG artifact;
- duplicate-key and manifest completeness checks;
- `git diff --check`.

Do not launch Stage 0 until implementation, tests, preflight, and worker-safety
checks are complete.
