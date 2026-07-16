# MCI-VLA Executable Prototype Protocol

Date: 2026-07-16 KST

Decision: `MCI_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING`

Method: `MCI-VLA`, Multi-Consistency Invariance for Base-preserving SmolVLA.

Proposal SHA-256:
`88CB11CC6236D19BA05602217C65C1819A68BEA53B041E17BA12796403BA0B9A`

Frozen inputs:

- proposal: `reports/mci_vla/researcher_proposal.md`
- Reviewer B attack: `reports/mci_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/mci_vla/researcher_rebuttal.md`
- mathematical audit: `reports/mci_vla/mathematical_mechanism_audit.md`
- preregistration: `reports/mci_vla/preregistration.md`

No MCI implementation, training, validation search, rollout, simulator
evaluation, or confirmatory-test tuning has happened before this protocol.

## Stage 0 Purpose

Stage 0 is a development-only source, data, objective, implementation, and
mechanism audit. It decides only whether MCI may proceed to bounded validation
search.

It is not a closed-loop scientific result and cannot be interpreted as a paper
claim or confirmatory test.

## Required Command Contract

Implement Stage 0 with:

- helper module: `tca_map/smolvla/mci_vla.py`;
- runner: `scripts/run_mci_vla_stage0.py`;
- focused tests: `tests/test_mci_vla.py`;
- serializer/preflight artifact:
  `reports/mci_vla/stage_0_serializer_preflight.json`;
- primary result artifact: `reports/mci_vla/stage_0_result.json`.

The runner must support the repository's WSL execution pattern:

```powershell
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e ./.venv/bin/python scripts/run_mci_vla_stage0.py
```

The runner may support explicit `--data-root`, `--output-dir`, `--resume`,
`--max-rows`, and `--serializer-preflight` arguments. Defaults must use
validated local SmolVLA/LIBERO paths discovered by existing repository helpers.

## Worker Safety And Resume

Before launching a worker, check existing PID, heartbeat/status, partial,
result, logs, manifest, and exit-code files.

- If an existing MCI worker is alive, monitor it only.
- If a final result already exists, adjudicate that result and refuse duplicate
  execution.
- If a worker died and `stage_0_partial.json` parses, resume only missing row
  keys.
- If heartbeat is stale, verify PID, status, logs, partial JSON parseability,
  manifest integrity, and exit-code file before deciding it is dead.

Resume may add only missing manifest keys and may not repeat completed keys.
Duplicate manifest keys, duplicate partial keys, missing keys, extra keys, and
split-overlap keys must all be zero before accepting a final result.

If a Windows Efficiency Mode, VM throttling, or other resource-contention
interval occurs, record it in status/result artifacts. Wall-clock latency,
throughput, utilization, and efficiency measured during that interval may not
be used as final paper evidence. Synchronous offline rows may remain valid only
when no timeout, exception, semantic change, identity change, or duplicate row
occurred.

## Required Helper API

The helper module must provide deterministic utilities for:

- protocol constants for `H=50`, `D=7`, `d_z in {16,32}`, `lambda_c` values,
  Huber beta values, representation variance floor, proposal hash, policy
  names, and stop decisions;
- canonical JSON serialization helpers;
- row-key construction and duplicate/missing/extra/split-overlap checks;
- legal split/task/demo/window enumeration;
- Base chunk, demonstration chunk, visual feature, proprioception, and task
  string shape/finite checks;
- official SmolVLA/LIBERO action semantics and postprocessor validity checks;
- official RoVLA asset/code inspection and transparent-proxy mismatch
  reporting;
- deterministic task-preserving instruction, observation/proprioception, and
  action-evolution transformation construction;
- transformation-pair health diagnostics by family;
- consistency-code, gate, residual, and Base-preserving action application;
- `mci_no_consistency_code_ablation` with matched adapter surface and budget;
- `augmentation_only_lora_killer` with matched legal augmentations and budget;
- legal trivial-baseline diagnostics for task identity, frame/demo phase
  audit-only proxy, action-magnitude statistics, and augmentation family;
- consistency-signal predictability versus strongest trivial baseline;
- representation variance and collapse diagnostics;
- exact Base passthrough and disk-reload diagnostics;
- objective magnitude and gradient-norm diagnostics for `L_code`, `L_act`,
  `L_fit`, `L_keep`, `L_var`, and `L_bound`;
- finite nonzero gradient diagnostics for expected trainable parameters;
- zero-gradient diagnostics for frozen SmolVLA Base parameters;
- gate activation, intervention frequency, clean-retention, and action-delta
  summaries by translation, rotation, and gripper groups;
- action-validity metrics under persisted official semantics;
- Stage 0 decision taxonomy.

The helper must not import simulator environments, read reward/success/done
fields, use object poses, use future observations, use future expert actions,
use reset identity as a feature, or access confirmatory identities.

## Required Artifacts

Stage 0 writes under `reports/mci_vla/`:

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

Stage 0 writes caches under `runs/mci_vla/stage0/` only if needed.

## Data Sources

Development tasks:

1. `libero_10/task_5`;
2. `libero_goal/task_5`;
3. `libero_object/task_3`;
4. `libero_spatial/task_3`.

Discovery demonstrations: `0..7`.

Validation demonstrations: `8..9`.

Minimum accepted final Stage 0 manifest:

- at least `512` discovery rows;
- at least `128` validation rows;
- every validation task has rows;
- no validation task fraction exceeds `0.40`;
- duplicate manifest keys `0`;
- duplicate partial keys `0`;
- missing manifest keys `0`;
- extra partial keys `0`;
- split-overlap keys `0`.

Confirmatory task/reset identities, rewards, success flags, done flags, object
poses, future observations, rollout outcomes, and confirmatory policy actions
are forbidden.

## Required Row Key

Every manifest and partial row must include a stable key containing:

`split | task_suite | task_identity | demo_id | window_start |
transform_family | policy | config_label | probe_label`

Completed keys may not be repeated during resume.

## Required Preflight

Before model-row work:

1. verify proposal hash equals
   `88CB11CC6236D19BA05602217C65C1819A68BEA53B041E17BA12796403BA0B9A`;
2. verify required source documents exist;
3. verify local LIBERO/SmolVLA source rows parse and contain only legal
   development rows for MCI;
4. persist official RoVLA asset/code status and whether policy 2 is official
   RoVLA or the transparent `rovla_multiconsistency_proxy`;
5. persist official SmolVLA/LIBERO action semantics;
6. verify JSON serialization of manifest rows, cache paths, action chunks,
   feature vectors, transformation metadata, consistency codes, gate values,
   gradient metrics, booleans, paths, and nested metric dictionaries;
7. verify CUDA and official SmolVLA checkpoint availability only when model
   decoding is required beyond existing caches;
8. persist preflight failures as implementation blockers without fabricating
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
2. `rovla_multiconsistency_proxy`;
3. `mci_full`;
4. `mci_no_consistency_code_ablation`;
5. `augmentation_only_lora_killer`;
6. `transformation_label_health_diagnostic`;
7. `consistency_observability_diagnostic`;
8. `identity_passthrough_reload_diagnostic`;
9. `objective_gradient_scale_diagnostic`.

Only the first five are policy comparisons. Diagnostic rows cannot replace
RoVLA as policy 2 and cannot be reported as inference methods.

## Frozen Stage 0 Constants

Fixed constants:

- action horizon `H = 50`;
- action dimension `D = 7`;
- latent dimension candidates `d_z in {16, 32}`;
- consistency coefficient candidates `lambda_c in {0.25, 0.50, 1.00}`;
- Huber beta values `0.05`;
- representation variance floor `gamma_var = 0.5`;
- identity tolerance `1e-7`;
- weighted objective gradient-norm ratio alert `100.0`;
- intervention fraction acceptable range `[0.02, 0.80]`;
- postprocessed action validity required rate `1.0`;
- no deterministic-action KL.

The emitted action is:

`A = postprocess(B + sigmoid(u_eta(o,p,l,B,z_phi(o,p,l,B))) * Delta * tanh(r_theta(o,p,l,B,z_phi(o,p,l,B))))`.

Identity initialization must make `R = 0`, `A_raw = B`, and `A = postprocess(B)`.

## Required Metrics

Each result must report:

- planned and completed row counts;
- exception count and last exception;
- duplicate/missing/extra/split-overlap key counts;
- source hashes and manifest hash;
- proposal hash match;
- no reward/success/done/object-pose/future-observation/confirmatory reads;
- discovery and validation row counts by task and demo id;
- Base and demonstration action shape, finite fraction, min, max, and
  postprocessor validity;
- visual feature and proprioception shape and finite fraction;
- transformation pair counts by family;
- positive and negative contrast counts by transformation family;
- task and demo coverage by transformation family;
- representation variance and noncollapse checks;
- mask or gate activation fraction by task and transformation family;
- trivial baseline scores for task identity, frame/demo phase audit-only
  proxy, action-magnitude statistics, and augmentation-family identity;
- legal consistency-signal predictor score versus strongest trivial baseline;
- RoVLA official/proxy status and mismatch list;
- RoVLA proxy score and remaining MCI headroom;
- augmentation-only LoRA score and remaining MCI headroom;
- MCI full versus Base, RoVLA proxy, no-code ablation, and augmentation-only
  LoRA;
- identity initialization and disk-reload max absolute error;
- finite nonzero gradients for expected consistency encoder, residual, and gate
  parameters;
- zero gradients for frozen SmolVLA Base parameters;
- objective magnitudes and weighted gradient norms for `L_code`, `L_act`,
  `L_fit`, `L_keep`, `L_var`, and `L_bound`;
- weighted objective gradient-norm ratio;
- action delta summaries by translation, rotation, and gripper groups;
- clean-retention deltas;
- normalized and postprocessed action-validity rate;
- recorded resource-contention intervals, if any.

## Stage 0 Pass Gates

All must pass:

- proposal hash matches
  `88CB11CC6236D19BA05602217C65C1819A68BEA53B041E17BA12796403BA0B9A`;
- no privileged or confirmatory input access;
- manifest and partial row keys are unique and complete;
- split overlap is zero;
- discovery row count is at least `512`;
- validation row count is at least `128`;
- every fixed task contributes validation rows;
- no validation task contributes more than `40%` of validation rows;
- every transformation family has at least `32` validation pairs;
- positive and negative contrast counts are both at least `16`;
- representation standard deviation exceeds the `L_var` floor on at least
  `80%` of latent dimensions;
- consistency signal beats the strongest trivial baseline by at least `0.02`
  normalized validation score;
- Base leaves measurable transformed-pair action/representation headroom;
- RoVLA proxy leaves measurable residual headroom for MCI;
- augmentation-only LoRA does not explain MCI;
- MCI full beats the strongest of RoVLA proxy, no-code ablation, and
  augmentation-only LoRA by at least `0.005` normalized validation mechanism
  proxy;
- exact Base passthrough and disk reload max absolute error are `<= 1e-7`;
- expected trainable parameters receive finite nonzero gradients;
- frozen SmolVLA Base parameters receive no gradients;
- weighted objective gradient-norm ratio is at most `100x`;
- intervention fraction lies in `[0.02, 0.80]`;
- action deltas respect preregistered translation, rotation, and gripper caps;
- postprocessed action validity is `1.0`.

## Required Decisions

The runner must emit exactly one final decision:

- `MCI_STAGE_0_PASS_TO_BOUNDED_VALIDATION`;
- `MCI_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `MCI_STAGE_0_NO_HEADROOM`;
- `MCI_STAGE_0_IMPLEMENTATION_FAILURE`;
- `MCI_STAGE_0_DESIGN_FAILURE`.

Any implementation/data/no-headroom/design failure is development-only and may
not be reported as a closed-loop scientific result.

## Validation And Commit Gate

Before committing an implementation or result:

- `python -m py_compile tca_map/smolvla/mci_vla.py scripts/run_mci_vla_stage0.py`;
- focused tests in `tests/test_mci_vla.py`;
- current governance tests;
- JSON parse checks for every MCI artifact;
- duplicate-key and manifest completeness checks;
- `git diff --check`.

Do not launch Stage 0 until implementation, tests, preflight, and
worker-safety checks are complete.
