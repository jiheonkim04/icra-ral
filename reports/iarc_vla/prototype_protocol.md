# IARC-VLA Executable Prototype Protocol

Date: 2026-07-15 KST

Decision: `IARC_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0A_PENDING`

Proposal hash:
`A1B0CF8BCBCF6A88F27B31EF5E38BAF408A3E62BB34206A1AC9F051EA6B57408`.

## Authorized Implementation Scope

Create only:

- `tca_map/smolvla/iarc_vla.py` for pure perturbation, partition, gradient
  vectorization, projection, classification, and summary helpers;
- `scripts/run_iarc_vla_stage0.py` for the real official SmolVLA Stage 0A
  audit;
- `tests/test_iarc_vla.py` for deterministic pure and artifact contract tests;
- Stage 0A artifacts under `reports/iarc_vla/` and checkpoint/cache artifacts
  under `runs/iarc_vla/`.

Do not refactor unrelated modules, alter prior method results, add another
objective or policy, or implement full training before Stage 0A adjudication.

## Runtime Preflight

Before launch verify:

- current campaign stage is
  `epoch_4_cycle_16_iarc_stage_0a_implementation_pending`;
- proposal hash file matches the constant in code;
- no active Linux research worker exists;
- no completed IARC Stage 0A result already exists;
- local checkpoint, VLM, dataset, split manifest, and stable Base artifact
  exist;
- CUDA is available on `NVIDIA GeForce RTX 5080` or the recorded local device;
- Hugging Face, transformers, and datasets offline flags are set;
- no rollout or OpenVLA gate is enabled;
- resource-contention registry is readable.

Missing assets or CUDA produce `implementation_blocker.json`; do not fall back
to CPU or download anything.

## Pure Helper Contract

Tests must cover:

- stable task-balanced row partitioning;
- zero split overlap;
- exact perturbation family/severity assignment;
- deterministic image/text perturbations;
- action-target and nonallowlisted-input preservation;
- agreeing, conflicting, partly conflicting, orthogonal, below-floor, and
  nonfinite projection cases;
- exact flatten/unflatten order and shapes;
- no projection on agreeing gradients;
- projection tolerance on conflict gradients;
- positive scale invariance of the reference gradient;
- conflict classification and one-check boundary;
- result classification cannot turn data/capacity/implementation failure into a
  scientific kill.

## Real Stage 0A Command

`wsl -d Ubuntu-22.04 bash -lc "cd /mnt/c/Users/jiheo/tca_map && /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_iarc_vla_stage0.py --mode audit"`

Hard limits:

- wall-clock cap: `4` hours;
- peak CUDA allocation cap: `15.5 GiB`;
- no downloads or installs;
- no simulator rollout;
- no validation search;
- no confirmatory decode;
- one rank-4 adapter;
- `20` micro-fit steps;
- `40` conflict-audit pairs;
- `40` validation diagnostic rows.

If the command is expected to exceed the interactive window, launch one
detached worker and persist:

- wrapper and child PID;
- heartbeat and status JSON;
- partial result;
- stdout/stderr logs;
- exit code;
- exact resume command;
- completed pair keys.

Resume only missing pair keys. Do not repeat completed gradient or action rows.

## Required Result Files

- `reports/iarc_vla/stage_0a_result.json`
- `reports/iarc_vla/stage_0a_result.md`
- `reports/iarc_vla/gradient_audit.json`
- `reports/iarc_vla/perturbation_manifest.json`
- `reports/iarc_vla/parameter_manifest.json`
- `reports/iarc_vla/checkpoint_manifest.json`
- `reports/iarc_vla/implementation_blocker.json` only on a blocker/failure
- checkpoint bundle under `runs/iarc_vla/stage0a/`

## Authoritative Stage 0A Schema

The result JSON must include:

- method, proposal hash, command, seed, git commit, and package versions;
- checkpoint, VLM, dataset, split-manifest, and resource-registry identities;
- preflight paths, CUDA device, dtype, autocast, memory, and elapsed diagnostics;
- source row counts, selection ranks, task/phase counts, and all overlap counts;
- confirmatory observations decoded and actions computed, both `0`;
- exact raw, processed, token, state, action, noise, time, and gradient shapes;
- perturbation counts, severities, deltas, hashes, duplicates, and target checks;
- resolved LoRA parameter names, shapes, dtypes, numel, module groups, and Base
  frozen hash;
- zero-effect identity error;
- micro-fit loss curve and gradient norms;
- checkpoint file hashes, disk-reload status, and reload output error;
- all `40` clean/robust loss and gradient records;
- shared-draw hash checks;
- conflict count/rate, cosine distribution, family/task/phase breakdown;
- projection coefficients, constraint residuals, update norms, and module
  contributions;
- discovery gradient-scale ratio and frozen joint-ablation `beta`;
- tiny-step diagnostic with its numerical-resolution status;
- validation clean/perturbed loss, action deltas, translation/rotation/gripper
  deltas, range validity, and clean retention;
- false-positive/false-negative risks, record independence, confidence, and
  exact evidence for a permanent stop;
- one preregistered decision and exact next command.

Timing/resource fields must carry a `paper_evidence_eligible` flag derived from
the resource-contention registry. Unknown overlap means `false`.

## Implementation Blocker Schema

On failure, record:

- failing prerequisite or invariant;
- expected and observed values;
- command and stack trace;
- decoded split counts at failure;
- whether any parameter update occurred;
- whether a bounded implementation repair is possible without method change;
- exact resume command;
- correct classification.

An implementation repair may fix loading, batching, dtype, hashing, checkpoint,
or vector reconstruction. It may not change method, rank, target modules,
perturbations, rows, thresholds, steps, seed, or policy list.

## Stage 0A Adjudication

Run the pure tests, then the frozen audit once. Read the existing result if it
completed. Do not rerun a valid result.

If direct pass, update state to Stage 0B headroom implementation pending. If the
one-check decision occurs, run only the preregistered `--mode one-check`. For
every failure, preserve partial artifacts and classify under the false-negative
safeguard.

## Automatic Continuation

Do not stop after documentation or implementation. After Stage 0A:

- pass -> implement/freeze the 20-episode Base headroom manifest and run it;
- unresolved -> run only the one allowed fixed check;
- valid data/capacity/implementation failure -> adjudicate honestly and move to
  the next method cycle without rescue;
- Stage 0B pass -> execute the six-trial validation-only training search;
- selected checkpoint -> freeze the five-policy Stage A manifest and continue.

Every milestone is validated, committed, and pushed. No routine WSL,
monitoring, resume, validation, commit, or push question is sent to the user.

