# COVI-VLA Prototype Protocol

Date: 2026-07-15 KST

Method: `COVI-VLA`

Proposal hash: `338430D2C6CF1D82410C036D79102ED3F38B2367BB35B9AE2811161698A3E621`

Protocol decision: `COVI_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`

Reviewer status: `APPROVE_WITH_FIXED_EMPIRICAL_RISKS`

## Immediate Execution

The next command is:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_covi_vla_stage0.py --mode audit
```

It must write one of:

- `reports/covi_vla/stage_0_result.json`
- `reports/covi_vla/implementation_blocker.json`

The runner may also write a Markdown rendering, feature cache, checkpoint, and
logs, but the JSON result or blocker is the authoritative next scientific
artifact.

## Scope

Stage 0 is executable development evidence. It must load the official local
SmolVLA checkpoint, decode real official LIBERO development images, exercise
the measured visual-token hook, train the fixed lightweight predictor, save
and reload it, and measure representation and policy consequences.

Stage 0 must not:

- decode or inspect reserved confirmatory-test records;
- launch closed-loop rollout;
- run the six-configuration search;
- select an architecture, coefficient, seed, or occlusion severity;
- use clean views, masks, task/reset identity, reward, success, simulator pose,
  or segmentation at COVI inference;
- treat the local VIM proxy as official VIM;
- claim synthetic Stage 0 masks validate physical scene-induced occlusion.

## Runtime Inputs

Frozen paths:

- checkpoint: `C:\assets\checkpoints\smolvla_libero`
- VLM: `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`
- dataset: `C:\assets\datasets\lerobot_libero`
- split manifest: `reports/official_smolvla_split_manifest.json`
- stable Base artifact: `reports/official_smolvla_stable_prediction_artifact.json`

Runtime must set the Hugging Face, transformers, and datasets offline flags.
Missing local assets produce `implementation_blocker.json`; they do not justify
a method kill.

## Authoritative Result Schema

`stage_0_result.json` must include:

- method, proposal hash, command, seed, git commit, and package versions;
- exact checkpoint, model, dataset, and manifest hashes or identities;
- decoded record counts and partition-overlap proof;
- test-record decode count fixed at `0`;
- exact raw, prepared, visual-token, prefix, state, and action shapes;
- source-gate manifest and forbidden-source audit;
- occluder construction, coverage, variance, and duplicate diagnostics;
- target-feature variance and normalization diagnostics;
- all Stage 0 comparator metrics;
- episode-cluster bootstrap interval and record/episode counts;
- normalization sensitivity;
- loss magnitudes and objective-specific gradient norms;
- model parameter counts and frozen-Base update count;
- checkpoint path, SHA256, and reload equality;
- Base, COVI, ablation, and oracle action diagnostics;
- residual norm, gate values, changed tokens, activation context;
- clean retention, output validity, latency, CUDA memory, and exceptions;
- false-positive risk, false-negative risk, ruling confidence, and exact
  evidence required for permanent kill;
- one of the preregistered Stage 0 decisions and the exact next command.

`implementation_blocker.json` must include the failing prerequisite, observed
and expected state, attempted command, stack trace or exception, whether a
bounded implementation repair is possible, and the exact resume command.

## False-Negative Safeguard

Before a non-GO, the runner and adjudication must answer:

1. What is the strongest fair reading of the frozen narrow COVI claim?
2. Is the issue fatal, robust empirical failure, unresolved power, or
   implementation/data failure?
3. How many independent episodes and records support the ruling?
4. What is the episode-cluster bootstrap interval?
5. Does the ruling survive raw, normalized, and train-z-scored targets?
6. Does the interval exclude a `0.02` practically useful advantage?
7. Is COVI worse, or merely not yet proven better?

Only `FATAL_PREIMPLEMENTATION` or `ROBUST_EMPIRICAL_DESIGN_FAILURE` can archive
COVI before rollout. `COVI_STAGE_0_UNDERPOWERED_ONE_CHECK_ALLOWED` triggers the
single fixed check in the preregistration. `IMPLEMENTATION_OR_DATA_FAILURE`
triggers bounded repair and rerun without changing the method.

## Stage 0 Pass Continuation

If Stage 0 returns `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`:

1. run the frozen six-configuration development/validation search;
2. save every attempted configuration and negative result;
3. select one checkpoint with the preregistered validation score;
4. freeze checkpoint, five-policy list, physical occlusion condition, paired
   task/reset manifest, metrics, and thresholds;
5. run the paper-oriented five-policy Stage A.

No Cycle 15 candidate generation is allowed while COVI remains active.

## Long-Running Execution

If Stage 0 exceeds the interactive command window, launch it detached and save:

- PID;
- heartbeat/status JSON;
- stdout and stderr logs;
- partial result;
- exact resume command.

Resume only missing feature-cache or action-smoke keys. Do not recompute a
completed metric under a different seed or normalization.

## Current Next Action

Implement `tca_map/smolvla/covi_vla.py` and
`scripts/run_covi_vla_stage0.py`, add executable tests, and run the frozen
Stage 0 command. No additional planning or protocol milestone is allowed
before implementation begins.

