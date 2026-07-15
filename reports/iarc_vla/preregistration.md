# IARC-VLA Preregistration

Date: 2026-07-15 KST

Decision: `IARC_PREREGISTRATION_FROZEN`

Proposal hash:
`A1B0CF8BCBCF6A88F27B31EF5E38BAF408A3E62BB34206A1AC9F051EA6B57408`.

## Frozen Claim

IARC tests whether an asymmetric projected SGD update during VLA clean
refinement preserves a paired perturbation-replay action objective and improves
the closed-loop robustness/clean tradeoff over:

- unchanged SmolVLA;
- a transparent STRONG-VLA proxy;
- unprojected joint replay;
- matched standard clean-only LoRA.

The contribution is a VLA robustness cross-paper synthesis. It is not a generic
optimizer, LoRA, continual-learning, or augmentation-conflict claim.

## Authoritative Documents

- candidate generation:
  `reports/epoch_4_cycle_16_candidate_generation.md`
- proposal: `reports/iarc_vla/researcher_proposal.md`
- Reviewer B attack: `reports/iarc_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/iarc_vla/researcher_rebuttal.md`
- mathematical audit:
  `reports/iarc_vla/mathematical_mechanism_audit.md`
- resource exclusion registry:
  `reports/resource_contention_intervals.json`

The rebuttal and mathematical audit repair the proposal's invalid
raw-gradient-plus-AdamW guarantee by freezing explicit Stage II SGD with zero
momentum and zero weight decay.

## Evidence Partitions

Offline source manifest:
`reports/official_smolvla_split_manifest.json`.

Frozen source counts:

- train/discovery: `1200` rows;
- validation: `400` rows;
- confirmatory test: `1200` rows.

Stage 0 row selection:

1. sort each split by `task_index`, `episode_index`, `frame_index`, then
   `sample_id`;
2. within each task, rank rows by absolute distance of `normalized_phase` from
   `0.5`, then the same stable ordering;
3. training rank `0` per task -> `40` micro-fit rows;
4. training rank `1` per task -> `40` independent conflict-audit rows;
5. validation rank `0` per task -> `40` validation diagnostic rows.

The fixed one-check, if authorized, uses training rank `2` per task and no
other rows.

The runner may read confirmatory manifest identities to prove partition
separation, but may decode zero confirmatory observations/actions, generate
zero confirmatory perturbations, and compute zero confirmatory policy outputs.

## Frozen Model And Low-Compute Path

- Base checkpoint: `C:\assets\checkpoints\smolvla_libero`;
- VLM dependency:
  `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`;
- dataset: `C:\assets\datasets\lerobot_libero`;
- policy: official LeRobot SmolVLA loader and processor;
- PEFT: official SmolVLA `wrap_with_peft` path;
- rank: `4`;
- target modules: official wrapper resolution, recorded and frozen on first
  valid load;
- Base parameters frozen;
- physical/logical batch size: `1`;
- no full-model fine-tuning;
- no rank sweep.

## Frozen Perturbations

Exactly four train/development families:

- Gaussian RGB noise, sigma `{0.02,0.05,0.10}`;
- cardinal image translation, `{4,8,16}` pixels, edge padding;
- exact instruction repetition, `{1,2,3}` extra copies;
- exact fixed context prefix, `{1,2,3}` copies.

All formulas, strings, seeding, camera behavior, and semantics checks are frozen
in the rebuttal and mathematical audit. No family or severity substitution is
allowed after outcomes.

## Stage 0A: Pure And Real-Batch Mechanism Audit

Command:

`wsl -d Ubuntu-22.04 bash -lc "cd /mnt/c/Users/jiheo/tca_map && /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_iarc_vla_stage0.py --mode audit"`

Fixed micro fit:

- `20` AdamW steps;
- learning rate `1e-4`;
- seed `1601`;
- `40` eligible fit rows, deterministic cycling/order;
- no validation selection.

Real gradient audit:

- `40` independent task-balanced pairs;
- shared flow noise `[1,50,32]` and time `[1]` per clean/perturbed pair;
- float32 gradient vectors over resolved rank-4 LoRA parameters;
- robust squared-norm floor `1e-12`;
- exact projected SGD direction;
- no optimizer update during audit pairs.

Required direct-pass gates:

- all preflight assets exist and CUDA is used;
- exact zero-effect adapter identity error `<= 1e-6`;
- only LoRA parameters are trainable;
- finite nonzero micro-fit gradients;
- fixed-subset loss decreases;
- Base weights do not change;
- checkpoint saves, hashes, reloads, and matches within `1e-6`;
- perturbation and partition health pass;
- shared flow/noise/time/action/state hashes pass `40 / 40`;
- at least `4 / 40` conflicts below cosine `-0.01`;
- at least two families activate;
- every conflict row satisfies the projection tolerance;
- every agreeing row is unchanged;
- IARC differs from clean and joint updates on all conflicts;
- all diagnostic actions are finite and range-valid;
- confirmatory decode/action count is zero;
- peak CUDA allocation is below `15.5 GiB`.

The tiny realized-step robust-loss check is diagnostic. Quantization or
finite-difference resolution may classify it unresolved, but it cannot override
an exact dot-product constraint pass or create a scientific kill by itself.

## Stage 0A Decisions

Allowed decisions:

- `IARC_STAGE_0A_PASS_HEADROOM_PENDING`
- `IARC_STAGE_0A_UNDERPOWERED_ONE_CHECK_ALLOWED`
- `IARC_DATA_OR_SUPERVISION_FAILURE`
- `IARC_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`
- `IARC_LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT`
- `IARC_DESIGN_FAILURE_NONACTING_MECHANISM`

`IARC_STAGE_0A_UNDERPOWERED_ONE_CHECK_ALLOWED` requires `1-3` conflicts or only
one activating family with otherwise healthy data, gradients, and
implementation. The only allowed command then is:

`... python scripts/run_iarc_vla_stage0.py --mode one-check`

The one-check uses training rank `2`, exactly `20` additional micro Stage I
steps, the same seed, learning rate, perturbations, thresholds, and rank. No
second check.

Zero conflicts do not automatically justify a scientific kill. The runner must
classify Stage I acquisition, adapter capacity, perturbation health, confidence,
and independence under the false-negative safeguard.

## Stage 0B: Base Closed-Loop Headroom

Stage 0B is forbidden until Stage 0A passes.

Primary frozen manifest:

- suites: `libero_spatial`, `libero_goal`;
- task IDs per suite: `[0,2,4,6,8]`;
- reset identity: `20261601`;
- ten task/reset pairs;
- clean and assigned middle-severity perturbed condition;
- `20` planned Base episodes;
- synchronous one-environment execution;
- no timeout, exception, duplicate, off-manifest row, or action modification.

Pass:

- clean-minus-perturbed success `>= 0.10`; or
- at least two clean-success/perturbed-failure flips.

If unresolved, one expansion uses the same tasks and reset `20261602`, adding
`20` episodes. Aggregate pass requires success delta `>= 0.10` or at least four
failure flips across twenty pairs. No second expansion.

Only a narrow paired interval excluding `0.10` useful degradation can establish
`IARC_NO_HEADROOM`. Wide/mixed evidence remains unresolved and is not a method
kill. Zero clean successes is unscoreable headroom.

Stage 0B result rows are development-only and may not tune family, severity,
task, threshold, or policy design.

## Full Training And Validation Search

Forbidden until Stage 0A and Stage 0B pass.

Stage I:

- two seeds `{1601,1602}`;
- `60` AdamW steps at `1e-4`;
- frozen text-then-visual three-severity curriculum.

Stage II:

- branch each seed into learning rates `{5e-5,1e-4,2e-4}`;
- `40` zero-momentum, zero-decay SGD steps;
- total trials `6`;
- save all outcomes;
- select one learning rate by mean validation score over both seeds;
- final designated seed fixed to `1601`.

No architecture, rank, module, stage length, optimizer, family, severity,
threshold, or seed sweep.

The validation score and eligibility gates are frozen in the mathematical
audit. Test data and confirmatory rollout identities cannot enter selection.

## First Serious Comparison

Exactly five policies:

1. `smolvla_base`
2. `strong_vla_transparent_proxy`
3. `iarc_vla_full`
4. `iarc_unprojected_joint_replay_ablation`
5. `standard_lora_clean_only`

No sixth policy is authorized before Stage A. Standard LoRA is supporting
evidence, not the contribution.

Reserved rollout reset identities:

- Stage A: `20261611`, `20261612`;
- Stage B: `20261613`, `20261614`;
- optional one expansion: `20261615`, `20261616`.

The task allocation and exact manifest must be frozen after validation
selection and before any Stage A outcome. No task cherry-picking.

## Result Classification

Use the active governance labels and the method-specific labels above.

Scientific paper viability requires Ours to beat Base, transparent STRONG,
joint replay, and standard LoRA where relevant, retain clean behavior, preserve
action validity, and show the intended projection mechanism. Unknown
performance is not a rejection reason.

Confirmatory outcomes cannot retune this method. A redesign after test is a new
method cycle.

## Resource And Durable Execution Rules

Before every long WSL launch:

1. read campaign state;
2. inspect newest PID, heartbeat/status, partial, result, logs, and exit code;
3. check Linux worker liveness;
4. parse partial JSON;
5. check completed/planned counts and exceptions.

Never duplicate a live or completed worker. Resume only missing
`(policy, suite, task_id, reset_identity)` keys after a dead worker and valid
partial result.

Record every Windows Efficiency Mode/resource-contention interval. Timing,
throughput, wall-clock, and resource-utilization evidence with unknown or
positive overlap is excluded from final paper evidence. Task-success rows
require synchronous execution, zero exceptions/timeouts, unchanged action
semantics and identities, and duplicate/manifest audit.

## Current Boundary

Only Stage 0A implementation, unit tests, and the frozen `--mode audit` command
are authorized. Do not run validation search, full training, headroom rollout,
Stage A, Stage B, or confirmatory decoding before their predecessor gates pass.

