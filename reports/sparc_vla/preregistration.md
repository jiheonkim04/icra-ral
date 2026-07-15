# SPARC-VLA Preregistration

Date: 2026-07-15 KST

Proposal hash:
`CC2F9ACCE2A26EC438C58F2854ADC95134354C245CAD8ED961D29A895DBC697D`.

Decision: `SPARC_PREREGISTRATION_FROZEN_STAGE_0A_PENDING`

## Frozen Method

SPARC fits no target failure activation into its operator. It combines the
target-success conceptor with an equal-task-weight covariance aggregate of four
source-task failure distributions, then gates the frozen SmolVLA post-residual
action-expert state.

No CAVM action memory, FANG action field, PCAV candidate generator, FAMR
checkpoint, adapted Base generator, target-failure fit, privileged inference
input, or test tuning is permitted.

## Frozen Tasks

Targets:

1. `libero_10/task_8`;
2. `libero_10/task_6`;
3. `libero_goal/task_8`.

Sources:

1. `libero_10/task_0`;
2. `libero_goal/task_0`;
3. `libero_object/task_2`;
4. `libero_spatial/task_8`.

Prohibited problem identities remain `libero_10/task_4` and
`libero_spatial/task_4`.

## Frozen Partitions

- discovery seeds: `20261901..20261912`;
- validation seeds: `20261921..20261926`;
- confirmatory seeds: `20261941..20261960`.

Partition overlap is zero over `(suite,task_id,reset_seed)`. Historical
five-reset task-selection rows are discovery context only and provide no
activation to the fit.

## Stage 0A: Math And Hook Smoke

Stage 0A runs before labeled simulator rollout.

It implements and tests:

- equal-episode covariance;
- conceptor construction;
- Boolean AND-NOT;
- covariance and mean-conceptor aggregation diagnostics;
- global and per-step gate serialization;
- manifest-key and canonical tensor hashing;
- post-residual SmolVLA capture/gate adapter;
- exact unconfigured and `beta=0` Base behavior;
- bounded nonzero configured behavior on official discovery observations.

The official model smoke uses the frozen checkpoint and legal local
demonstration observations only. It does not use outcome labels or simulator
test identities.

Pass requires:

- all pure-math tests pass;
- captured tensor shape `[1,50,720]`;
- exactly ten ordered denoising captures per action generation;
- capture-only, removed-hook, unconfigured, and beta-zero identity error `0`;
- finite acting result at a synthetic PSD operator and `beta=0.1`;
- configured checkpoint reload error `<=1e-6`;
- finite valid postprocessed action;
- no installed package edit;
- no confirmatory record read.

Failure is `IMPLEMENTATION_FAILURE`. One bounded repair of wiring, serialization,
or tensor orientation is allowed before labeled collection because it does not
change the scientific method. Every failed attempt is preserved.

## Stage 0B: Frozen Discovery Collection

Stage 0B is authorized only after Stage 0A passes and its implementation is
committed.

Run all `84` Base episodes: seven tasks times twelve discovery reset seeds.
Do not stop early after class quotas. Capture candidate residual sites
`{0,5,11,14}` and full action/replan metadata.

Durable files:

- PID;
- heartbeat JSON;
- status JSON;
- atomic partial JSON;
- episode manifest JSON;
- activation manifest JSON;
- final result JSON and Markdown;
- stdout/stderr logs;
- exit-code file.

Resume only missing episode keys. A partial episode is recomputed under the
same key; completed rows are not repeated.

Data pass requires per source task at least three successes and three failures,
per target at least three successes, phase coverage, finite variance, and exact
manifest/hash integrity. A missing class after all 84 episodes is
`DATA_FAILURE`; no task replacement or extra reset is allowed.

## Stage 0C: Geometry And Headroom

After data pass:

1. apply the fixed 16-replan equal-episode sampler;
2. select layer and aperture without target failures;
3. construct SPARC, all source COAST candidates, key ablation, and diagnostics;
4. run numerical, LOO, prefix, aggregation, and random-null audits;
5. run offline Base/Ours action consequence and clean-retention smoke;
6. adjudicate exact headroom and safety gates.

Stage 0C pass requires every numerical, stability, headroom, and action-safety
gate in the rebuttal and mathematical audit.

Classifications are `IMPLEMENTATION_FAILURE`, `UNDERPOWERED_OR_UNRESOLVED`,
`NO_HEADROOM`, or `DESIGN_FAILURE` as defined. No failed Stage 0C result is a
closed-loop scientific kill.

## Standard LoRA Control

Train only after SPARC Stage 0C passes. Use the exact rank-4, seed-1919,
2,000-step filtered-BC specification in the rebuttal. Training data is exactly
the capped successful target Base observation/action pairs used for target
success conceptors.

The final-step adapter is persisted and reloaded. Underpowered or invalid LoRA
is not counted as a SPARC win and blocks the five-policy scientific comparison
until honestly classified.

## Validation Search

Exactly six SPARC configurations: global/per-step crossed with beta
`{0.1,0.3,0.5}`. Layer, aperture, source pool, aggregation, and thresholds are
already fixed.

Validation target cases are all three targets at seeds `20261921..20261926`,
eighteen paired cases per policy/config. Clean validation uses:

- `libero_object/task_0`;
- `libero_object/task_4`;
- `libero_spatial/task_0`;

at the same validation seed range. Clean tasks do not select target success;
they only measure retention and action disruption.

For config `c`, define:

`score_c = target_success_c
           - 0.5 max(0, clean_success_base-clean_success_c)
           - 0.1 invalid_action_rate_c
           - 0.05 global_destructive_rate_c
           - 0.01 normalized_compute_overhead_c`.

`global_destructive_rate` is the fraction of episodes violating any frozen
component delta threshold. A config with any exception, invalid action,
duplicate, clean catastrophic loss greater than `0.20`, or zero action delta
is ineligible.

Ties within `1e-12` prefer higher clean success, lower beta, global strategy,
lower uncontaminated compute, then lexicographic config id.

Freeze one selected config and all policy artifacts. Confirmatory data cannot
change it.

## First Serious Five-Policy Comparison

1. `smolvla_base`;
2. `coast_single_source_transfer_proxy`;
3. `sparc_full`;
4. `sparc_source_failure_only`;
5. `standard_lora_target_success`.

Stage A target manifest contains ten paired cases per policy:

- `libero_10/task_8`, seeds `20261941..20261944`;
- `libero_10/task_6`, seeds `20261941..20261943`;
- `libero_goal/task_8`, seeds `20261941..20261943`.

Stage A permanently kills only for mechanism invalidity, no headroom that
escaped Stage 0, catastrophic degradation, clear prior/ablation/control
dominance, or exact trivial equivalence after adequate evidence. Small
differences advance.

Stage B contains forty total paired cases per policy, including Stage A:

- `libero_10/task_8`, seeds `20261941..20261954` (`14`);
- `libero_10/task_6`, seeds `20261941..20261953` (`13`);
- `libero_goal/task_8`, seeds `20261941..20261953` (`13`).

One unresolved expansion uses all sixty frozen target cases, twenty per task.
No identities are added to reach eighty. If sixty remains unresolved, report
`UNDERPOWERED_OR_UNRESOLVED` and do not retune or add resets in this cycle.

## Statistical Reporting

Report per policy and task:

- successes and denominators;
- paired wins/losses/ties;
- paired delta and failure-rate reduction;
- deterministic paired bootstrap 95% interval, seed `1919`, `10,000` draws;
- exact McNemar/binomial discordant-pair p-value when applicable;
- action and activation mechanism summaries;
- clean retention;
- latency, throughput, and resource evidence only outside quarantined
  intervals.

No p-value alone defines the decision. Effect direction, confidence interval,
prior/ablation/control comparisons, mechanism, and clean retention all enter.

## Paper-Candidate Gate

SPARC must:

- beat Base;
- beat the COAST proxy on the no-target-failure-fit claim axis;
- beat source-failure-only;
- not be explained by matched filtered-BC LoRA;
- retain clean behavior;
- satisfy target-success/failure geometry predictions;
- preserve novelty after current-paper recheck.

Only then run Quantized OpenVLA-OFT INT4 plus Ours and one held-out-task or
second-benchmark condition.

## No-Rerun And No-Rescue Rules

- do not change tasks, reset identities, outcomes, labels, source pool, hook,
  cap, weights, layer candidates, apertures, beta values, thresholds, LoRA
  schedule, policy list, or statistics after results;
- do not use confirmatory outcomes for source, config, or checkpoint selection;
- do not add target failures to the SPARC fit;
- do not repeat completed rows;
- do not rescue PCAV, FAMR, or a valid SPARC kill;
- a major redesign is a new method cycle.

## Resource Contention

The two recorded Windows gaming/Efficiency Mode intervals remain quarantined.
Any overlapping or unknown-overlap latency, throughput, wall-clock efficiency,
or utilization is excluded. Synchronous, unchanged-semantics, exception-free,
timeout-free, duplicate-free success rows may remain valid after manifest
audit.
