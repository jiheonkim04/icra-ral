# HASTE-VLA Preregistration

Date: 2026-07-15 KST

Proposal hash:
`5415BC1533A24EC55CC511DDEB014BB11D9C19F603C59D1F1D3E151E15B930A6`.

Decision: `HASTE_PREREGISTERED_STAGE_0A_ALLOWED`

## Frozen Method

- closest prior: StaKe;
- SmolVLA rank-4 LoRA scaffold;
- zero LoRA B initialization;
- event threshold: gripper-command adjacent difference greater than `1e-8`;
- event horizons: `{20,50}` only;
- hazard coefficients: `{0.25,0.5,1.0}` only;
- relative target: cumulative six-dimensional postprocessed arm actions to the
  first future command event;
- discovery-only coordinate normalization with standard-deviation floor
  `1e-6`;
- displacement coefficient: `1.0`;
- clean-retention coefficient: `1.0`;
- whole ordinary SmolVLA inference path; no auxiliary head at inference;
- no output clipping or intervention.

No event-definition, target, task, source, horizon, architecture, coefficient,
threshold, or failure-rule repair is allowed after execution begins.

## Fixed Development Sources

Tasks:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

Discovery demonstrations: `0..7`.

Validation demonstrations: `8..9`.

Stage 0 manifests every legal frame with at least one observed future interval.
Key:

`(partition, suite, task_identity, source_hash, demo_id, frame_index, H_e)`.

Duplicate, missing, extra, and partition-overlap counts must all be zero.

## Stage 0A Gates

Stage 0A performs source, label, frozen-Base headroom, frozen-feature probe,
identity, and persistence audits only. No adapter optimization or simulator
rollout is allowed.

All hard gates:

1. proposal and source hashes match;
2. manifest and partial JSON parse;
3. all planned keys persist exactly once;
4. split overlap, duplicates, missing keys, and extras are zero;
5. finite images, actions, Base latents, predictions, and targets;
6. at least `128` uncensored and `128` censored discovery rows overall;
7. at least `16` uncensored validation rows per task;
8. at least `5` occupied event-offset quintile bins overall;
9. positive discovery variance in all six displacement coordinates;
10. equal-task sampling limits every task to at most `40%` of uncensored rows;
11. Base event-near arm error is at least `10%` above event-far/censored, or
    Base gripper sign error is at least `0.05` higher;
12. validation linear hazard-probe NLL is at least `2%` below the frozen
    global constant-hazard NLL;
13. validation linear displacement-probe Huber is at least `2%` below the
    frozen global discovery-mean Huber;
14. initialized and disk-reloaded HASTE reproduces Base flow and actions within
    `1e-6`;
15. Base checkpoint hash is unchanged;
16. exceptions are zero.

Decisions:

- all pass: `HASTE_STAGE_0A_PASS_STAGE_0B_ALLOWED`;
- source or label collapse: `HASTE_STAGE_0A_DATA_FAILURE`;
- no event-near Base deficit: `HASTE_STAGE_0A_NO_HEADROOM`;
- legal features cannot predict targets: `HASTE_STAGE_0A_DESIGN_FAILURE`;
- hash, finite, alignment, identity, persistence, or execution defect:
  `HASTE_STAGE_0A_IMPLEMENTATION_FAILURE`.

None is a scientific kill.

## Stage 0B Gates

Stage 0B is permitted only after an unchanged Stage 0A pass. Run exactly `20`
optimizer steps for the StaKe proxy, HASTE, no-hazard, and standard LoRA.

Required:

- objective and gradient audit passes;
- finite nonzero expected gradients;
- checkpoints persist and disk reload;
- all four adapted policies are distinct from Base and each other;
- HASTE beats trivial hazard and displacement validation baselines;
- HASTE event-near action proxy beats Base and no-hazard;
- every decoded action is valid;
- event-far action delta p95 is at most the discovery Base-error p95;
- Base checkpoint unchanged;
- no privileged inference input, simulator rollout, or confirmatory access;
- zero exceptions and exact manifest integrity.

Failure maps to data, no-headroom, design, or implementation failure. No rescue
within HASTE.

## Validation Search

At most six HASTE configurations: `{20,50} x {0.25,0.5,1.0}` for event horizon
and hazard coefficient. At most two lightweight seeds per configuration.

Validation score:

- `35%` event-near action proxy improvement;
- `20%` hazard NLL improvement;
- `15%` displacement Huber improvement;
- `20%` event-far clean retention;
- `5%` action validity;
- `5%` normalized compute overhead.

Any invalid action or failed retention gate makes a configuration ineligible.
Tie break: smaller event horizon, then smaller hazard coefficient. Save every
configuration and negative result; freeze one before rollout.

## First Comparison

Exactly five policies:

1. Base;
2. transparent StaKe proxy;
3. HASTE;
4. HASTE no-hazard;
5. standard LoRA.

Validation resets: `20262201..20262210`.

Stage A resets: `20262211..20262220`.

Stage B adds: `20262221..20262250`.

Use one matched paired manifest. Stage A may kill only for mechanism
invalidity, no headroom, catastrophic degradation, clear prior/ablation
dominance, or exact trivial equivalence. Small differences advance. Stage B
uses at least 40 paired episodes per key policy and one expansion to 80 only if
genuinely unresolved.

## Resource Governance

Before every long WSL command, audit state, newest PID/heartbeat/status/partial/
result/log/exit files, worker liveness, JSON parseability, counts, exceptions,
duplicates, and manifest integrity. Never duplicate a live or completed worker;
resume only missing keys after a dead worker with valid partial data.

All Windows gaming and Efficiency Mode intervals remain recorded. Overlap or
overlap-unknown timing, throughput, wall-clock, latency, and resource metrics
are ineligible for paper evidence. Synchronous timeout-free closed-loop success
rows may survive only after identity, semantics, exception, duplicate, and
manifest checks.
