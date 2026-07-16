# CCIF-VLA Preregistration

Date: 2026-07-16 KST

Decision: `CCIF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`

Proposal: `reports/ccif_vla/researcher_proposal.md`

Proposal SHA-256:
`2AFC40F050FD7F0D28507344358CBCB70BF27CC901C57474A501D3EB87E7FAA1`

Reviewer attack: `reports/ccif_vla/reviewer_attack.md`

Researcher rebuttal: `reports/ccif_vla/researcher_rebuttal.md`

Mathematical audit:
`reports/ccif_vla/mathematical_mechanism_audit.md`

## Fixed Claim

CCIF-VLA tests a Base-preserving continuous coarse motor-intent residual
constraint around an already trained continuous SmolVLA chunk.

It is not a generic coarse-to-fine VLA, not a discrete action-token tokenizer,
not official Coarse-to-Control unless official assets are installed and
verified, not standard LoRA, and not a TSC rescue.

## Evidence Partitions

`DISCOVERY / TRAINING`

- fixed legal demonstrations `0..7` for each source task;
- used to fit intent normalization, construct labels, fit trainable CCIF
  components, prior proxy, ablation, and standard LoRA;
- may be used for implementation debugging and small gradient/magnitude audit;
- may not include confirmatory reset identities or outcomes.

`VALIDATION`

- fixed legal demonstrations `8..9` for each source task;
- used for Stage 0 headroom and health gates;
- used for bounded validation search if Stage 0 passes;
- may select exactly one final configuration;
- cannot use confirmatory outcomes.

`CONFIRMATORY TEST`

- no confirmatory task/reset identities, simulator outcomes, rewards, success
  flags, done flags, object poses, future observations, or policy actions may
  be read during Stage 0 or validation search;
- used once only after method, configuration, policy list, ablation, tasks,
  reset identities, metrics, thresholds, and artifacts are frozen.

## Fixed Development Sources

Use these four source task families:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

Discovery/training demonstrations: `0..7`.

Validation demonstrations: `8..9`.

Minimum rows:

- at least `512` discovery windows;
- at least `128` validation windows;
- every task must contribute validation rows;
- no task may contribute more than `40%` of the Stage 0 validation subset.

## Frozen Intent Definition

Use exactly the `m = 31` intent vector from the mathematical audit:

- mean translation: `3`;
- mean rotation: `3`;
- terminal gripper mean over steps `45..49`: `1`;
- cumulative translation waypoints at `[9, 19, 34, 49]`: `12`;
- cumulative rotation waypoints at `[9, 19, 34, 49]`: `12`.

Intent normalization is fitted on discovery/training rows only with
`eps_c = 1e-6`. Any retained component with discovery standard deviation below
`1e-6` is collapsed and stops Stage 0 as
`CCIF_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

The `m = 37` duplicate-terminal draft in the mathematical audit is explicitly
invalid.

## Stage 0 Development Audit

Stage 0 happens before expensive training, validation search, Stage A rollout,
or confirmatory testing.

Required Stage 0 outputs:

- manifest JSON;
- partial JSON with one row per `(policy_or_probe, task, demo, frame)` key;
- result JSON;
- result Markdown;
- validation JSON;
- preflight JSON;
- status JSON;
- heartbeat JSON;
- PID file when detached;
- stdout/stderr logs;
- exit-code file for detached runs;
- action-semantics JSON when policy chunks are materialized.

Stage 0 must report:

- planned and completed row counts;
- exception count;
- duplicate, missing, extra, and split-overlap key counts;
- proposal hash check;
- discovery/validation split overlap;
- no confirmatory reads;
- intent component variance and collapse count;
- task/phase mean intent Huber;
- endpoint-only intent Huber;
- deployment-input intent prediction Huber;
- Base-to-expert residual Huber;
- prior-proxy Huber;
- no-intent ablation Huber;
- CCIF validation Huber;
- CCIF-minus-prior and CCIF-minus-ablation margins;
- action validity;
- identity initialization and disk reload max error;
- action delta statistics;
- residual activation frequency;
- objective term magnitudes;
- gradient norms and frozen-parameter gradient count.

## Stage 0 Pass Gates

To pass to bounded validation search, all must hold:

1. JSON artifacts parse and proposal hash matches.
2. Planned rows equal completed rows and exceptions are zero.
3. Duplicate, missing, extra, and split-overlap key counts are zero.
4. No confirmatory records, reset identities, simulator rewards, success flags,
   done flags, object poses, or future observations are read.
5. Discovery and validation rows have zero overlap.
6. Intent labels are noncollapsed.
7. Deployment-input intent prediction beats task/phase mean by at least `5%`
   relative validation Huber or `0.005` absolute normalized Huber.
8. Endpoint-only intent does not explain the full intended mechanism; if it
   matches CCIF proxy performance within the frozen margin, Stage 0 stops as
   design failure.
9. Base-to-expert residual chunks have positive validation variance in all
   seven action dimensions after valid-step masking.
10. The Coarse-to-Control continuous proxy leaves CCIF residual headroom of at
    least `5%` relative Huber or `0.005` absolute normalized Huber.
11. CCIF beats `ccif_no_coarse_intent_ablation` by at least `5%` relative
    Huber or `0.005` absolute normalized Huber on validation rows.
12. Initialized and disk-reloaded CCIF reproduces Base within `1e-6`.
13. Expected CCIF parameters receive finite nonzero gradients.
14. Frozen Base parameters receive zero gradients.
15. Weighted objective gradient-norm ratio across trainable objective terms is
    at most `100:1`.
16. Postprocessed action validity is preserved.
17. Residual activation is bounded: not all rows and dimensions change
    materially; exact thresholds must be reported in Stage 0 result JSON.

## Stage 0 Stop Classes

Allowed Stage 0 decisions:

- `CCIF_STAGE_0_DATA_OR_SUPERVISION_FAILURE`
- `CCIF_STAGE_0_NO_USABLE_HEADROOM`
- `CCIF_STAGE_0_DESIGN_FAILURE`
- `CCIF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`
- `CCIF_STAGE_0_PASS_TO_BOUNDED_VALIDATION`

Stage 0 is development-only. A Stage 0 stop is not a closed-loop scientific
kill and cannot be rescued by threshold changes after seeing results.

## Bounded Validation Search

If and only if Stage 0 passes, run at most six configurations:

1. CCIF `waypoints=2`, `lambda_c=0.3`, `g_max=0.10`;
2. CCIF `waypoints=4`, `lambda_c=0.3`, `g_max=0.10`;
3. CCIF `waypoints=4`, `lambda_c=1.0`, `g_max=0.10`;
4. `coarse_to_control_continuous_proxy`;
5. `ccif_no_coarse_intent_ablation`;
6. matched `standard_lora`.

One seed per configuration by default. A second seed is allowed only if the
first fixed validation result is genuinely unresolved before confirmatory test.

Validation selection score:

`S = 0.30 * validation_success_or_proxy
   + 0.25 * CCIF_minus_prior_proxy_margin
   + 0.20 * clean_action_retention
   + 0.15 * postprocessed_action_validity
   + 0.10 * intent_overhead_score`.

If closed-loop validation is not feasible, the prototype protocol must freeze
the exact development proxy before execution. Offline action L2 alone cannot
select the configuration.

## First Serious Comparison

Exactly five policies:

1. `smolvla_base`;
2. `coarse_to_control_continuous_proxy`;
3. `ccif_full`;
4. `ccif_no_coarse_intent_ablation`;
5. `standard_lora`.

No additional policy may replace these without a pre-result Reviewer B
amendment.

## Confirmatory Discipline

After validation selection:

- freeze the single selected configuration and checkpoint;
- save every tried configuration and negative result;
- freeze baseline list, ablation, tasks, reset identities, seeds, metrics, and
  thresholds;
- run confirmatory test once;
- never retune CCIF using confirmatory outcomes;
- classify any major redesign after confirmatory outcomes as a new method
  cycle.

## Next Step

Write the executable CCIF prototype protocol before implementation.
