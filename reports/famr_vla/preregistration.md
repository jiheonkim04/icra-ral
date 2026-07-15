# FAMR-VLA Preregistration

Date: 2026-07-15 KST

Decision: `FAMR_PREREGISTRATION_FROZEN_STAGE_0A_PENDING`

## Frozen Claim

FAMR tests whether a new-task finetuned VLA checkpoint can be merged with its
generalist initialization using groupwise action-function responses, improving
held-out LIBERO-90 task success over Base and scalar RETAIN while retaining the
original 40-task policy better than standard finetuning and a target-only
functional merge.

The claim is specific to action-function-constrained model retention. LoRA is
the low-compute task-vector realization, not the scientific method.

## Authoritative Documents

- prior map: `reports/epoch_4_cycle_17_prior_mechanism_map.md`
- candidate selection: `reports/epoch_4_cycle_17_candidate_generation.md`
- proposal: `reports/famr_vla/researcher_proposal.md`
- reviewer attack: `reports/famr_vla/reviewer_attack.md`
- rebuttal: `reports/famr_vla/researcher_rebuttal.md`
- mathematical audit: `reports/famr_vla/mathematical_mechanism_audit.md`
- executable protocol: `reports/famr_vla/prototype_protocol.md`

Any contradiction is resolved in favor of this preregistration, then the
mathematical audit, then the executable protocol.

## Evidence Partitions

Target tasks:

1. `KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf`;
2. `LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray`;
3. `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy`.

Target demonstration episodes per task:

- discovery/training: `0-34`;
- validation: `35-44`;
- confirmatory offline test: `45-49`.

Target rollout reset identities:

- discovery/headroom: `20261701-20261704`;
- validation selection: `20261711-20261713`;
- Stage A confirmatory: `20261721-20261724`;
- Stage B confirmatory: `20261731-20261744`.

Original-task retention uses task indices `0`, `16`, and `29` from the official
40-task benchmark. Reset identities use the same numeric partitions and a
separate suite key, so every complete key remains unique.

Original-task offline discovery/validation/test rows come from the existing
stable artifact partitions. Its `1,200` test rows are not decoded before
configuration freeze.

## Frozen Base And Endpoint

- Base checkpoint: `/mnt/c/assets/checkpoints/smolvla_libero`;
- model: official local `lerobot/smolvla_libero` files and processors;
- trainable method endpoint: PEFT LoRA only;
- rank: `4`;
- alpha: `8`;
- dropout: `0`;
- bias: `none`;
- target expression:
  `model.vlm_with_expert.lm_expert.*.(q|v)_proj`, `model.state_proj`,
  `model.action_in_proj`, `model.action_out_proj`,
  `model.action_time_mlp_in`, `model.action_time_mlp_out`;
- optimizer: AdamW;
- learning rate: `1e-4`;
- weight decay: `0`;
- physical batch size: `1`;
- gradient accumulation: `8`;
- full endpoint steps: `300` optimizer steps;
- seed: `1701`;
- no rank, target-module, optimizer, step, or seed sweep.

All comparison policies reuse this one endpoint. No policy receives additional
demonstrations or gradient updates.

## Stage 0A: Provenance, Data, Identity, And Capacity

Stage 0A may decode discovery rows only. It performs no closed-loop rollout and
no full 300-step training.

Required audits:

1. checkpoint metadata proves `40` pretraining tasks;
2. normalized exact intersection with the three target identities is zero;
3. each HDF5 source has at least `45` usable demonstrations and all three split
   partitions;
4. train/validation/test episode and frame intersections are zero;
5. duplicate row hashes are zero within and across partitions;
6. images, state, actions, language, BDDL, and processor mappings are finite and
   match the official LIBERO/SmolVLA path;
7. expert replay or source terminal metadata confirms demonstration success;
8. zero-effect adapter matches Base at `<=1e-6` postprocessed max error;
9. only frozen target modules are trainable;
10. `20` micro-fit steps on `24` fixed discovery rows reduce fixed-subset mean
    loss by at least `1%` with finite nonzero gradients;
11. checkpoint save/reload error is `<=1e-6`;
12. group assignment covers every trainable tensor exactly once;
13. coefficient `0/1` reproduces Base/full, and single-group effective weight
    scaling error is `<=1e-6` relative max error;
14. peak CUDA allocation is at most `15.5 GiB`.

Failure classification:

- invalid source, mapping, split, checkpoint, scaling, gradient, or reload:
  `IMPLEMENTATION_OR_DATA_FAILURE`;
- valid data but rank-4 cannot reduce subset loss after one preregistered
  rank-8 capacity check: `LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT`;
- exact target overlap or unavailable essential source:
  `FATAL_PREIMPLEMENTATION`.

One rank-8 capacity check is allowed only when rank-4 gradients are healthy but
the subset loss does not decrease by `1%`. It repeats the same `20` steps, rows,
seed, optimizer, and targets. It does not change the scientific method. No
second capacity check is allowed.

## Stage 0B: Full Endpoint And Discovery Headroom

After Stage 0A passes:

1. train the frozen 300-step standard-LoRA endpoint once;
2. persist and disk reload it;
3. evaluate subset and held-out validation losses;
4. run Base and standard LoRA on the `12` target discovery reset cases;
5. run Base and standard LoRA on the `12` original-task retention discovery
   cases;
6. run action validity and policy-disruption diagnostics.

Headroom passes when Base fails at least `3 / 12` target cases. Endpoint
capacity passes when it fits the fixed subset, produces an action effect above
the practical threshold, remains valid, and is not more than `4 / 12` paired
target cases worse than Base.

Base saturation is `CONDITION_TOO_SEVERE_OR_NO_HEADROOM`. Endpoint loading,
training, or semantic failure is not a scientific method result.

## Stage 0C: Response Audit And Six-Configuration Search

Compute direct discovery group responses for coarse and fine groupings under
shared flow draws. Required health:

- every response finite;
- at least two groups exceed the practical action threshold;
- response matrix effective rank at least `2`;
- condition number reported;
- no group assignment or hash mismatch.

Exactly six selectable configurations:

1. `famr_fine_l01`;
2. `famr_fine_l1`;
3. `famr_fine_l10`;
4. `famr_coarse_l1`;
5. `retain_a05`;
6. `retain_a08`.

Coefficients are fit on discovery only. Validation selects already materialized
checkpoints. The top two FAMR offline configurations receive the `9` target and
`9` retention validation reset cases; RETAIN's two scalar arms receive the same
validation identities. The frozen validation score and tie-breaks in the
proposal select one FAMR and one RETAIN arm.

The selected grouping with `lambda=0` is materialized as
`famr_target_only`. `standard_lora_new_task`, Base, and this fixed ablation are
not selection configurations.

## Stage 0C Mechanism Smoke

Before Stage A:

- selected coefficients are finite and in `[0,1]`;
- coefficient dispersion exceeds `0.05` (`max(c)-min(c)`);
- FAMR differs above the practical threshold from RETAIN and target-only on at
  least `25%` of relevant validation rows;
- equal-mean scalar does not practically match full FAMR on target fit,
  retention, and action validity;
- response-fidelity thresholds in the rebuttal pass;
- all five policies persist and disk reload;
- finite and Base-relative action-validity gates pass;
- FAMR original-task validation success is no more than `1 / 9` below Base;
- no privileged inference input, extra inference call, or test identity exists.

If coefficient dispersion fails or equal-mean shrinkage matches, classify
`SIMPLE_BASELINE_EXPLAINS_METHOD`. If valid independent response evidence
decisively misses the frozen fidelity thresholds, classify
`ROBUST_EMPIRICAL_DESIGN_FAILURE`. Borderline stochastic fidelity receives the
one repeated-draw check from the rebuttal.

## Stage A Confirmatory Screen

Freeze exactly five policies in this order:

1. `smolvla_base`;
2. `retain_scalar_proxy`;
3. `famr_full`;
4. `famr_target_only`;
5. `standard_lora_new_task`.

Target Stage A: `3` tasks x `4` reset identities = `12` paired cases per policy,
`60` episodes.

Original-task retention Stage A: `3` tasks x `4` reset identities = `12`
paired cases per policy, `60` episodes. It may share the same detached run but
must use separate manifest keys and summaries.

Stage A may permanently kill only for the active governance's catastrophic
conditions, exact trivial equivalence, or a valid simple-baseline explanation.
Small gaps and ties advance to Stage B.

## Stage B Confirmatory Prototype

Target Stage B: `3` tasks x `14` reset identities = `42` paired cases per
policy, `210` episodes.

Original-task retention Stage B uses the same `42`-case structure, `210`
episodes. It may run after target Stage B if target evidence remains viable.

Report success/count, task-balanced success, paired wins/losses/ties, bootstrap
CI, effect size, relative failure-rate reduction, per-task results, action
validity, coefficient mechanism evidence, latency, and VRAM. Timing evidence is
eligible only with proven zero resource-contention overlap.

One expansion to `84` paired target cases is allowed only if Stage B is
directionally positive or unresolved and does not exclude useful improvement.
No second expansion.

## Prototype GO

`PROTOTYPE_GO` requires:

- FAMR target success exceeds Base, RETAIN, target-only, and standard LoRA;
- FAMR-minus-each key comparator paired point estimate is positive;
- FAMR-minus-strongest-comparator absolute success gain is at least `0.10`, or
  the paired interval excludes zero with at least `0.05` gain;
- original-task success is no more than `0.05` below Base and exceeds standard
  LoRA;
- action validity passes;
- response and coefficient evidence supports nonuniform functional retention;
- novelty remains defensible against current primary sources.

After GO, immediately verify Quantized OpenVLA-OFT INT4, one second condition,
direct recent baselines, compute, latency, and figure/table artifacts.

## Result Classification

Use the current governance labels exactly. Data, implementation,
parameterization, no-headroom, and underpowered outcomes are not permanent
scientific kills. A valid Stage B loss to RETAIN, ablation, standard LoRA, or
Base is a current-formulation scientific decision.

No threshold, task, reset, policy identity, coefficient, grouping, endpoint,
processor, action postprocessor, or result interpretation may change after
confirmatory testing.

## Durable Execution And Resource Rules

Every long WSL stage runs detached with PID, heartbeat, status, partial result,
final result, stdout, stderr, exit code, and exact resume command. Resume only
missing `(policy, suite, task, reset_identity)` keys.

Before acceptance, validate JSON, completed/planned counts, exception count,
duplicate keys, manifest missing/extra keys, checkpoint hashes, simulator
synchrony, and action semantics.

The previously recorded Windows Efficiency Mode interval remains quarantined.
Any timing, throughput, wall-clock, or utilization with positive or unknown
overlap is excluded. Closed-loop success rows require synchronous execution,
zero timeout/exception, unchanged identities and semantics, and no duplicates.

## Current Boundary

Only Stage 0A implementation and audit are authorized. Stage 0B and later stages
require the preceding frozen gate to pass; they do not require routine user
approval.
