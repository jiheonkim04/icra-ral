# TSC-VLA Preregistration

Date: 2026-07-16 KST

Method: `TSC-VLA`

Proposal hash:
`0DF143D2D8773D7ABF4FC76AB7CC083FE7EE65DF84EA06631E67C2445F6DC941`

Decision: `TSC_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`

This preregistration is frozen after Researcher A proposal, Reviewer B attack,
Researcher A rebuttal, and mathematical mechanism audit. It does not alter or
rescue CFR-VLA.

## Evidence Partitions

`DISCOVERY`:

- existing official LIBERO demonstrations only;
- used to cache Base chunks, compute robust residual scales, compute the fixed
  per-dimension `0.80` residual quantile thresholds, inspect label health, and
  train small Stage 0 probes.

`VALIDATION`:

- existing official LIBERO demonstrations only;
- disjoint demo/frame keys from discovery;
- used for Stage 0 mechanism scoring, clean-retention checks, action-validity
  checks, and later bounded validation search if Stage 0 passes.

`CONFIRMATORY TEST`:

- not used in Stage 0;
- no confirmatory reset identity, task outcome, rollout success, reward, or
  held-out closed-loop result may influence Stage 0 thresholds, architecture,
  coefficients, or stop decision.

## Stage 0 Purpose

Stage 0 is a development-only audit. It determines whether TSC has usable data,
headroom, a nontrivial mask mechanism, bounded policy integration, and a
faithful closest-prior proxy path before any closed-loop rollout.

Stage 0 is not a closed-loop scientific kill unless a later protocol explicitly
advances to closed-loop evaluation.

## Fixed Stage 0 Data Construction

Use official LIBERO demonstration records and frozen SmolVLA Base predictions.

Minimum records:

- at least `512` valid discovery windows;
- at least `128` valid validation windows;
- at least `4` task identities when locally available under the existing
  demonstration cache path.

For each valid window:

- cache deployment-observable features;
- cache `A_B in R^[50,7]` from frozen Base SmolVLA;
- align `A_E in R^[50,7]` from the demonstration action chunk;
- compute `R = A_E - A_B`;
- compute valid-step mask.

Fixed label construction:

- `S_d = median_discovery(|R_d|) + 1e-6`;
- `Tau_d = quantile_0.80_discovery(|R_d| / S_d)` over valid steps;
- `Y_h,d = 1[|R_h,d| / S_d >= Tau_d]`;
- inference mask threshold `eta = 0.5` for Stage 0;
- diagnostic action scale `alpha = 0.1` for Stage 0 action-delta and validity
  smoke only.

If discovery or validation labels are all zero or all one, stop as
`TSC_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

## Stage 0 Policies And Probes

Stage 0 must instantiate or faithfully proxy:

1. `smolvla_base`
2. `ts_mask_continuous_proxy`
3. `tsc_full`
4. `tsc_no_targeted_mask_ablation`
5. `standard_lora` diagnostic or explicit preregistered reason it is deferred
   to bounded validation because Stage 0 remains offline/probe-only.

The closest-prior proxy must not be a strawman. It must share split, inputs,
action semantics, and comparable capacity, while omitting only the
Base-error-targeted mask.

## Stage 0 Required Outputs

Artifacts:

- `reports/tsc_vla/stage_0_manifest.json`
- `reports/tsc_vla/stage_0_partial.json`
- `reports/tsc_vla/stage_0_result.json`
- `reports/tsc_vla/stage_0_result.md`
- `reports/tsc_vla/stage_0_validation.json`
- `reports/tsc_vla/stage_0_adjudication.md`
- `reports/tsc_vla/stage_0_status.json`
- `reports/tsc_vla/stage_0_heartbeat.json`
- `reports/tsc_vla/stage_0_pid.txt`
- `reports/tsc_vla/stage_0_exit_code.txt`
- `reports/tsc_vla/stage_0_stdout.log`
- `reports/tsc_vla/stage_0_stderr.log`

Metrics:

- discovery/validation record counts;
- task and phase coverage;
- positive/negative mask counts;
- label variance and mask positive fraction;
- split overlap count;
- duplicate manifest and partial key counts;
- Base finite/action-shape validity;
- mask predictor accuracy/AUROC or balanced proxy metric;
- mask predictor margin over trivial-majority and magnitude-only baselines;
- completion validation Huber for `tsc_full`, `ts_mask_continuous_proxy`, and
  `tsc_no_targeted_mask_ablation`;
- unselected-cell Base-clamp error;
- changed-cell count and mask positive rate;
- action delta mean/p95/max by translation, rotation, and gripper groups;
- official action validity;
- checkpoint save/reload;
- objective term magnitudes and gradient norms;
- frozen-parameter gradient count;
- no privileged inference input check.

## Stage 0 Pass Criteria

Pass to bounded validation only if:

- JSON artifacts parse;
- completed rows equal planned rows;
- exception count is zero or all exceptions are non-scored and adjudicated;
- duplicate key counts are zero;
- discovery/validation split overlap is zero;
- mask labels are noncollapsed on discovery and validation;
- validation mask predictor beats trivial-majority and magnitude-only baselines;
- `tsc_full` beats `ts_mask_continuous_proxy`;
- `tsc_full` beats `tsc_no_targeted_mask_ablation`;
- unselected-cell Base-clamp max error is at numerical tolerance;
- action deltas are sparse and bounded;
- official action validity passes;
- expected trainable parameters receive finite nonzero gradients;
- no frozen Base parameters receive gradients;
- no privileged inference input is used.

## Stage 0 Stop Classes

- `TSC_STAGE_0_DATA_OR_SUPERVISION_FAILURE`
- `TSC_STAGE_0_NO_USABLE_HEADROOM`
- `TSC_STAGE_0_DESIGN_FAILURE`
- `TSC_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`
- `TSC_STAGE_0_PASS_TO_BOUNDED_VALIDATION`

No rescue or rerun is allowed after a valid Stage 0 stop unless the result is
classified as implementation-invalid under the frozen stop classes.

## Bounded Validation Search If Stage 0 Passes

Maximum six total configurations.

Allowed factors:

- at most three values for `alpha`;
- at most two mask/completion architecture choices;
- at most one clean-retention coefficient family;
- at most one mask positive-rate/threshold variant.

No combinatorial grid over many variables. Selection uses validation-only
evidence and must combine mechanism activation, clean retention, action
validity, closest-prior comparison, ablation comparison, and compute overhead.
Offline action L2 alone is not a valid selection rule.

## First Closed-Loop Comparison Policy

If bounded validation selects one final configuration, the first serious
closed-loop comparison uses:

1. `smolvla_base`
2. `ts_mask_continuous_proxy` or official `ts_mask_vla` if installed
3. `tsc_full`
4. `tsc_no_targeted_mask_ablation`
5. `standard_lora`

No confirmatory-test outcome may retune the same TSC method.
