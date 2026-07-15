# KITE-VLA Preregistration

Date: 2026-07-15 KST

Proposal SHA-256:
`FA00DE56D14E4C69388BE1642F7D52153841D58E77FD5A3F5C68B6C624A152B8`.

## Frozen Method

- horizons: `{5,20}`;
- global discovery-only ridge affine operator per horizon;
- ridge coefficient: `1e-4`;
- standard-deviation floor: `1e-6`;
- Huber delta: `1.0`;
- fixed rank-4 SmolVLA LoRA target set;
- KITE coefficients: `{0.1,0.3,1.0}` only;
- no inference-time method component.

No event, hazard, memory, smoothing, candidate search, clipping, KL, reward,
or success label enters the method.

## Evidence Partitions

Fixed HDF5 task sources:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

- discovery/training demos: `0..7`;
- validation demos: `8..9`;
- confirmatory evaluation: official simulator resets only after configuration,
  policies, tasks, thresholds, and manifests are frozen.

The operator is fitted only on discovery. Validation can select one KITE
coefficient. Confirmatory outcomes cannot retune KITE.

## Stage 0A: Data, Operator, Headroom, And Identity

Enumerate every frame with `t+20` available, and both horizons for each frame.
Persist key

`(partition,suite,task_identity,source_hash,demo_id,frame_index,horizon)`.

All hard gates:

1. proposal and source hashes match;
2. manifest JSON round-trips and reproduces its canonical hash before detached
   launch;
3. partial/result serializers round-trip NumPy fixtures before launch;
4. all actions and `ee_states` are finite and aligned;
5. duplicate, missing, extra, and split-overlap keys are zero;
6. at least `512` discovery and `96` validation rows per horizon;
7. positive discovery variance in all six command and state coordinates;
8. every task has validation rows and contributes at most `40%` of the sampled
   audit subset;
9. each task-agnostic `F_H` has finite rank `6` and beats the global
   discovery-mean state target by at least `50%` validation MSE;
10. each task reports operator error; no task may be silently dropped;
11. frozen Base median normalized realized-state Huber exceeds the
    demonstrated-action operator residual by at least `25%`, or the absolute
    gap is at least `0.02`, on the deterministic validation subset;
12. objective magnitudes and gradients are finite; KITE sends nonzero gradient
    to expected LoRA targets and none to frozen Base parameters;
13. initialized and disk-reloaded rank-4 LoRA reproduces Base native flow and
    decoded actions within `1e-6`;
14. Base parameter hash is unchanged;
15. exceptions are zero.

Decisions:

- all pass: `KITE_STAGE_0A_PASS_STAGE_0B_ALLOWED`;
- source, alignment, variance, count, or operator gate:
  `KITE_STAGE_0A_DATA_FAILURE`;
- no Base realization deficit: `KITE_STAGE_0A_NO_HEADROOM`;
- objective cannot send legal nonzero action-path gradient:
  `KITE_STAGE_0A_DESIGN_FAILURE`;
- hash, serialization, processor, identity, persistence, or execution defect:
  `KITE_STAGE_0A_IMPLEMENTATION_FAILURE`.

None is a scientific kill. A failed class may not be relabeled from outcome
knowledge.

## Stage 0B: Fixed Mechanism Micro-Fit

Only after Stage 0A pass, run exactly `20` optimizer steps for:

1. KITE with `lambda_k=0.3`;
2. transparent GeoPredict proxy;
3. cumulative-action-target ablation;
4. standard LoRA.

Use identical discovery rows, noise/time draws, optimizer, rank, targets, and
step count. Require finite nonzero expected gradients, persisted/reloaded
checkpoints, KITE distinct from all controls, lower KITE realization error,
bounded action delta, and clean validation action retention. This is not
configuration selection or closed-loop evidence.

## Bounded Validation Search

At most six total trained configurations:

- three KITE coefficients `{0.1,0.3,1.0}`;
- one GeoPredict proxy;
- one cumulative-action-target ablation;
- one standard LoRA.

One seed per configuration unless the fixed run is genuinely unresolved; no
more than two seeds may then be used before selection. KITE selection score:

- `40%` normalized multi-horizon realization improvement;
- `30%` clean-action reconstruction improvement;
- `20%` Base action retention;
- `10%` action validity.

Any nonfinite action, reload failure, Base-hash change, or retention failure is
ineligible. Tie break: smaller `lambda_k`. Save all configurations and negative
results. Freeze one KITE checkpoint before confirmatory rollout.

## First Serious Closed-Loop Comparison

Exactly five policies in this order:

1. `smolvla_base`;
2. `geopredict_kinematics_proxy`;
3. `kite_full`;
4. `cumulative_action_target`;
5. `standard_lora`.

Stage A uses four fixed task identities and resets `20262301..20262303` per
task: `12` paired episodes per policy, `60` total. Stage A may permanently stop
only for mechanism invalidity, no headroom, catastrophic degradation, clear
prior/ablation dominance, or exact trivial equivalence. Small differences
advance.

Stage B adds resets `20262304..20262310` per task, producing `40` paired
episodes per policy and `200` total across Stage A+B. Report task-balanced
official success, paired wins/losses/ties, bootstrap interval, effect size,
failure-rate reduction, per-task success, realization error, clean retention,
action validity, checkpoint identity, compute, and latency. Resource-contention
overlap or overlap-unknown efficiency evidence is excluded.

One expansion to `80` paired episodes per key policy is allowed only if Stage B
is genuinely unresolved under active governance.

## Capacity Diagnostic

Rank `8` is not a search configuration. One rank-8 diagnostic is allowed only
after a valid Stage 0B proves
`LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT`, before confirmatory access, with
the scientific method, coefficient set, data, and gates unchanged. It is not
allowed after any other failure class.

## Paper-Candidate Gate

Require KITE to beat Base, the GeoPredict proxy, cumulative-action target, and
standard LoRA on matched success while retaining clean behavior and reducing
realization error. Then verify unchanged KITE on Quantized OpenVLA-OFT INT4 via
compatible QLoRA and add one claim-specific second condition.
