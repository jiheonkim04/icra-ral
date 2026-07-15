# VDR-VLA Preregistration

Date: 2026-07-16 KST

Proposal SHA-256:
`0229EBC15901F4FE1EDD3839AB6B984AFA3E0E99836B5C88CF21F2C7DE2B3E72`.

## Frozen Method

- visual feature: frozen SmolVLA-compatible visual feature pooled to `[960]`;
- projection: discovery-fitted PCA/whitening to `K=32`;
- horizons: `{4,12}`;
- static predictor: discovery-only ridge actionless predictor;
- ridge coefficient: `1e-4`;
- Huber delta: `1.0`;
- VDR coefficients: `{0.1,0.3,1.0}`;
- identity-preserving rank-4 LoRA or equivalent zero-effect adapter;
- no inference-time VDR module.

No KL, reward, success, event, future-action latent, action-history residual,
candidate reranking, clipping, scheduler, or end-effector realization target
enters the method.

## Evidence Partitions

Fixed HDF5 task sources:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

- discovery/training demonstrations: `0..7`;
- validation demonstrations: `8..9`;
- confirmatory evaluation: official simulator resets only after method,
  checkpoint, policies, tasks, thresholds, and manifests are frozen.

PCA/whitening and static predictors are fitted only on discovery rows.
Validation can select only one VDR coefficient. Confirmatory outcomes cannot
retune VDR.

## Stage 0A: Data, Residual, Headroom, And Identity

Enumerate every frame with `t+12` available. Persist key:

`(partition,suite,task_identity,source_hash,demo_id,frame_index,horizon)`.

Hard gates:

1. proposal and source hashes match;
2. manifest JSON round-trips and reproduces its canonical hash before detached
   launch;
3. partial/result serializers round-trip NumPy fixtures before launch;
4. visual features, actions, and proprioception are finite and aligned;
5. duplicate, missing, extra, and split-overlap keys are zero;
6. at least `512` discovery and `128` validation rows per horizon;
7. positive discovery variance in all retained residual coordinates;
8. every task has validation rows and no task contributes more than `40%` of
   the sampled audit subset;
9. the actionless static predictor beats the discovery-mean future-feature
   target by at least `25%` validation MSE;
10. the action-conditioned residual probe beats the actionless residual probe
    by at least `5%` relative validation MSE or `0.02` normalized Huber;
11. the FutureVLA-style full future-latent proxy leaves enough residual error
    for at least the same margin;
12. objective magnitudes and gradients are finite;
13. VDR sends nonzero gradient to expected trainable parameters and none to
    frozen Base parameters;
14. initialized and disk-reloaded adapter reproduces Base native flow and
    decoded actions within `1e-6`;
15. Base parameter hash is unchanged;
16. action validity is `1.0`;
17. exceptions are zero.

Decisions:

- all pass: `VDR_STAGE_0A_PASS_STAGE_0B_ALLOWED`;
- source, alignment, variance, count, or target gate:
  `VDR_STAGE_0A_DATA_OR_SUPERVISION_FAILURE`;
- no Base or FutureVLA-proxy residual headroom:
  `VDR_STAGE_0A_NO_USABLE_HEADROOM`;
- action-conditioned residual probe fails over actionless probe:
  `VDR_STAGE_0A_DESIGN_FAILURE`;
- hash, serialization, processor, identity, persistence, gradient, or action
  validity defect: `VDR_STAGE_0A_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.

None is a scientific kill.

## Stage 0B: Fixed Mechanism Micro-Fit

Only after Stage 0A pass, run exactly `20` optimizer steps for:

1. VDR with `lambda_v=0.3`;
2. FutureVLA-style full future-latent proxy;
3. no-action-residual ablation;
4. standard LoRA.

Use identical discovery rows, noise/time draws, optimizer, rank, targets, and
step count where technically valid. Require finite nonzero expected gradients,
persisted/reloaded checkpoints, VDR distinct from all controls, lower dynamic
residual error, bounded action delta, clean validation retention, and action
validity.

## Bounded Validation Search

At most six total trained configurations:

- three VDR coefficients `{0.1,0.3,1.0}`;
- one FutureVLA proxy;
- one no-action-residual ablation;
- one standard LoRA.

One seed per configuration unless the fixed run is genuinely unresolved; no
more than two seeds may then be used before selection.

VDR selection score:

- `35%` dynamic residual improvement;
- `25%` full-versus-ablation margin;
- `20%` clean action retention;
- `15%` action validity;
- `5%` efficiency.

Any nonfinite action, reload failure, Base-hash change, retention failure, or
action-validity failure is ineligible. Tie break: smaller `lambda_v`. Save all
configurations and negative results.

## First Serious Closed-Loop Comparison

Exactly five policies in this order:

1. `smolvla_base`;
2. `futurevla_latent_alignment_proxy`;
3. `vdr_full`;
4. `vdr_no_action_residual`;
5. `standard_lora`.

Stage A uses four fixed task identities and resets `20262401..20262403` per
task: `12` paired episodes per policy, `60` total. Stage A may permanently
stop only for mechanism invalidity, no headroom, catastrophic degradation,
clear prior/ablation dominance, or exact trivial equivalence. Small
differences advance.

Stage B adds resets `20262404..20262410` per task, producing `40` paired
episodes per policy and `200` total across Stage A+B. Report task-balanced
success, paired wins/losses/ties, bootstrap interval, effect size,
failure-rate reduction, per-task success, residual error, clean retention,
action validity, checkpoint identity, compute, and latency. Resource-contention
overlap or overlap-unknown efficiency evidence is excluded.

One expansion to `80` paired episodes per key policy is allowed only if Stage B
is genuinely unresolved under active governance.

## Capacity Diagnostic

Rank `8` is not a search configuration. One rank-8 diagnostic is allowed only
after a valid Stage 0B proves `LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT`,
before confirmatory access, with the scientific method, coefficient set, data,
and gates unchanged. It is not allowed after any other failure class.

## Paper-Candidate Gate

Require VDR to beat Base, the FutureVLA proxy, the no-action-residual ablation,
and standard LoRA on matched success while retaining clean behavior, preserving
action validity, and reducing action-conditioned dynamic residual error. Then
verify unchanged VDR on Quantized OpenVLA-OFT INT4 and add one
claim-specific second condition.
