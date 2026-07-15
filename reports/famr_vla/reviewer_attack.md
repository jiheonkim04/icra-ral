# FAMR-VLA Reviewer B Attack

Date: 2026-07-15 KST

Decision: `FAMR_REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

## Sources Reviewed

- FAMR candidate generation and prior map;
- RETAIN paper and official code;
- Fisher-weighted model merging paper;
- official `lerobot/libero` dataset card;
- local IARC, MTF, RAR, EAC, LIFT, and campaign negative evidence.

## Strongest Fair Reading

FAMR asks a useful and narrow question: whether model merging for a new VLA
skill should be calibrated in the policy's action function rather than by a
single parameter interpolation coefficient. The held-out LIBERO-90 source and
original 40-task retention split make the problem locally testable.

The strongest publishable claim is not a new general theory of model merging.
It is a VLA-specific prior extension showing that action-response-constrained
merging can improve limited-demo new-skill adaptation while retaining the
generalist policy.

## Blocking Findings

### 1. Linear Response May Be Too Inaccurate

LoRA groups interact through nonlinear transformer layers. Summing one-group
action effects need not predict the action of the jointly materialized merge.
Researcher A must freeze a response-fidelity metric, evaluate direct merged
checkpoints, and stop if the response model is noninformative.

### 2. The Solver Could Be Validation Overfitting

Fitting coefficients and selecting `lambda` on the same rows would leak
selection into construction. Coefficients must be fit on discovery rows for
each fixed configuration; validation may rank configurations but may not
refit them. Confirmatory rows must remain unread.

### 3. RETAIN Fidelity Is At Risk

RETAIN studies scalar and modality-specific merging on openpi policies. A
SmolVLA LoRA interpolation is only a transparent core proxy. The report must
not imply an official reproduction, and any win must be scoped to the matched
local proxy until the OpenVLA-OFT phase.

### 4. New-Task Provenance Must Be Demonstrated

LIBERO-90 files being locally separate does not by itself prove the checkpoint
never trained on those task identities. The official 40-task metadata, exact
task-name set, and intersection with the three selected tasks must be saved.

### 5. Raw HDF5 Processing Is A Major Failure Surface

Camera order, image orientation, proprio dimensions, delta end-effector action,
gripper convention, padding, task language, and normalization must match the
official SmolVLA path. A subset loss decrease is insufficient if semantics are
wrong. The audit needs hashes, shapes, ranges, and a successful expert-replay
or official conversion check.

### 6. Task-Vector Scaling Must Be Exact

Scaling arbitrary serialized PEFT tensors can alter effective updates
quadratically or leave module scaling metadata inconsistent. Researcher A must
show mathematically and numerically that group coefficient `c` produces
`Delta W_group(c) = c Delta W_group(1)` and that `c=0/1` reproduces Base/full.

### 7. Simple Shrinkage Could Explain Everything

If all fitted coefficients are equal, FAMR is scalar RETAIN. If the full method
only wins because its average coefficient is smaller, a scalar interpolation
at that average is the actual explanation. Report coefficient dispersion and
an equal-mean scalar diagnostic before scientific adjudication.

The equal-mean diagnostic may be offline unless it becomes the strongest
plausible alternative after validation. It must not silently become a sixth
confirmatory policy without a preregistered decision rule.

### 8. Offline Action Fit Is Not Policy Improvement

Earlier campaign methods improved offline action metrics and failed in closed
loop. The validation score must include bounded closed-loop new-task and clean
retention evidence before Stage A. Offline L2 alone cannot select FAMR.

### 9. Headroom Must Exist Against Both Base And RETAIN

Base may already transfer to simple LIBERO-90 tasks. Standard LoRA may also fail
to learn them under the local budget. Either case removes a decisive test.
Use only discovery/validation identities for this audit and classify no
headroom or insufficient parameterization honestly.

### 10. Action Validity Must Be Base-Relative And Absolute

IARC demonstrated that empirical demonstration ranges can be narrower than
valid model outputs. FAMR must not inherit or retroactively change IARC's gate.
For this new method, predeclare finite checks, simulator acceptance, absolute
semantic limits, and Base-relative out-of-range excess before seeing FAMR.

### 11. Clean Retention Requires Closed Loop

Small action drift is mechanism evidence, not proof that original skills are
retained. Include matched original-task validation and confirmatory rollouts.

### 12. Resource Evidence Must Stay Quarantined

The recorded Windows Efficiency Mode interval predates this experiment, but
every run must carry overlap metadata. Timing or utilization with unknown or
positive overlap is excluded. Success rows remain subject to synchronous,
exception-free, duplicate-free manifest checks.

## Required Rebuttal

Researcher A must freeze:

1. discovery-fit versus validation-select separation;
2. exact group scaling and endpoint identity tests;
3. response fidelity and rank diagnostics;
4. task provenance and HDF5 semantic audits;
5. Base and standard-LoRA headroom rules;
6. action-validity thresholds;
7. RETAIN proxy disclosure and equal-mean shrinkage diagnostic;
8. exact rollout partitions and duplicate-key rules;
9. false-negative classification for every pre-rollout stop.

## False-Negative Calibration

Current evidence is design review only. It cannot scientifically kill FAMR.

- invalid task provenance, preprocessing, scaling, or checkpoint loading:
  `IMPLEMENTATION_OR_DATA_FAILURE`;
- nonacting rank-4 endpoint with otherwise valid method:
  `LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT`;
- Base saturation: `CONDITION_TOO_SEVERE_OR_NO_HEADROOM`;
- noninformative response after valid endpoint and adequate independent rows:
  potentially `ROBUST_EMPIRICAL_DESIGN_FAILURE`, but only after uncertainty
  excludes the preregistered useful fidelity threshold;
- small coefficient differences or wide intervals:
  `UNDERPOWERED_OR_UNRESOLVED`, with one cheap preregistered check.

False-positive risk: a development-tuned merge can appear better offline and
fail closed loop, or a weak RETAIN proxy can inflate the apparent gain.

False-negative risk: a small first-task sample or one weak LoRA endpoint can
hide a real checkpoint-merging advantage.

Reviewer confidence: `0.78` that the proposal is worth a bounded audit if all
rebuttal requirements are frozen.
