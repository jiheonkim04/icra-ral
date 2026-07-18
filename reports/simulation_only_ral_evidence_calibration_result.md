# Simulation-Only RA-L Evidence Calibration

Decision: `SIMULATION_ONLY_RAL_EVIDENCE_CALIBRATION_ADOPTED`

This calibration changes only post-Stage-0 evidence breadth, Stage A/B sizing,
Pareto adjudication, optional strengthening experiments, and the final paper
gate. It does not change the active method, Stage 0 mechanism, objectives,
discovery/validation rows, practical thresholds, archived results, or the
prohibition on confirmatory-test tuning. It accessed no outcomes and launched
no training or rollout.

## Narrow simulation claim

The allowed claim is:

> `ROBUST VLA MANIPULATION UNDER SIMULATED WRIST-CAMERA FAILURES`

Official policy-controlled closed-loop LIBERO execution and official task
success remain required. Physical manipulation is prohibited. No result may
be described as real-world robustness, sensor reliability, hardware safety,
sim-to-real transfer, or deployment readiness.

## Novelty depth

The existing [focused overlap audit](action_consistent_missing_view_distillation_overlap_audit_result.md)
already covers the four mechanism groups required by the calibration:

- multi-camera-to-single/missing-view policy distillation;
- VLA robustness to missing or corrupted visual input;
- wrist prediction and cross-view reconstruction; and
- retrieval/imputation under partial observation.

It retains Acar et al. (RA-L 2023), WristWorld, RL4IL, ReconVLA, and CRT as
direct anchors. Its 23-source count is not itself novelty evidence; the
mechanism comparison and surviving narrow distinction are. The frozen novelty
decision remains `INCREMENTAL_BUT_POTENTIALLY_PUBLISHABLE`.

## Calibrated stages

Stage 0 remains a validity and mechanism-activation gate, not a final
closed-loop superiority test. A consistent, noncatastrophic directional signal
with unresolved uncertainty is handled by the existing fixed-confirmation or
underpowered rule rather than an automatic permanent kill.

Stage A begins with three tasks and three held-out identities per task for the
five key policies. It expands once to five identities per task only if the
initial result is positive but uncertain, task-mixed, or near the frozen
boundary.

Stage B uses at least four tasks and three wrist-camera failure conditions. It
begins with 60 matched paired failure-condition rows per key policy. If paired
uncertainty resolves the frozen performance or noninferiority claim, it stops.
If the interval overlaps the boundary, it expands exactly once to 80 rows and
then stops regardless of significance.

## RL4IL and deployment-cost claim

Ours may establish either direct performance superiority, statistically
comparable/noninferior success plus one large structural deployment advantage,
or comparable/noninferior success plus two moderate useful advantages. Success
margins, clean retention, latency/memory protocols, retrieval storage, and
major/moderate effect thresholds must be frozen before Stage A/B outcomes.

## Optional strengthening

A second backbone is optional strong generalization evidence after positive
X-VLA Stage B. `CAMERA-ONLY REAL-IMAGE ACTION-STABILITY VALIDATION` is optional
supplementary evidence. Neither is a universal `PAPER_CANDIDATE_GO` gate. A
single-backbone simulation-only candidate instead carries a stronger burden on
tasks, wrist-failure conditions, ablations, paired uncertainty, and complete
resource reporting.

The next authorized work is unchanged: freeze and validate the one Stage 0
method specification, then preregister the numerical-noise calibration and
actual-path microbatch preflight before any optimizer step.
