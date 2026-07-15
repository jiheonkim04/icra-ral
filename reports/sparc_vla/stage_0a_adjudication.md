# SPARC-VLA Stage 0A Adjudication

Date: 2026-07-15 KST

Raw runner decision: `SPARC_STAGE_0A_IMPLEMENTATION_FAILURE`

Campaign decision:
`SPARC_STAGE_0A_IMPLEMENTATION_OR_PROTOTYPE_ACTION_VALIDITY_FAILURE_NO_SCIENTIFIC_KILL`

Failure class: `IMPLEMENTATION_OR_DATA_FAILURE`

This is a frozen pre-rollout implementation and prototype action-validity
failure. It is not a scientific kill of the SPARC mechanism.

## Execution Continuity

Attempt 1 completed and atomically persisted both planned observation rows
before a missing capture reset caused an extra hook invocation. The exception
is preserved in `stage_0a_implementation_blocker_attempt_1.json`, and the full
runtime directory is preserved under
`runs/sparc_vla/stage0a_attempt_1_capture_reset/`.

The one preregistered Stage 0A implementation repair added the missing capture
reset and an assertion before the synthetic reference prediction. No method,
operator, coefficient, threshold, task, identity, or evidence partition was
changed.

The final worker, PID `306`, is dead and is not rerun. Its durable state is:

- status and heartbeat: `completed`;
- exit code: `0`;
- partial and final JSON: parsed;
- completed/planned observations: `2 / 2`;
- exception count: `0`;
- duplicate observation indices: `0`;
- missing/extra observation indices: `0 / 0`.

## Gates That Passed

- proposal hash matched
  `CC2F9ACCE2A26EC438C58F2854ADC95134354C245CAD8ED961D29A895DBC697D`;
- the post-residual hook captured all `10` denoising steps at `[1, 50, 720]`;
- capture-only, unconfigured, identity-operator, removed-hook, and configured
  reload maximum action errors were all exactly `0.0`;
- all `10 / 10` activation rows and `2 / 2` action rows acted;
- the synthetic conceptor was finite, with eigenvalues in
  `[0.0009967021, 0.2669203]`;
- the Base checkpoint hash was unchanged;
- confirmatory records read were `0`.

## Failing Frozen Gate

Both action rows failed only the preregistered Base-relative range-safety
gate. Absolute maxima and translation, rotation, gripper, and full-chunk delta
limits passed on both rows. The failures were:

- row 0 outside fraction `0.142857` versus allowed `0.07`, and p99 exceedance
  `0.043343` versus allowed `0.031530`;
- row 1 outside fraction `0.10` versus allowed `0.078571`, and p99 exceedance
  `0.078015` versus allowed `0.064260`.

The operator used here was a synthetic unlabeled smoke operator, not a fitted
SPARC operator built from frozen discovery labels. No labeled activation
collection, training, bounded validation search, closed-loop rollout, or
confirmatory evaluation occurred. The failed smoke therefore establishes
that this frozen prototype integration is ineligible to advance; it does not
measure the scientific claim against Base, COAST, the ablation, or LoRA.

## Ruling

The single allowed Stage 0A repair has been consumed. Do not change the ridge,
aperture, operator construction, beta, action-validity thresholds, hook site,
or synthetic smoke and rerun SPARC. Do not start Stage 0B, validation search,
rollout, or confirmatory testing. A materially redesigned method belongs to a
new cycle and may not be presented as a SPARC rescue.

## Resource Evidence

The two user-reported Windows gaming and Efficiency Mode intervals remain
recorded in `reports/resource_contention_intervals.json`. Stage 0A contains no
closed-loop success rows. Latency, throughput, wall-clock efficiency, CUDA
memory, and resource utilization are not used as paper evidence or in this
decision.

## Campaign Action

Close SPARC unchanged and continue automatically to Epoch 4 Cycle 20. Generate
exactly three new prior-anchored candidates and select exactly one. Do not
rescue FAMR, PCAV, or SPARC.
