# HEST-VLA Stage 0A Adjudication

Date: 2026-07-15 KST

Decision: `HEST_STAGE_0A_IMPLEMENTATION_FAILURE`

Scientific result: `False`.

Stage 0B allowed: `False`.

## Runtime Integrity

- worker PID: `370`, dead after normal completion;
- runner and host exit: `0`;
- completed windows: `160 / 160`;
- resumed windows: `0`;
- exceptions: `0`;
- duplicate manifest / partial keys: `0 / 0`;
- missing / extra keys: `0 / 0`;
- discovery-validation overlap: `0`;
- persisted chunk hash or reload errors: `0`;
- proposal and manifest hashes independently recompute exactly;
- result decision independently recomputes exactly.

The wrapper initially persisted the literal character `n` in the exit-code
file because its shell `printf` quoting was malformed. The foreground host
command and runner both returned `0`; only that bookkeeping file was corrected
to `0` before independent validation. No action row, metric, threshold, source,
method code, or decision was changed.

## Frozen Gate Evidence

Passed:

- source shape and finite checks;
- all arm support ranges are noncollapsed;
- validation gripper-transition coverage: `23 / 32`;
- endpoint maximum error: `4.440892098500626e-16`;
- first-action maximum error: `3.3306690738754696e-15`;
- gripper maximum error: `0.0`;
- HEST acting fraction: `0.8125`;
- median cumulative-arm energy reduction: `0.28936200680886914`;
- all three controls are distinct from HEST;
- disk round-trip maximum error: `0.0`.

Failed:

- frozen `all_variant_support_valid` gate: `False`.

Invalid support rows across all 160 windows:

- Base: `1`;
- HEST: `1`;
- MovingAverage: `1`;
- NoEndpoint: `31`;
- SplineProxy: `117`.

The single invalid Base row is a validation window outside discovery-defined
coordinate support. HEST's required whole-chunk Base fallback preserves that
row exactly, so HEST is invalid under the same frozen support test. This is not
a simulator outcome or a policy comparison.

## Scientific Ruling

This is a pre-rollout implementation/prototype support failure under the
frozen classifier, not a scientific kill. No SmolVLA model, CUDA runtime,
simulator, reward, success, done flag, video, confirmatory reset identity, or
closed-loop task-success row was read or produced.

Do not widen support, clip actions, change fallback, replace the source split,
change lambda or alpha, rerun Stage 0A, implement Stage 0B, or rescue HEST.
Continue automatically to Epoch 4 Cycle 22 exact-three candidate generation.

Timing, throughput, wall-clock, latency, and resource-utilization measurements
remain ineligible for paper evidence. The reported Windows Efficiency Mode
interval remains recorded separately and has no bearing on this decision.
