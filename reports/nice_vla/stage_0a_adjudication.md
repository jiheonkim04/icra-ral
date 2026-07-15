# NICE-VLA Stage 0A Adjudication

Date: 2026-07-15 KST

Proposal hash:
`898BA577B38966D877E3EEC724EB98751BD8C2685CCD0BBA620EB6B6B9598C0A`.

Raw runner decision: `NICE_STAGE_0A_PASS_STAGE_0B_ALLOWED`.

Adjudicated decision: `NICE_STAGE_0A_PASS_STAGE_0B_ALLOWED`.

## Runtime Integrity

The first detached `nohup` wrapper exited before Python started. It produced no
runner PID, status, partial, stderr, exit code, or Linux worker and therefore is
a launcher failure, not a Stage 0A attempt and does not consume the one allowed
implementation repair. Its empty stdout log is preserved.

The persistent hidden Windows WSL host started exactly one Linux worker, PID
`382`. The worker completed and is dead. The Windows host also exited. Final
wrapper exit code is `0`; status and heartbeat are `completed`.

- planned/completed pairs: `128 / 128`;
- exceptions: `0`;
- duplicate manifest keys: `0`;
- duplicate result keys: `0`;
- missing manifest keys: `0`;
- extra result keys: `0`;
- split-overlap keys: `0`;
- partial JSON, result JSON, status, heartbeat, and validation JSON parse;
- manifest and partial key sets are exactly equal.

No row was repeated and no resume was required.

## Source And Data Gates

VLA-Corrector source commit
`9d23a0ba6fad562d3ed1a68fc52c8a12459abb41` and Apache-2.0 license passed.
The two preregistered lexicographic task mappings resolved without substitution:

- `libero_10/task_1` to the black-bowl/bottom-drawer task;
- `libero_goal/task_1` to the top-drawer/bowl task.

Each task supplied demos 0 and 1 with 32 fixed frames per demo. Frozen visual
tokens measured `[128,960]`, matching two concatenated `[64,960]` camera token
streams. Actions measured width 7 and all normalized action components were
finite and inside `[-1,1]`. Every delta latent was finite and nonconstant.

## Mathematical And Integration Gates

- rank-8 basis shape: `[122880,8]`;
- basis orthonormal max error: `4.17e-7`;
- mean gradient norm: `0.17618`;
- diagonal covariance gradient norm: `111.99485`;
- rank-8 covariance gradient norm: `30.64226`;
- frozen mean gradient during covariance fit: `0.0`;
- covariance scale clamped fraction: `0.0`;
- checkpoint reload max error: `0.0`;
- Woodbury Mahalanobis error: `1.42e-14`;
- determinant-lemma logdet error: `3.55e-15`;
- monitor-disabled Base queue/action error: `0.0`.

Tiny smoke objectives moved sensibly: mean cosine loss `1.00771 -> 0.77424`
and diagonal covariance NLL `14.24366 -> 2.68643`. These values establish only
that the interface is trainable and numerically valid.

## Evidence Boundary

Validation records, confirmatory records, task outcomes, and simulator
rollouts read: `0 / 0 / 0 / 0`. No validation search or confirmatory tuning
occurred. The tiny smoke fit is not the scientific mean or a selected
configuration.

The run happened after the reported Efficiency Mode intervals. Its elapsed
time is nevertheless not paper evidence, and Stage 0A contains no closed-loop
efficiency claim.

## Decision

Stage 0A validly passes. This is not evidence that NICE predicts real
closed-loop failures, beats VLA-Corrector, or improves success. It authorizes
only a separately frozen development-only Stage 0B1 offline observability and
headroom audit. Confirmatory tasks and reset identities remain sealed.
