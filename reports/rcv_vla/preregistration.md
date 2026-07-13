# RCV-VLA Preregistration

Date: 2026-07-13 KST

Decision: `PREREGISTERED_BEFORE_IMPLEMENTATION`

Frozen proposal hash:
`86044E841D178DB5AA485B7D12B01FF8E4274CBDFDCDAC7D427477BF0646F26F`

## Method

`RCV-VLA` trains logistic verifier heads from frozen SmolVLA queued-vs-fresh first-action disagreement. The frozen SmolVLA policy is not updated.

Primary tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

These are the same hard-task axes used in recent campaign stages, but RCV uses fresh contiguous reset identities so the PSE outcome pattern is not selected into this method.

Reset identity mapping:

- `RCV_RESET_IDENTITY_BASE = 20260801`;
- exact LIBERO initial-state index = `identity - RCV_RESET_IDENTITY_BASE`;
- the finite official initial-state vectors may overlap prior campaigns, but the method uses fresh episode keys and does not choose identities from observed RCV outcomes.

## Stage 0: Problem Diagnostic

Purpose:

- verify that queued-vs-fresh action disagreement exists;
- measure normal queued success and stateless first-action success on the same small manifest;
- confirm external chunk instrumentation is valid.

Manifest:

- identities: `20260801` through `20260805`;
- both tasks for each identity.

Variants:

- `queued_frozen_smolvla`;
- `stateless_first_action`.

Diagnostic rows record per-step `d_t = ||a_t^queue - a_t^fresh||_1 / 7`, chunk index, policy-call count, success, and exception state.

Stage 0 hard kill:

- mean `d_t < 1e-4` and max `d_t < 1e-3` across all diagnostic steps; or
- valid instrumentation proves queued and fresh actions are exactly equivalent in this environment.

Otherwise proceed to Stage 1.

## Stage 1: Acquisition, Training, Calibration

Acquisition manifest:

- train identities: `20260806` through `20260812`;
- calibration identities: `20260813` through `20260815`;
- both tasks for each identity;
- normal queued frozen SmolVLA execution.

Training labels:

- compute `tau_train` as the `0.75` quantile of `d_t` on train identities only;
- label `y_t = 1[d_t > tau_train]`.

Models:

- `rcv_full`: features `[q_t, a_t^queue, a_{t-1}, rho_t, task_one_hot]`;
- `rcv_no_context_ablation`: features `[a_t^queue, rho_t, task_one_hot]`.

Training:

- logistic verifier;
- binary cross-entropy plus L2 penalty;
- deterministic seed `260713`;
- no success labels.

Threshold selection:

- choose `theta_train` on calibration identities only by maximizing F1;
- if tied, choose the higher threshold to reduce replans.

Stage 1 hard kill:

- `rcv_full` calibration balanced accuracy is not above the majority-class baseline by at least `0.02`; and
- `rcv_full` calibration F1 is not above the no-context ablation.

Otherwise proceed to Stage 2A.

## Stage 2A: Early Paper Comparison

Manifest:

- identities: `20260816` through `20260820`;
- both tasks for each identity;
- identical task/reset manifest for every variant;
- `10` episodes per variant, `50` total episodes.

Variants:

1. `queued_frozen_smolvla`;
2. `sv_deviation_proxy`;
3. `rcv_full`;
4. `rcv_no_context_ablation`;
5. `stateless_first_action`.

The `sv_deviation_proxy` computes fresh first-action disagreement at every step and replans when `d_t > tau_train`. It is a transparent closest-prior proxy, not an official SV-VLA reproduction.

Primary metric:

- task-balanced closed-loop success.

Secondary metrics:

- raw success rate;
- paired wins/losses/ties versus each baseline;
- mean `d_t` where measurable;
- replan rate;
- heavy-policy calls per environment step;
- latency per step;
- peak CUDA allocation;
- exceptions.

Stage 2A catastrophic kill:

- `rcv_full` has `0 / 10` success while any paired baseline has at least `4 / 10`; or
- `rcv_full` is at least `30` absolute percentage points below the strongest baseline or the no-context ablation; or
- `rcv_full` uses a forbidden inference signal.

If not catastrophically killed:

- proceed to Stage 2B when RCV is tied, narrowly negative, or positive;
- allow no permanent scientific kill from a one- or two-episode difference.

## Stage 2B: Prototype Stage B

Run only if Stage 2A is not catastrophically killed.

Manifest:

- identities: `20260821` through `20260840`;
- both tasks for each identity;
- identical manifest for every variant;
- `40` episodes per variant, `200` total episodes.

Stage 2B `PROTOTYPE_GO`:

- `rcv_full` beats the strongest baseline and `rcv_no_context_ablation`;
- absolute gain is at least `10` task-balanced points at prototype scale, or paired evidence is consistently positive with meaningful failure-rate reduction;
- mechanism is active;
- no privileged inference signal is used;
- heavy-policy-call rate is lower than `stateless_first_action` and `sv_deviation_proxy`.

Stage 2B permanent kill:

- Stage 2B is complete and valid; and
- `rcv_full` is clearly worse; or
- the upper confidence bound excludes useful improvement; or
- `sv_deviation_proxy`, `stateless_first_action`, or `rcv_no_context_ablation` explains the result.

One expansion to `80` paired episodes per key policy is allowed only if Stage 2B is unresolved under current governance. No third expansion is allowed.

## Measurement Repair

Exactly one repair may occur only for invalid instrumentation, mismatched manifests, corrupt checkpoint loading, or missing required metrics discovered before adjudication. No threshold, manifest, or variant may be changed because the result is weak.
