# RCV-VLA Researcher Proposal

Date: 2026-07-13 KST

Method: `RCV-VLA`, Receding-Chunk Verification for Frozen VLAs.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

## Claim

Frozen action-chunk VLAs can fail because queued suffix actions are executed after the observation has changed. `RCV-VLA` trains a lightweight verifier from the frozen policy's own queued-vs-fresh first-action disagreement and uses it to decide when to reset/replan a chunk. It is a problem-first prior extension of SV-VLA, not a claim that generic frequent replanning is novel.

## Closest External Prior

SV-VLA: https://arxiv.org/abs/2604.02965.

SV-VLA uses a heavy VLA as a low-frequency macro-planner and a lightweight verifier to check whether planned actions remain valid under updated observations. RCV differs by using frozen-policy self-disagreement as the supervision source and by testing whether a tiny local verifier can approximate fresh-action deviation without image-verifier training or success labels.

Official SV-VLA code is reported at https://github.com/edsad122/SV-VLA. If it cannot be run directly on SmolVLA/LIBERO, the first prototype must use a faithful transparent proxy and never call it an official reproduction.

## Method

At training-data acquisition time, run frozen SmolVLA with its normal queued action chunks. At each step:

1. record the queued action `a_t^queue`;
2. compute the stateless fresh first action `a_t^fresh = first(pi(o_t, q_t, l))`;
3. compute disagreement `d_t = ||a_t^queue - a_t^fresh||_1 / 7`;
4. create label `y_t = 1[d_t > tau_train]`, where `tau_train` is a training-only quantile of `d_t`;
5. train a lightweight verifier `h_phi(z_t)` with binary cross-entropy.

Features:

- current robot state/proprioception;
- queued postprocessed action;
- previous postprocessed action;
- chunk index fraction;
- task id one-hot.

At inference, RCV executes the queued action when `h_phi(z_t) <= theta_train`; otherwise it resets/replans and executes the fresh first action from the newly generated chunk.

## Prototype Comparison

1. `queued_frozen_smolvla`
2. `sv_deviation_proxy`
3. `rcv_full`
4. `rcv_no_context_ablation`
5. `stateless_first_action`

The `sv_deviation_proxy` directly computes queued-vs-fresh action disagreement at every step and replans by threshold. It is the closest-prior proxy and an upper-cost baseline.

The `stateless_first_action` variant is the simple reviewer-killer baseline: always replan every step.

## Falsifiable Hypothesis

Observed failure/assumption:

`queued action chunk suffixes are stale under updated observations`

Intermediate mechanism:

`queued-vs-fresh action disagreement spikes at stale suffix steps`

Policy behavior:

`RCV replans at high-disagreement steps and continues low-disagreement prefixes`

Closed-loop outcome:

`RCV improves success over queued base and approaches stateless/SV proxy performance with fewer heavy-policy calls`

## Kill Conditions

Kill if any holds:

- no measurable queued-vs-fresh disagreement exists in Stage 0;
- the verifier cannot predict held-out disagreement above a trivial majority baseline;
- `rcv_full` does not beat `queued_frozen_smolvla`;
- `rcv_no_context_ablation` matches `rcv_full`;
- `stateless_first_action` or `sv_deviation_proxy` explains all improvement without RCV showing meaningful heavy-policy-call savings;
- RCV requires reward, success, simulator state, object pose, or future observations at inference.
