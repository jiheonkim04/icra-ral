# RCV-VLA Reviewer B Attack

Date: 2026-07-13 KST

Reviewed frozen proposal hash:
`SHA256(researcher_proposal.md)=86044E841D178DB5AA485B7D12B01FF8E4274CBDFDCDAC7D427477BF0646F26F`

Decision: `IMPLEMENTATION_ALLOWED_WITH_CAVEATS`

## Independent Prior Search

Closest direct prior:

- SV-VLA, "Open-Loop Planning, Closed-Loop Verification: Speculative Verification for Vision-Language-Action Models", https://arxiv.org/abs/2604.02965.

Adjacent current papers screened:

- TTT-VLA, https://arxiv.org/abs/2606.03127, adapts a latent prompt from interaction data but does not make the chunk-validity decision.
- Retrieve-then-Steer / Online Success Memory, https://arxiv.org/abs/2605.10094, retrieves successful deployment memory and steers generation rather than verifying queued suffix validity.
- TempoVLA, https://arxiv.org/abs/2606.06491, learns speed conditioning rather than chunk verification.
- Realtime-VLA V2 / FASTER-style responsiveness papers are relevant to efficiency but are not the same replan-vs-continue verifier.
- Very recent test-time visual-foresight and harness/memory-agent work was noted, but it does not replace SV-VLA as the closest stale-chunk verification prior.

## Novelty Attack

The proposal is very close to SV-VLA. The paper claim must not be "closed-loop verification for chunked VLA" or "replanning when a verifier says stale"; that is already the SV-VLA axis.

The only defensible novelty is narrower:

1. use a frozen policy's own queued-vs-fresh first-action disagreement as supervision;
2. train a tiny non-image verifier on low-dimensional execution context;
3. test whether this verifier preserves most of the receding-horizon benefit while avoiding fresh VLA calls at every control step.

If `sv_deviation_proxy` matches or beats RCV and RCV does not materially reduce heavy-policy calls at comparable success, RCV is not paper-worthy.

## Trivial-Equivalence Attack

The method may collapse into one of three simple baselines:

- frequent replanning / adaptive chunk size;
- direct deviation thresholding, which is the SV-VLA local proxy;
- stateless first-action replanning every step.

The prototype must include all five declared policies and must not add more internal controls before comparing the closest prior proxy.

## Mathematical Attack

The BCE objective is mathematically valid because it compares a Bernoulli label to a verifier probability. No decorative KL or probability claim is present.

Risks:

- the threshold `tau_train` is a quantile, so the label may be arbitrary rather than task-critical;
- `d_t = ||a_t^queue - a_t^fresh||_1 / 7` may reflect harmless action stochasticity rather than stale physical state;
- low-dimensional features omit images, so the verifier may be unable to detect object-state changes;
- selecting `theta_train` after seeing evaluation results would invalidate the method.

Required before evaluation:

- compute `tau_train` only from acquisition training identities;
- choose `theta_train` only from held-out acquisition/calibration identities;
- keep evaluation identities untouched by verifier training, thresholding, or repair decisions.

## Leakage And Privilege Attack

Allowed inference features:

- current robot proprioception/state;
- current queued postprocessed action;
- previous postprocessed action;
- chunk index fraction;
- task one-hot.

Forbidden inference features:

- reward;
- success;
- object pose or simulator state;
- future observations;
- per-episode outcome labels;
- evaluation-set disagreement labels for threshold tuning.

Calling the frozen policy only when RCV replans is allowed. Calling it every step inside `rcv_full` to compute hidden disagreement is not allowed.

## Required Prototype

The first serious comparison must contain exactly:

1. `queued_frozen_smolvla`;
2. `sv_deviation_proxy`;
3. `rcv_full`;
4. `rcv_no_context_ablation`;
5. `stateless_first_action`.

The `sv_deviation_proxy` is a faithful local proxy, not an official SV-VLA reproduction.

## Reviewer Decision

Implementation may proceed only because performance is unknown, the mathematical formulation is valid, and the closest prior can be compared early.

Reviewer B will kill RCV if the evaluation shows it is merely generic replanning, matched by the no-context ablation, matched by stateless first-action replanning without efficiency savings, or worse than the direct SV-style deviation proxy without a defensible compute tradeoff.
