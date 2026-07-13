# Epoch 4 Cycle 1 Prior Mechanism Map

Date: 2026-07-13 KST

Decision: `PRIOR_MECHANISM_MAP_COMPLETED`

## Problem Evidence From Local Campaign

PSE-VLA was killed, but it revealed a useful diagnostic fact: every PSE variant used stateless `predict_action_chunk(... )[:, 0]` at each environment step rather than the normal queued `select_action` execution. On the expanded hard-task manifest, stateless clean SmolVLA reached `48 / 80`, bright-single reached `51 / 80`, and full PSE reached `50 / 80`. The photometric ensemble did not help, but repeated first-action replanning remained competitive across `80` paired cases.

This points to a different unresolved problem: action chunks can become stale after the world evolves, and the success-critical decision may be whether to keep executing a queued suffix or replan from the latest observation.

## Close Literature

### SV-VLA

Source: https://arxiv.org/abs/2604.02965 and https://arxiv.org/html/2604.02965v1

`AUTHOR_STATED`:

- SV-VLA combines open-loop long-horizon planning with lightweight closed-loop online verification.
- It uses a heavy VLA as low-frequency macro-planner and a lightweight verifier at control frequency.
- It triggers replanning when a deviation between planned action and closed-loop reference action exceeds a threshold.
- Official code is reported at https://github.com/edsad122/SV-VLA.

`INDEPENDENTLY_INFERRED`:

- The core mathematical object is action disagreement under updated observations.
- The method assumes a trained verifier can approximate closed-loop reference actions cheaply.
- The key untested local question is whether a frozen SmolVLA policy's own fresh first-action disagreement can supervise a useful verifier without extra labels or success feedback.
- A faithful local proxy is possible even if official SV-VLA code does not directly support SmolVLA/LIBERO: compare queued chunk actions against fresh first-action references and replan under the same deviation rule.

`CROSS_PAPER_SYNTHESIZED`:

- SV-VLA's stale-chunk diagnosis aligns with PSE's local observation that first-action replanning can be a strong baseline.
- The missing local mechanism is not "verify candidate success"; it is a cheap estimate of when the queued suffix has diverged from the frozen policy's current closed-loop action preference.

### Critical-Moment Uncertainty

Source: https://arxiv.org/abs/2603.18342

`AUTHOR_STATED`:

- Mean rollout uncertainty can miss short-lived risk spikes.
- The paper preserves transient uncertainty through max-based sliding windows, motion-aware weighting, and DoF-adaptive calibration for failure prediction.

`INDEPENDENTLY_INFERRED`:

- The paper is about failure prediction, not action generation.
- It motivates critical-moment pooling, but does not by itself define how to intervene on a frozen continuous-action VLA without a human or external recovery channel.

`CROSS_PAPER_SYNTHESIZED`:

- Critical-moment pooling could make a verifier sensitive to brief chunk-staleness spikes, but it must be tied to an action-generation decision such as replan/continue to become a VLA method.

### Online Success Memory

Source: https://arxiv.org/abs/2605.10094 and https://arxiv.org/html/2605.10094v1

`AUTHOR_STATED`:

- Frozen generative VLAs can improve in persistent deployment by storing successful observation-action segments and retrieving state-relevant chunks.
- A progress critic filters reusable segments, and retrieved chunks steer flow-based action generation.

`INDEPENDENTLY_INFERRED`:

- The method assumes repeated deployment where prior successes are available and reliable progress filtering exists.
- It is less suited to a one-shot exact-reset prototype unless a memory-building phase is explicitly part of the deployment condition.

`CROSS_PAPER_SYNTHESIZED`:

- Success memory is complementary to chunk verification: memory addresses what action prior to reuse after success, while verifier-style methods address when the current queued plan has become invalid.

### PiL-World

Source: https://arxiv.org/abs/2606.05773 and https://arxiv.org/html/2606.05773v1

`AUTHOR_STATED`:

- Policy-in-the-loop world models are needed because open-loop world models do not evaluate VLA policies that repeatedly observe, act, and replan.

`INDEPENDENTLY_INFERRED`:

- The mechanism targets evaluation, not directly closed-loop control improvement.
- Training a visual world model is not locally feasible for the current RTX 5080 campaign.

`CROSS_PAPER_SYNTHESIZED`:

- The paper strengthens the core assumption behind Epoch 4: evaluating only offline action similarity or open-loop chunks is insufficient because closed-loop observation feedback changes the state distribution.

### TempoVLA

Source: https://arxiv.org/abs/2606.06491

`AUTHOR_STATED`:

- VLA policies inherit a fixed demonstration speed.
- TempoVLA learns explicit speed control through variable-speed trajectory augmentation and model-side conditioning.

`INDEPENDENTLY_INFERRED`:

- The method requires training a speed-conditioned VLA, which is not directly available for frozen SmolVLA.
- Frozen action scaling is only a weak proxy and risks collapsing to previously killed global scaling routes.

`CROSS_PAPER_SYNTHESIZED`:

- Speed control is a meaningful deployment axis, but a local frozen-policy prototype must avoid being just fixed action scaling.

## Mechanism Opportunity

The highest-priority local opportunity is chunk-staleness verification:

observed failure/assumption -> action chunks are executed after observations change;
intermediate mechanism -> queued suffix disagrees with the frozen policy's fresh first action under current observation;
policy behavior -> continue only locally valid chunk prefixes and replan when predicted disagreement spikes;
closed-loop outcome -> fewer stale suffix actions without paying full heavy-policy inference at every step.

This is closest to SV-VLA and must be framed as a prior extension or cross-paper synthesis, not a wholly new problem.
