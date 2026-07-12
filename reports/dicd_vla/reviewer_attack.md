# Reviewer Attack: DICD-VLA

Date: 2026-07-12 KST
Branch: `codex/auto-method-20260712-01-dicd-vla`
Role: Reviewer B

Frozen proposal hash:

`B3D53F728974517A21DD91E45444C0611137AF1B10E15E46298F43FF5D150CC1`

## Primary Prior-Art Risk

DICD-VLA sits close to the newest delay/latency VLA work.

- DEFLECT explicitly targets delay-robust VLA execution by constructing stale/fresh counterfactual pairs and optimizing a flow-matching likelihood surrogate.
- TIC-VLA explicitly addresses latency consistency by decoupling slow reasoning and fast reactive control.
- RobustVLA covers robustness-aware post-training under observation and actuation perturbations.

However, the proposal is not killed as exact prior art at this stage because the specific implemented component is different:

- no flow-matching likelihood ratio;
- no stale/fresh preference objective;
- no slow/fast semantic-control architecture;
- no online RL post-training;
- no backbone fine-tuning.

The method must remain an explicit delay-indexed adapter over action chunks. If implementation drifts into generic delay post-training, adaptive chunk-length tuning, or direct DEFLECT replication, it is killed.

## Strongest Simple Baseline

The simple killer baseline is:

`direct_chunk_index_delay`

It executes `A_t[d]` from the frozen policy chunk under declared delay `d`. If this baseline matches or beats DICD, the learned adapter is unnecessary.

## Strongest Direct Baseline

The closest direct baseline is a transparent DEFLECT proxy. Full DEFLECT is not locally faithful without flow-matching likelihood access for SmolVLA, so the preregistered local proxy is direct stale/fresh chunk-index preference:

- stale action: `A_t[0]`;
- fresh indexed action: `A_t[d]`;
- no learned history adapter.

The direct baseline must be described transparently as a proxy, not as an official reproduction.

## Leakage Risks

Training traces must not include the evaluation reset identities.

No simulator state, task success, future observation, reset identity, or evaluation outcome may be used at inference.

The declared delay `d` is allowed because it is a deployment condition, not a privileged task signal.

## Stronger-Backbone Objection

OpenVLA-OFT INT4 previously solved the SmolVLA hard slices under clean execution. A SmolVLA-only DICD success may be a weak-backbone artifact. Prototype GO is allowed on SmolVLA, but paper-readiness requires second-backbone testing on quantized OpenVLA-OFT INT4 and clean retention.

## Kill Conditions Before Rollout

Kill or classify as implementation/data failure before expensive rollout if:

- action chunk access collapses to a single action;
- `H <= d`;
- training labels are nearly constant;
- no trainable parameter receives finite nonzero gradient;
- loss does not decrease on the training trace;
- disk-reloaded checkpoint does not reproduce the in-memory adapter;
- full delayed actions equal the direct chunk-index baseline on the smoke set.

## Reviewer Decision

Decision: `PROCEED_TO_PREREGISTRATION_AND_IMPLEMENTATION`

This is not a novelty endorsement. It is a cheap evidence gate: DICD-VLA may proceed only because the exact simple baseline is clear and the first prototype can quickly reveal whether the learned adapter adds anything beyond direct chunk-index execution.
