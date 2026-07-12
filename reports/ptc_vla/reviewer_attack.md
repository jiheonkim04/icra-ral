# PTC-VLA Reviewer Attack

Date: 2026-07-12 KST

Role: Reviewer B

Frozen proposal hash: `15A3027E02DFE46EF2B56461A245307E9588F13431A1C92952DDD76683964CC7`

## Closest Work

Primary sources checked:

- ALAM: https://arxiv.org/html/2605.10819v1
- Conservative Offline Robot Policy Learning via Posterior-Transition Reweighting: https://arxiv.org/html/2603.16542v1
- LaWAM: https://arxiv.org/html/2606.15768
- From Pixels to Tokens: https://arxiv.org/html/2605.04678v1
- RotVLA: https://arxiv.org/html/2605.13403v1
- LaST0: https://arxiv.org/html/2601.05248v1
- IntentVLA: https://arxiv.org/abs/2605.14712

## Novelty Attack

PTC-VLA is close to the broad latent-transition/action-supervision family. ALAM is the nearest conceptual prior because it learns structured latent transitions and transfers them to VLA policy learning. LaWAM and RotVLA also use latent action/world-action structures to condition action generation. Posterior-transition reweighting is close in name and supervision flavor.

Reviewer B does not reject before implementation because PTC-VLA is materially narrower and different:

- no action-free video pretraining;
- no algebraic video latent encoder;
- no joint flow-matching VLA backbone;
- no posterior sample reweighting objective;
- no SO(n) latent action representation;
- no selector, ranker, verifier, residual correction, or image repair.

This is not enough for a paper claim yet, but it is enough for a bounded prototype under current governance.

## Baseline Attack

The method is vulnerable to simple baselines:

- `global_mean_action` may succeed on stereotyped local tasks.
- `phase_mean_action` may capture most of the transition prior.
- `ptc_no_transition_ablation` may match full if the transition latent is useless.
- `frozen_smolvla` may dominate because the learned head discards image information.

These must be preregistered as kill/Stage B gates.

## Leakage Attack

The prototype must not use:

- simulator state;
- object pose;
- reset identity;
- reward;
- success;
- future observation;
- future action target;
- BDDL predicates.

Training may use paired policy-input state transitions from collected traces. Inference may use only current and previous policy-input state, task code, phase from elapsed step fraction, and learned training priors.

## Implementation Attack

If the full head is simply a state-only MLP or phase-mean lookup, it is not a method. The no-transition ablation and phase-mean baseline must be implemented in the same runner and reported with identical task/reset allocation.

## Reviewer Decision

Decision: `IMPLEMENTATION_REQUIRED`

Reason: no exact prior-art duplication, no trivial mathematical equivalence proven before implementation, and required resources are locally available.
