# CCIF-VLA Reviewer B Attack

Date: 2026-07-16 KST

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Reviewed proposal: `reports/ccif_vla/researcher_proposal.md`

Proposal SHA-256:
`2AFC40F050FD7F0D28507344358CBCB70BF27CC901C57474A501D3EB87E7FAA1`

## Independent Closest-Prior Check

Reviewer B independently identifies the closest current primary-source
neighbors as:

1. Coarse-to-Control, https://arxiv.org/abs/2606.07107
2. CAC-VLA, https://arxiv.org/abs/2607.04816
3. CF-VLA, https://arxiv.org/abs/2604.24622

Additional pressure comes from Libra-VLA, https://arxiv.org/abs/2604.24921,
because it also decomposes manipulation into coarse intent and continuous
residual/refinement.

The proposal's closest-prior choice is acceptable. Coarse-to-Control is the
right primary anchor because it directly argues that action-space planning is a
better intermediate medium than text or images and reports a strong positive
LIBERO result.

## Main Novelty Risk

CCIF is dangerously close to existing coarse-to-fine and latent-action
conditioning methods:

- Coarse-to-Control already predicts coarse action-space plans before
  executable actions.
- CAC-VLA already predicts coarse-to-fine latent actions and uses a context gate
  to condition the action expert.
- CF-VLA already performs coarse initialization followed by fine residual
  refinement.
- Libra-VLA already combines discrete macro-intent and continuous micro-action
  refinement.

Therefore CCIF cannot claim novelty merely by using a coarse action
representation, future action segment labels, intent conditioning, or a gate.
Those are already active prior art.

The only defensible narrowed novelty is:

`continuous coarse motor-intent field as a Base-preserving residual constraint
around an already trained continuous SmolVLA chunk`.

This means CCIF must be tested as a conservative overlay around Base, not as a
new coarse-to-fine action generator.

## Strawman Baseline Risk

The proposed `coarse_to_control_continuous_proxy` could become too weak if it
only predicts a crude net displacement and then directly decodes actions with a
small head. That would unfairly favor CCIF.

Reviewer B requires the prior proxy to be as strong and transparent as the
local budget permits:

- same deployment-observable inputs as CCIF;
- same train/validation rows;
- same optimizer and step budget where applicable;
- same coarse-intent labels;
- same low-frequency waypoint information;
- no Base-preserving residual clamp;
- direct intent-conditioned action generation or refinement;
- no worse feature access than CCIF.

If official Coarse-to-Control code/checkpoints become locally available before
the first serious comparison, policy 2 must switch from the proxy to the
official or faithful reproduction path after a documented compatibility check.

## Mechanism Plausibility Risk

The method assumes the Base chunk is locally plausible but globally wrong in a
coarse-intent sense. This may be false. If Base residuals are dominated by
high-frequency gripper timing, action normalization, or action validity rather
than coarse displacement/rotation/waypoint errors, CCIF will be decorative.

Stage 0 must prove all of the following before training-heavy validation:

- retained coarse-intent components have nonzero variance;
- intent labels are predictable from deployment inputs above task/phase mean;
- Base-to-expert residuals have usable coarse-intent headroom;
- the prior proxy leaves residual headroom for CCIF;
- CCIF differs from the prior proxy and no-intent ablation for the intended
  reason, not because of a larger parameter budget;
- residuals are bounded and do not globally perturb every action dimension.

## Simple Equivalent Risk

A reviewer will ask whether CCIF is just one of these simpler methods:

- task/phase mean coarse intent plus residual;
- endpoint-displacement regression plus residual;
- low-pass filtered action target;
- standard LoRA with clean retention;
- Base-to-expert residual regression with a gate.

The first simple reviewer-killer remains `standard_lora`, but Stage 0 must also
include cheap diagnostics for task/phase mean intent and endpoint-only intent.
These diagnostics need not be full rollout policies unless they explain the
validation signal.

## Data And Leakage Risk

Future action chunks are legal training labels, but they must not leak into
inference or confirmatory selection. The mathematical audit and preregistration
must freeze:

- exact intent vector components and units;
- waypoint count;
- normalization statistics and which split fits them;
- train/validation/confirmatory demonstration identities;
- no confirmatory decode or action access during Stage 0;
- no use of simulator success, reward, done, object state, or reset identity in
  label construction.

## Required Conditions For Rebuttal

Researcher A must explicitly accept these conditions before mathematical audit:

1. CCIF novelty is narrowed to Base-preserving continuous coarse-intent
   residual constraint, not generic coarse-to-fine action generation.
2. `coarse_to_control_continuous_proxy` remains policy 2 unless official
   Coarse-to-Control assets are installed and verified.
3. The prior proxy must not be a strawman and must receive the same legal
   inputs, coarse-intent labels, data split, and comparable optimizer budget.
4. `ccif_no_coarse_intent_ablation` remains the key ablation.
5. Matched `standard_lora` remains the mandatory simple reviewer-killer.
6. Stage 0 must include task/phase mean intent and endpoint-only intent
   diagnostics.
7. Intent labels, waypoint count, normalization, residual cap, and pass gates
   must be frozen before implementation.
8. Offline action L2 alone cannot select a configuration or support a paper
   claim.
9. Identity-preserving initialization and disk reload must reproduce Base
   within `1e-6`.
10. No privileged inference input, confirmatory-test tuning, task/reset change,
    or TSC rescue is allowed.

## Conditional Decision

Conditional pass to Researcher A rebuttal.

CCIF is not yet dead because Base-preserving continuous intent-constrained
residualization is meaningfully different from Coarse-to-Control's discrete
plan-execute tokenizer, CAC-VLA's VLM-native latent-action gate, and CF-VLA's
coarse initialization for low-NFE generation. But the distinction is narrow.

If Researcher A does not accept the narrowed novelty boundary and proxy/ablation
conditions, CCIF should be killed before mathematical audit.
