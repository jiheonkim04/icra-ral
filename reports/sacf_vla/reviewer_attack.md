# SACF-VLA Reviewer B Attack

Date: 2026-07-12 KST

Proposal hash: `1C43D99A42AD97C29C1BDBDED1AB1326214C8FF0F514F79309266738C5FD1A20`

Decision: `IMPLEMENTATION_REQUIRED_WITH_STRONG_BASELINES`

## Closest Papers

1. BayesVLA, "Seeing to Act, Prompting to Specify: A Bayesian Factorization of Vision Language Action Policy": https://arxiv.org/html/2512.11218v1
2. CAST, "Counterfactual Labels Improve Instruction Following in Vision-Language-Action Models": https://arxiv.org/html/2508.13446v2
3. CAG / LIBERO-CF, "When Vision Overrides Language": https://arxiv.org/html/2602.17659v1
4. IGAR, "Restoring Linguistic Grounding in VLA Models via Train-Free Attention Recalibration": https://arxiv.org/abs/2603.06001
5. FineVLA, "Fine-Grained Instruction Alignment for Steerable VLA Policies": https://arxiv.org/html/2605.27284v1

## Strongest Novelty Attack

SACF is vulnerable to the claim that it is a small local BayesVLA/CAST variant:

- BayesVLA already decomposes policy behavior into a vision-action prior and language-conditioned likelihood.
- CAST already uses counterfactual language/action data to force language-conditioned action differences.
- CAG and IGAR already target visual shortcut / linguistic blindness failures.
- FineVLA already argues that action-aligned language supervision changes steerable control.

The proposal is not a near-exact duplicate because SACF differs in multiple axes:

- it uses existing same-scene LIBERO task-family contrasts rather than VLM-generated counterfactuals;
- it trains a local semantic-prefix factorization rather than a full VLA architecture or attention recalibration;
- it uses a fixed prefix plus frozen-VLA handoff rather than proposal scoring, dual-branch guidance, or full policy fine-tuning;
- it is intended as a low-compute prototype around standard LIBERO assets.

However, any paper claim would need to position SACF as a small-data same-scene counterfactual factorization method, not as a general solution to VLA language grounding.

## Simplest Equivalent Method

The simplest equivalent method is not SACF. It is a task/phase mean prefix or a plain BC prefix trained from the same demonstrations.

If either matches SACF, the contribution collapses.

## Strongest Simple Killer Baseline

`task_phase_mean_prefix`:

- compute mean action per task and phase bin from training demonstrations;
- execute this mean prefix for the same fixed handoff fraction;
- then hand off to frozen SmolVLA.

This baseline tests whether SACF is merely learning an open-loop task script.

## Direct Prior Proxy

`cag_null_guidance`:

- query frozen SmolVLA with the true task instruction;
- query frozen SmolVLA with a null instruction;
- use predeclared fixed guidance `a = clip(a_true + 0.5 * (a_true - a_null), -1, 1)`.

This is not a faithful official CAG reproduction, but it is a cheap local dual-branch language-guidance proxy.

## Leakage Risks

- HDF5 object states must not be used as default inference features.
- Closed-loop evaluation identities must not be used to tune prefix length, guidance scale, phase bins, hidden size, or task list.
- If demonstration actions are from a different action convention than official rollout actions, the method must be killed or repaired once before confirmatory Stage A.
- Training on exact target-task demonstrations is allowed for a prototype only if plain BC and task-phase mean baselines use the same demonstrations.

## Required Implementation Conditions

Implement only if all are satisfied:

1. The SACF module exposes the factorization terms so tests can prove `sacf_full` is not identical to `plain_bc_prefix`.
2. Synthetic tests prove the counterfactual factorization term can recover semantic action factors when they exist.
3. Real-demo training records row counts, task family coverage, loss decrease, factor activation, and checkpoint hashes.
4. Stage A uses the frozen task list and five variants from the proposal.
5. No LIBERO-90 data is used in Stage A training unless a separate report amends and justifies the protocol before any Stage A result is inspected.

## Reviewer Decision

SACF is too close to BayesVLA/CAST/CAG/IGAR to be accepted as a paper method on concept alone, but current governance forbids killing for broad similarity. It is not an exact duplicate across problem, representation, supervision, objective, policy component, inference, data, and claim.

Proceed to the cheapest decisive implementation. Kill without rescue if plain BC, phase mean prefix, or null-guidance explains the result.
