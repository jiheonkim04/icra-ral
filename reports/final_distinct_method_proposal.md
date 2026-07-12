# Final Distinct Method Proposal

Date: 2026-07-12 KST
Branch: `codex/censorcredit-one-repair-and-final-method`

Method name: `Intervention-Set Action-Chunk Fine-Tuning (ISAC-VLA)`

Proposal status after review: `KILLED_BEFORE_IMPLEMENTATION`

## Researcher Proposal

ISAC-VLA starts from the local evidence:

- ECHO found no robust headroom from static candidate selection.
- PhaseBarrier changed actions but the phase-conditioned component was worse than a no-phase ablation.
- CensorCredit collapsed because prefix-only and recovered-outcome labels never differed.

The proposed method would avoid post-hoc scalar action wrappers. Instead, it would collect closed-loop intervention records and fine-tune a VLA action distribution on paired action chunks:

- negative chunk: the policy action sequence immediately before a human or scripted intervention;
- positive chunk: the corrective action sequence that recovered the task;
- context: image, instruction, step fraction, previous actions, and optional contact features.

The training objective would be sequence-level and distributional. For context `x`, positive chunk set `P(x)`, and negative chunk set `N(x)`, learn a chunk policy `pi_theta(a_{t:t+H}|x)` that increases likelihood or energy margin for `P(x)` while decreasing probability mass around `N(x)`.

One possible objective:

`L = -log sum_{a+ in P(x)} pi_theta(a+|x) + beta * log sum_{a- in N(x)} exp(log pi_theta(a-|x) / tau)`

An implementation that survived review would use OpenVLA-OFT-style continuous action chunk fine-tuning rather than a scalar temporal hold wrapper.

## Why It Is Distinct From Local Failures

ISAC-VLA is not ECHO because it trains the policy distribution rather than selecting among frozen action candidates.

It is not PhaseBarrier because it does not use a geometric feasibility margin or phase-conditioned action projection.

It is not CensorCredit because it does not fit a post-hoc temporal trust head. It requires explicit paired negative and corrective action chunks, so the missing censored/uncensored disagreement would be supplied by intervention data rather than inferred from weak effect scores.

## Targeted Novelty Review

Primary sources checked:

- Set-Supervised Diffusion Policy, arXiv:2606.01865, https://arxiv.org/abs/2606.01865
- TORL-VLA, arXiv:2606.09337, https://arxiv.org/abs/2606.09337
- ConRFT, arXiv:2502.05450, https://arxiv.org/abs/2502.05450
- OpenVLA-OFT, arXiv:2502.19645, https://arxiv.org/abs/2502.19645

The review finds that the core ISAC-VLA idea is already occupied:

- SDP explicitly trains from paired positive and negative action chunks from human corrections.
- TORL-VLA uses human-intervention data and an intervention-censored critic to avoid crediting pre-intervention actions for post-intervention success.
- ConRFT covers offline and online VLA fine-tuning with human interventions.
- OpenVLA-OFT supplies the exact action-chunk VLA fine-tuning substrate that ISAC-VLA would need.

## Reviewer Decision

Reviewer decision: `FINAL_METHOD_KILLED_BEFORE_IMPLEMENTATION`

Allowed kill grounds:

- `NEAR_EXACT_PRIOR_ART_DUPLICATION`
- `HARD_UNAVAILABLE_RESOURCE`

The proposal is conceptually useful but not a valid final implementation target. If implemented faithfully, it is too close to SDP plus TORL-VLA/ConRFT and would need human or robot intervention/correction data not present in the local repository. If reduced to available local LIBERO traces, it would collapse into ordinary action reweighting, post-hoc wrapper training, or a trivial behavioral-cloning variant, which the campaign rules do not allow as a final distinct method.
