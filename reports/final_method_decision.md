# Final Method Decision

Date: 2026-07-12 KST

Final decision: `PROTOTYPE_EVIDENCE_INSUFFICIENT_FOR_TERMINAL_CLAIM`

## Decision

The implementation-v2 campaign should not be described as two genuine method-level kills.

The correct postmortem status is:

- `PhaseBarrier-VLA`: `UNDERPOWERED_PROTOTYPE_INCONCLUSIVE`
- `CensorCredit-VLA`: `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

Both prototypes failed their preregistered GO gates. That remains true. What changes is the strength of the scientific conclusion: the artifacts do not support the terminal claim that both implemented methods were genuinely killed.

## Why The Terminal Claim Is Insufficient

PhaseBarrier had only `2` held-out episodes per variant, all variants scored `0/2`, and training positives were effect-compatibility labels concentrated in contact rather than task-success labels. Full PhaseBarrier changed actions, but the result is an underpowered negative smoke test.

CensorCredit had only `2` held-out episodes per variant and, more importantly, the full censored model was identical to the uncensored ablation because every generated label pair was identical. The intended censored-credit mechanism was not tested as a distinct mechanism.

## No Final Method Is Promoted

No final method direction is ready.

The only coherent next mechanism would be intervention-generated, sequence-level policy-distribution training rather than post-hoc action wrappers. Focused primary-source review shows this is already close to Set-Supervised Diffusion Policy, TORL-VLA, ConRFT, VLA-Corrector, and OpenVLA-OFT. The current artifacts do not contain enough fresh evidence to define a review-resistant distinct method.

## Operational Consequence

Do not merge this postmortem into `main` unless the user wants ledgers there later. Preserve it on `codex/implementation-v2-empirical-postmortem`.

Do not rerun rollouts, retrain PhaseBarrier, tune CensorCredit thresholds, or rebrand either method. Any future work must start from a new preregistered goal that either:

- repairs CensorCredit only enough to create genuinely different censored versus uncensored labels, then repeats a statistically meaningful evaluation; or
- proposes a new policy-distribution training method after targeted novelty review against the closest primary sources.

This postmortem itself authorizes neither path.
