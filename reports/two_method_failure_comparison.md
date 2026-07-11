# Two-Method Failure Comparison

Date: 2026-07-12 KST

Overall conclusion: the previous terminal claim `TWO_IMPLEMENTED_METHODS_KILLED` was overstrong. The two prototypes produced real executable evidence, but not enough evidence for two genuine method-level kills.

## Classification Table

| Method | Previous result label | Postmortem classification | Main reason |
| --- | --- | --- | --- |
| `PhaseBarrier-VLA` | `PHASE_BARRIER_VALID_KILL` | `UNDERPOWERED_PROTOTYPE_INCONCLUSIVE` | Full method changed actions, but all variants scored `0/2`; training positives were effect-compatibility labels, not task-success labels; evaluation had no statistical power. |
| `CensorCredit-VLA` | `CENSOR_CREDIT_VALID_KILL` | `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` | Censored and uncensored labels were identical for `24/24` records, yielding identical saved models and near-identical full/ablation behavior. |

## Shared Empirical Causes

1. Both were frozen-backbone post-hoc action wrappers.

Neither prototype changed the SmolVLA policy distribution through training. PhaseBarrier projected postprocessed actions. CensorCredit blended postprocessed actions with previous actions. Both could perturb execution, but neither taught the policy to generate a new family of action chunks.

2. Both used very small short-horizon supervision.

PhaseBarrier trained on `20` records from `5` states. CensorCredit trained on `24` records from `6` states. Both were generated from one training reset identity and two tasks.

3. Both lacked sequence-level success labels at training time.

PhaseBarrier had `0/20` short-horizon task-success positives; its `8` positives came from effect compatibility and were all `contact` rows. CensorCredit had `0/24` prefix successes and `0/24` recovered successes; its `4` positives came from score thresholds and did not differ between censored and uncensored targets.

4. Both had only `2` held-out episodes per variant.

With `n=2`, a `0/2` raw success result still has a Wilson 95% upper bound of about `0.658`, and a `1/2` result has an interval of about `[0.095, 0.905]`. The prototypes can reject the preregistered GO gate, but cannot support terminal method-family claims.

5. Both selected the same hard task pair.

The held-out evaluation consisted of one reset each for `libero_spatial/task_4` and `libero_10/task_4`. Prior project state already showed these were hard SmolVLA slices, while quantized OpenVLA-OFT INT4 solved the matched hard slice. That makes a frozen-SmolVLA local wrapper failure especially hard to interpret as a general VLA method failure.

## Different Failure Details

PhaseBarrier did not collapse to the backbone. Full PhaseBarrier shaped `255/280` and `460/520` steps, with mean action deltas `0.141024` and `0.081844`. Its failure is underpowered and success-sparse, not an integration non-action.

CensorCredit did collapse to the key ablation. Full and uncensored models have identical weights because the generated labels are identical. The saved rollout confirms near-identical behavior: full success `1/2`, ablation success `1/2`, full mean action delta `0.119921`, ablation mean action delta `0.113220`.

## What This Does And Does Not Prove

Proved:

- The two implemented prototypes did not pass their preregistered GO gates.
- PhaseBarrier, as run, did not show closed-loop improvement.
- CensorCredit, as run, did not instantiate a distinct censored-credit head.
- Post-hoc frozen-backbone action wrappers are a weak substrate for these hard slices.

Not proved:

- Phase-conditioned physical feasibility is scientifically dead.
- Intervention-censored temporal credit is scientifically dead.
- Frozen-backbone local wrappers can never work.
- A final RA-L method direction is ready.

The deeper shared cause is insufficient mechanism-bearing supervision plus no policy-distribution training, not a clean disproof of either research idea.
