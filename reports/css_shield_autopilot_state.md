# CSS-Shield Autopilot State

## Current State

STATE 1 completed as a bounded rollout-first diagnostic.

Previous route: Target-Prior TCA-Map low-compute RA-L route is killed and archived.

Current project: CSS-Shield.

Latest result:

- rollout happened: yes, one bounded LIBERO/RoboSuite exact-init diagnostic task.
- proposal source: native SmolVLA on CPU.
- training/loss/LoRA training: no.
- downloads/GPU/OpenVLA-OFT/benchmark rollout/paper claim: no.
- full CSS-Shield reduced unsafe actions versus no shield and clipping-only by `0.8`.
- full CSS-Shield did not beat safety-only in this run.
- wrong-target semantic reduction was not exercised because the counterfactual object was not present as an observation object key.
- reward/success remained `0.0 / false`.

## Next State

STATE 2: bounded semantic-coverage diagnostic.

The next diagnostic must directly test wrong-target intervention with both intended and counterfactual targets observable, or produce a concrete blocker explaining why that cannot be done safely. Do not expand into broad planners.

## Bounds

- CPU first.
- No OpenVLA-OFT.
- No full fine-tuning.
- No large downloads without green risk assessment.
- No paper-grade claim until rollout evidence exists.
- Max 5 major milestones per execution.
- Stop if diagnostics produce no metric, concrete blocker, or kill/continue decision.

