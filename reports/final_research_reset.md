# Final Research Reset

Date: 2026-07-07

## Decision

Do not continue Target-Prior TCA-Map or CSS-Shield as the current main RA-L-stable routes.

Both routes produced useful diagnostics and reusable infrastructure, but neither cleared the required online robotics evidence gates. The next project should start only after a short literature-driven topic selection step, and it should be rollout-first and baseline-first.

## Killed Routes

### Target-Prior TCA-Map

- original hypothesis: target-conditioned action decoding with fixed or learned target priors can improve wrong-target robustness under low compute.
- strongest positive evidence: fixed-prior TCA produced strong offline proxy gains, prior-source audit found no inference-time leakage, and TCA beat ActionMap in some 7D action-quality diagnostics.
- decisive negative evidence: the best online 7D TCA head did not beat the mean-action baseline, and valid rollout-level support was not established.
- kill criterion triggered: online action-quality gate failed before credible rollout support.
- reusable artifacts: LIBERO split tooling, target-prior audits, ActionMap/TCA comparisons, 7D action bridge, expert replay sanity, online 7D diagnostic heads.
- why not RA-L-stable: offline proxy strength did not transfer into a deployable online action source, and TCA-Select had no measurable headroom.

### CSS-Shield

- original hypothesis: a lightweight counterfactual semantic/safety shield can reduce wrong-target and unsafe VLA actions while preserving useful behavior.
- strongest positive evidence: controlled proposal diagnostics showed full CSS-Shield beating clipping-only and safety-only on semantic wrong-target metrics.
- decisive negative evidence: Phase 2 native-action diagnostic had full vs safety-only wrong-target delta `0.0`, full vs clipping-only wrong-target delta `0.0`, full intervention rate `1.0`, and reward/success `0.0 / false`.
- kill criterion triggered: full CSS-Shield failed to beat safety-only on native-action wrong-target reduction and behaved like a full-intervention shield.
- reusable artifacts: WSL LIBERO/RoboSuite setup, native SmolVLA CPU inference, rollout diagnostic scripts, safety/object metric framework, Phase 2 autopilot gates.
- why not RA-L-stable: native-action evidence does not show semantic value beyond safety-only, so scaling would risk polishing a non-novel shield.

## Common Failure Patterns

- Offline proxy evidence did not transfer reliably to online rollout support.
- Semantic/safety components sometimes beat no-method baselines but failed against simpler baselines such as safety-only or mean-action.
- Planner/report sprawl can hide the absence of loss, rollout, or baseline-gap evidence.
- Native policy weakness limits downstream method evaluation unless competence is verified first.
- Action-quality gates are essential before spending time on rollout variants.
- A rollout-first gate should be reached within 48 hours for any new RA-L control topic.

## Reset Rule

The next topic must produce either a rollout metric, a concrete baseline gap, or a concrete failure diagnosis early. Do not start another long planning chain without execution evidence.

