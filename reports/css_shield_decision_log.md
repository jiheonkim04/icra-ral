# CSS-Shield Decision Log

## 2026-07-06: Start CSS-Shield

Decision: start a new rollout-first project named Counterfactual Semantic Safety Shield for VLA Manipulation.

Rationale: the previous Target-Prior TCA-Map route produced useful offline fixed-prior evidence but failed online 7D action-quality and rollout-support gates.

Constraint: do not continue the old route as the main RA-L claim. Reuse reports, diagnostics, infrastructure, and negative evidence only as artifacts.

Next decision point: STATE 1 must produce simulator/rollout safety metrics or a concrete blocker.

## 2026-07-06: State 1 Minimal Rollout Diagnostic

Decision: continue to a narrow State 2 semantic-coverage diagnostic, but do not claim CSS-Shield superiority.

Rationale: State 1 produced real bounded LIBERO/RoboSuite rollout metrics using native SmolVLA actions. Full CSS-Shield reduced unsafe action rate versus no shield and clipping-only by `0.8`, but it did not beat safety-only and did not exercise wrong-target semantic intervention because the counterfactual object was missing from the observation object keys.

Constraints: next evidence must test semantic wrong-target intervention directly against clipping-only and safety-only. Reward/success stayed zero, so this is not paper-grade evidence.

