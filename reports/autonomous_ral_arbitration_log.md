# Autonomous RA-L Arbitration Log

## 2026-07-12 KST

Decision: `APPROVE_WITH_FIXED_KILL_GATE`

Reason:

- Researcher A has frozen a concrete method and produced real SmolVLA trace evidence.
- Reviewer B has identified the direct delayed chunk-index and no-history ablation as decisive comparisons.
- The cheapest decisive action is the already preregistered Stage A closed-loop rollout, not another audit.

Next action: run Stage A exactly as preregistered.

## 2026-07-12 KST Update

Decision: `MEASUREMENT_INFRASTRUCTURE_REPAIR_ALLOWED`

Reason:

- The first Stage A launch ran for about one hour without writing any partial or final artifact.
- No `stage_a_result.json` existed, no method decision was observed, and no thresholds were changed.
- To respect the single-command governance limit, the runner must checkpoint after each episode before Stage A is restarted.

Allowed repair: add per-episode partial JSON and progress logging only. Do not change variants, tasks, identities, metrics, or GO/KILL rules.

## 2026-07-12 KST Stage A Decision

Decision: `KILL_DICD_VLA_SIMPLE_BASELINE_EXPLAINS_METHOD`

Reason:

- The checkpointed Stage A rollout completed all `50 / 50` preregistered episodes.
- There were zero rollout exceptions.
- Full DICD reached `1 / 10`, task-balanced success rate `0.10`.
- Direct chunk-index delay reached `2 / 10`, task-balanced success rate `0.20`.
- Frozen delay-only reached `2 / 10`, task-balanced success rate `0.20`.
- The no-history ablation matched full DICD at `1 / 10`.
- The mechanism changed actions, but the extra delay-indexed history-conditioned adapter did not improve closed-loop success.

Reviewer B ruling: this is a valid scientific kill, not a measurement-invalid result. No repeat or rescue is allowed. Start Cycle 2 with a genuinely distinct method family.
