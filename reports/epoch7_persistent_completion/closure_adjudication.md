# Stability-Qualified Completion Closure

Decision: `NO_REPEATABLE_GAP` at the frozen pre-policy demonstration gate.

All ten LIBERO-Goal tasks completed validly, all actions were finite, both cold-repeat tasks matched, and every standard-selected demonstration reached native success. No model or policy was loaded, no dataset reward/done signal was read, and no Ours mechanism ran.

Three tasks failed the 30-step immediate neutral dwell: wine bottle on cabinet, bowl on cabinet, and bowl on plate. Each failure was recovered when the exact unused expert suffix was executed before the same dwell. This is a useful bounded observation: the `On` predicate can become true before the selected demonstration has reached a durable placement state.

It is not the frozen cross-mechanism problem. All three disagreements belong to the same `placement` mechanism and the same `On` predicate family, so mechanism diversity was one rather than the required two and a single predicate family explained 100% rather than at most two-thirds. All recoveries likewise belong to one mechanism. The task9 wine-rack suffix instability is retained as a secondary observation but does not satisfy the primary immediate-dwell gap.

The contract permits no outcome-driven suite expansion. Relaxing diversity, adding Long tasks after seeing this result, promoting the last-action-repeat control, or designing a general termination method from an `On`-specific effect are prohibited rescues. The result does not support any policy reliability or ranking claim.
