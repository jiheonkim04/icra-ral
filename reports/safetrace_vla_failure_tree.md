# SafeTrace-VLA Failure Tree

Root failure: SafeTrace-VLA is not a valid RA-L continuation route after the bounded STATE 0-1 gate.

## Branch 1: Simple Baseline Collapse

- SafeTrace proxy preference accuracy: `1.0`.
- Safety-only/risk-only preference accuracy: `1.0`.
- Generic DPO proxy accuracy: `1.0`.
- Triggered kill criterion: safety-only and generic preference baselines matched the intended method signal.

## Branch 2: Utility Retention Not Established

- Task-success labels in sampled local proxy traces: unavailable.
- Safe success / unsafe success: unavailable.
- Stop-on-risk utility-loss pair rate in the proxy: `0.0125`, but this is not benchmark success retention.
- Triggered risk: the route cannot claim utility-preserving safety without official benchmark success or replay/control utility.

## Branch 3: Official Safety Benchmark Not Reproduced

- SafeManip: official path identified, not local.
- LIBERO-Safety: official code/data path identified, not local.
- ForesightSafety-VLA: process metrics identified, no local code/data path found in the audit.
- Local standard LIBERO HDF5 was used only as a proxy source.
- Triggered risk: local proxy-first method invention cannot support a paper-grade safety route.

## Branch 4: Temporal Preference Novelty Not Separated

- Preference pair generation worked, but the labels were solved by the monitor-risk score itself.
- The proposed temporal preference objective did not improve on generic monitor-derived preferences.
- Triggered risk: the method contribution collapses to relabeling risk-only monitor outputs.

## Surviving Positive Signal

The source audit, temporal monitor smoke, preference-pair metrics, and anti-baseline comparison are reusable as an early kill gate. They should be used only after an official safety benchmark/source reproduction is available, not as the starting point for another custom method.

