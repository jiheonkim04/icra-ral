# AMP-GD State 2 Result

Diagnostic-only kill gate. This is not paper-grade evidence.

- final decision: `kill_or_reframe`
- reason: Toy evidence is matched by simple informative-probe heuristics and the tiny LIBERO diagnostic did not beat simple baselines.
- toy utility metric bug found: `False`
- utility-drop interpretation: policy utility is higher than no-probe; negative is improvement, not a metric bug
- AMP-GD privileged inference info used: `False`
- toy killed as main evidence: `True`
- LIBERO object observability green: `True`
- wrong-target metric computable: `True`
- safe micro-probe action available: `True`
- active ambiguity signal available: `False`
- LIBERO micro-probe diagnostic ran: `True`

## LIBERO Metrics

| policy | wrong-move | target move | unsafe | probe cost | reward | success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_probe_greedy` | 0.0 | 0.000109378 | 0.0 | 0.0 | 0.0 | False |
| `random_probe` | 1.0 | -9.6021e-05 | 0.0 | 0.035 | 0.0 | False |
| `safety_only_clipping` | 0.0 | 0.000109378 | 0.0 | 0.0 | 0.0 | False |
| `nearest_target` | 0.0 | 0.000109378 | 0.0 | 0.0 | 0.0 | False |
| `amp_gd_micro_probe` | 1.0 | 4.4975e-05 | 0.0 | 0.035 | 0.0 | False |

- AMP beats random-probe: `True`
- AMP beats safety-only: `False`
- random-probe matches AMP wrong-target movement: `True`
- safety-only matches AMP wrong-target movement: `False`
- nearest matches AMP wrong-target movement: `False`

## Toy Robustness

- deterministic informative-probe heuristic matches AMP-GD: `True`
- entropy-greedy heuristic matches AMP-GD: `True`
- AMP beats random-probe in all toy profiles: `True`
- AMP beats safety-only in all toy profiles: `True`
