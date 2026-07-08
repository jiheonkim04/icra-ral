# PRISM-VLA Reusable Artifacts

PRISM-VLA is killed as a main route, but several artifacts remain useful.

## Data Integration

- Official LIBERO-Para metadata CSV integration.
- Matching from LIBERO-Para original instructions to local LIBERO BDDL task instructions.
- Local LIBERO HDF5 action-chunk loading for offline action-distribution proxy labels.
- Fallback local exploratory paraphrase generation when LIBERO-Para metadata is missing.

## Split And Audit Infrastructure

- Deterministic held-out paraphrase group split.
- Whole-group train/held-out separation using original instruction plus LIBERO-Para `eval`, `high`, `mid`, and `low` fields.
- Split audit fields: task count, paraphrase count, group count, held-out object subset, held-out syntactic subset, group leakage, and action-label leakage note.

## Diagnostic Runner

- Safe runner: `scripts\190_prism_vla_paraphrase_diagnostic.ps1`.
- Module: `tca_map.prism_vla.paraphrase_diagnostic`.
- Focused tests: `tests\test_prism_vla_paraphrase_diagnostic.py`.
- The runner refuses downloads, GPU gates, rollouts, simulator gates, heavy-import gates, runtime-install gates, and OpenVLA/OFT gates.

## Metrics

- Clean proxy.
- Held-out paraphrase proxy.
- Paraphrase drop.
- Clean retention.
- PRIDE and difficulty-weighted robustness.
- Same-task paraphrase consistency.
- Object lexical variation robustness.
- Syntactic variation robustness.
- Counterfactual object sensitivity.
- Action trajectory divergence.
- Instruction sensitivity preservation.

## Baseline Lessons

- Canonicalization-only must be a required language-robustness baseline.
- Simple paraphrase augmentation is not strong enough as the only anti-baseline.
- Consistency gains must be checked against counterfactual object/target sensitivity.
- PRIDE/held-out robustness should be separated from auxiliary consistency metrics.

## Reuse Boundary

These artifacts can support future language robustness diagnostics or topic selection. They should not be used to revive PRISM-VLA unless a future predeclared diagnostic beats canonicalization-only on primary held-out metrics and preserves counterfactual sensitivity.
