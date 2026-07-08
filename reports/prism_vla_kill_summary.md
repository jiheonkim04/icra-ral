# PRISM-VLA Kill Summary

Decision: kill PRISM-VLA as the current main RA-L route.

## Original Hypothesis

PRISM-VLA proposed paraphrase-robust semantic-action consistency training for VLA robot policies. The core hypothesis was that policies should produce consistent action distributions for paraphrases that preserve task semantics, while preserving action-distribution differences for true object or target changes.

The route targeted the LIBERO-Para failure mode: VLA policies can suffer large drops under paraphrased instructions, especially object-level lexical variation, and these failures are primarily planning-level trajectory divergence rather than low-level execution noise.

## Strongest Positive Evidence

- Official LIBERO-Para metadata was integrated with local LIBERO task instructions and HDF5 action chunks.
- A deterministic held-out paraphrase group split was implemented before training.
- The held-out split preserved train and held-out paraphrase groups with no group leakage.
- Base held-out paraphrase degradation was measurable: clean proxy `0.519538`, held-out paraphrase proxy `0.457110`, drop `0.062428`.
- PRISM+canonicalization beat simple paraphrase augmentation on primary held-out robustness: `+0.055205`.
- The diagnostic produced PRIDE, consistency, object lexical, syntactic, counterfactual sensitivity, clean retention, and action-trajectory divergence proxy metrics.

## Decisive Negative Evidence

- Canonicalization-only beat the best PRISM variant on primary held-out paraphrase proxy: `0.474066` versus `0.436356`.
- Canonicalization-only beat the best PRISM variant on PRIDE: `46.686731` versus `31.985592`.
- Best PRISM primary held-out delta versus canonicalization was negative: `-0.030420`.
- Counterfactual sensitivity was not preserved versus canonicalization-only.
- PRISM improved auxiliary consistency and beat simple augmentation, but that did not survive the stronger canonicalization baseline.

## Exact Kill Criterion Triggered

The canonicalization dominance gate failed: canonicalization-only matched or beat every PRISM variant on primary held-out paraphrase/PRIDE robustness metrics, and the best PRISM variant weakened counterfactual/object sensitivity.

This triggers the route-level kill criterion: a language-robustness method is invalid for continuation if canonicalization-only beats it on held-out paraphrase robustness, or if it improves paraphrase consistency by weakening counterfactual/object sensitivity.

## Why Not Attempt The Real VLA Diagnostic Yet

A real SmolVLA adapter diagnostic would be premature because the low-cost primary anti-baseline gate already failed. Running a real adapter comparison before PRISM beats canonicalization on held-out proxy metrics would spend compute testing a method that is currently explained by a simpler lexical normalization baseline.

The only valid future real-adapter diagnostic would need a new predeclared reason why the canonicalization-only result is an artifact of the tiny surrogate. It would also need to directly compare canonicalization-only, PRISM, and PRISM+canonicalization without full fine-tuning, rollout, GPU work, downloads, OpenVLA-OFT, or paper-grade claims.

## Why Not RA-L-Stable

PRISM-VLA has a real literature-motivated failure mode and useful diagnostic infrastructure, but it does not currently provide method evidence beyond canonicalization-only. A RA-L-stable route needs a robust baseline gap on primary held-out metrics while preserving sensitivity to real instruction changes. PRISM failed both the canonicalization dominance gate and the counterfactual sensitivity gate.

Execution boundary for this archive: documentation only. No new experiment, training, rollout, loss computation, GPU job, download, heavy VLA import, OpenVLA-OFT execution, or paper claim occurred.
