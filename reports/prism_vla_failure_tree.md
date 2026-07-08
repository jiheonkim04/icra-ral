# PRISM-VLA Failure Tree

Root failure: PRISM-VLA does not justify continuation as the current main RA-L route because canonicalization-only beats it on the primary held-out paraphrase robustness gate.

## Branch 1: The Target Failure Mode Is Real

- LIBERO-Para metadata was available and integrated.
- Base clean-to-held-out paraphrase degradation was measurable: `0.062428`.
- Held-out object lexical variation and syntactic variation subsets were present.
- This branch supports the problem selection, not the PRISM method.

## Branch 2: PRISM Beats A Weak Baseline

- Best PRISM variant: `prism_vla_plus_canonicalization`.
- Best PRISM held-out proxy: `0.436356`.
- Simple augmentation held-out proxy: `0.417930`.
- Best PRISM primary held-out delta versus simple augmentation: `+0.055205`.
- This branch is positive but insufficient because simple augmentation was not the strongest baseline.

## Branch 3: Canonicalization Dominates

- Canonicalization-only held-out proxy: `0.474066`.
- Canonicalization-only PRIDE: `46.686731`.
- Best PRISM held-out proxy: `0.436356`.
- Best PRISM PRIDE: `31.985592`.
- Best PRISM primary held-out delta versus canonicalization: `-0.030420`.
- This is the decisive branch: the stronger simple lexical baseline explains or exceeds the apparent PRISM gain.

## Branch 4: Counterfactual Sensitivity Weakens

- The PRISM objective was supposed to keep same-task paraphrases close while preserving differences for true object/target changes.
- State 2 reported counterfactual sensitivity preserved versus canonicalization: false.
- Improving paraphrase consistency while weakening object/target sensitivity violates the core hypothesis.

## Branch 5: Real Adapter Diagnostic Is Not The Next Step

- SmolVLA assets and runtime are ready in the repository, but the PRISM route failed before the real-adapter gate.
- Building a real adapter diagnostic now would bypass the anti-baseline rule.
- A later real-adapter diagnostic is valid only if a new predeclared hypothesis explains why canonicalization dominance is likely a surrogate artifact.

## Terminal Decision

Kill or archive PRISM-VLA as a main route. Preserve the diagnostic infrastructure, but do not scale PRISM training, run OpenVLA-OFT, or claim RA-L readiness.
