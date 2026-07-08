# PRISM-VLA Experiment Plan

## State 1 Diagnostic

Run the smallest executable paraphrase robustness diagnostic:

- data: official LIBERO-Para metadata CSV if present, plus local LIBERO BDDL instructions and local HDF5 action chunks,
- model: tiny CPU NumPy semantic-action distribution policy,
- output: softmax distribution over candidate local LIBERO action chunks,
- evidence label: exploratory offline proxy, not standard success.

## Required Variants

1. `base_no_paraphrase_training`: train only on clean original instructions.
2. `simple_paraphrase_augmentation`: train on clean instructions plus selected paraphrases with direct supervision.
3. `instruction_canonicalization_baseline`: train on clean instructions with deterministic lexical canonicalization at train/eval.
4. `prism_vla_consistency`: train with supervised clean/paraphrase loss plus difficulty-weighted same-task consistency and counterfactual separation.

## PRISM Objective

The PRISM arm optimizes:

- supervised action-distribution/action-chunk loss on clean and paraphrase examples,
- same-task paraphrase distribution and action consistency,
- difficulty weighting from LIBERO-Para structural and keyword similarity,
- counterfactual object/target separation so true instruction changes are not collapsed,
- clean retention tracking against the base clean proxy.

## Metrics

Required metrics:

- clean success/proxy,
- paraphrase success/proxy,
- paraphrase consistency,
- object-lexical variation robustness,
- counterfactual object sensitivity,
- action trajectory divergence,
- clean retention,
- PRIDE/difficulty-weighted robustness when LIBERO-Para metadata is available.

## Scaling Gate

Continue only if the CPU diagnostic shows:

- measurable base degradation under paraphrases,
- PRISM beats simple paraphrase augmentation on at least one robustness metric,
- clean proxy does not collapse,
- counterfactual sensitivity is preserved.

Next milestone after a continue decision: held-out paraphrase split or a real local VLA adapter smoke. Do not launch OpenVLA-OFT or heavy training locally.
