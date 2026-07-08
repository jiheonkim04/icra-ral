# PRISM-VLA Related Work Matrix

| Work | Evidence | PRISM-VLA relevance | Gap PRISM targets |
| --- | --- | --- | --- |
| LIBERO-Para (2026) | Reports 4,000+ paraphrased instructions across 10 scenarios, 22-52 pp drops, object-level lexical fragility, planning-level trajectory divergence, and PRIDE. Sources: https://arxiv.org/abs/2603.28301, https://cau-hai-lab.github.io/LIBERO-Para/, https://github.com/cau-hai-lab/LIBERO-Para | Main anchor benchmark and metric source. | Needs a training method, not only an evaluation benchmark. |
| PRIDE metric | Uses keyword and structural similarity to weight success by paraphrase difficulty. Source: https://github.com/cau-hai-lab/LIBERO-Para/tree/master/metrics | Directly motivates difficulty-aware consistency weighting. | PRIDE is evaluative; PRISM turns difficulty into a training signal. |
| LIBERO-Plus (2025/2026) | Studies broad VLA brittleness across object layout, camera, initial state, language, lighting, texture, and noise. Source: https://arxiv.org/abs/2510.13626 | Shows robustness failures are broader than paraphrases and supports multi-dataset evaluation later. | PRISM starts with semantic-action consistency and should later test non-language perturbation interactions. |
| LIBERO-PRO (2025) | Argues standard LIBERO evaluation can overstate competence and exposes memorization under manipulated objects, instructions, and environments. Source: https://arxiv.org/abs/2510.03827 | Reinforces the need for counterfactual object/target sensitivity checks. | PRISM must not buy paraphrase stability by ignoring true target changes. |
| Multilingual VLA language-sensitivity work (2026) | Reports non-English instruction drops and step-wise language sensitivity. Source: https://arxiv.org/abs/2606.11906 | Suggests PRISM may need temporally localized consistency beyond whole-chunk alignment. | State 1 only measures whole action chunks; later work should add step-wise consistency. |
| Simple paraphrase augmentation | Common baseline: train on original plus paraphrased instructions with same action labels. | Required baseline for PRISM. | May overfit seen paraphrases and lacks counterfactual separation or difficulty weighting. |
| Instruction canonicalization/dropout | Low-cost baseline that normalizes or masks language variation. | Required baseline for PRISM. | Can improve lexical invariance but risks discarding object/target distinctions. |

## Positioning

PRISM-VLA is not an online action gate or rollout wrapper. It is a training objective for semantic-action consistency under same-task paraphrases, with explicit counterfactual separation to preserve target sensitivity.
