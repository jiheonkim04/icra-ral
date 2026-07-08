# PRISM-VLA Task Definition

## Topic

**PRISM-VLA: Paraphrase-Robust Semantic-Action Consistency Training for Vision-Language-Action Robot Policies**

## Anchor Problem

LIBERO-Para reports a large paraphrase robustness gap in VLA models: paraphrased instructions can cause 22-52 percentage-point success drops across seven VLA configurations, with object-level lexical variation a dominant failure source and most failures attributed to planning-level trajectory divergence rather than low-level execution noise.

Primary sources:

- LIBERO-Para paper: https://arxiv.org/abs/2603.28301
- LIBERO-Para project page: https://cau-hai-lab.github.io/LIBERO-Para/
- Official code and metadata: https://github.com/cau-hai-lab/LIBERO-Para
- Hugging Face dataset page: https://huggingface.co/datasets/HAI-Lab/LIBERO-Para

## Hypothesis

VLA policies should keep action distributions close for paraphrases that preserve task semantics, while remaining sensitive to true object or target changes. A difficulty-aware consistency objective should improve paraphrase robustness beyond simple paraphrase augmentation, instruction canonicalization, and generic perturbation training.

## First Diagnostic Scope

The first executable milestone is CPU-only and exploratory:

- use local LIBERO task instructions and HDF5 action chunks when available,
- optionally use the official LIBERO-Para metadata CSV for paraphrase categories and PRIDE-style difficulty,
- train only tiny NumPy surrogate action-distribution policies,
- report offline proxy metrics only,
- avoid OpenVLA-OFT, GPU jobs, simulator rollouts, heavy VLA imports, and paper-grade claims.

## Non-Goals

This direction does not continue the killed routes:

- Target-Prior TCA-Map
- CSS-Shield
- ExecSpec-Repair
- AMP-GD
- ResetSpec-Retarget
- Phase-Locked Retiming
- TL-ChunkRepair
- ContactTube-Aug

Existing LIBERO, HDF5, report, test, and safe-runner infrastructure may be reused when it helps produce bounded evidence.
