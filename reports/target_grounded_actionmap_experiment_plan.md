# Target-Grounded ActionMap Experiment Plan

Date: 2026-07-08

This plan is a gated research plan, not an authorization to run experiments. No experiment, training, rollout, download, GPU job, OpenVLA-OFT execution, or local proxy diagnostic happened in this pass.

## Stage 0: Anchor Gate Required First

Purpose: verify that the ActionMap-style action decoder substrate is credible before adding target grounding.

Required comparisons:

1. Mean-action baseline.
2. Linear/L1 action head.
3. Simple MLP action head.
4. ActionMap-style heatmap/candidate head.
5. Oracle nearest candidate upper bound, clearly labeled invalid as method evidence.

Required checks:

- held-out action L2;
- translation L2;
- rotation L2;
- gripper error;
- candidate diversity/collapse;
- no eval-action leakage;
- no future-action use at inference.

Continue only if ActionMap-style head beats mean, linear/L1, and simple MLP and does not collapse.

## Stage 1: Feasibility Only

No heavy training. No real VLA fine-tuning. No rollout. No GPU. No downloads.

Questions:

- Can the ActionMap-style baseline be approximated in a credible official-style setup?
- Can target/object priors be obtained without leakage?
- Can LIBERO-Para object lexical or paraphrase splits be linked to the selected LIBERO tasks?
- Can old fixed-prior TCA positive evidence be mapped to a target-conditioned heatmap design without using the old weak head?
- Can LoRA/adapter later be used as a training tool rather than topic novelty?

Stage 1 output:

- source/asset/readiness table;
- exact target-prior source audit;
- exact baseline suite;
- updated GO/NO-GO decision.

## Stage 2: Bounded Diagnostic After Stage 1 Green

Run only after Stage 0 and Stage 1 are green.

Compare:

1. Mean action.
2. Linear/L1 action head.
3. Simple MLP action head.
4. ActionMap-style heatmap.
5. ActionMap plus canonicalization.
6. ActionMap plus single grounded 3D point.
7. ActionMap plus destination-only point.
8. Target-Grounded ActionMap.
9. Oracle target upper bound, clearly labeled invalid as method evidence.

Metrics:

- held-out action L2;
- translation L2;
- rotation L2;
- gripper error;
- heatmap top-k target consistency;
- wrong-target proxy;
- paraphrase/object lexical robustness;
- clean retention;
- counterfactual target sensitivity;
- candidate diversity/collapse;
- leakage audit status.

Continue only if Target-Grounded ActionMap beats ActionMap alone on wrong-target or object lexical/paraphrase subsets, while preserving clean retention and counterfactual target sensitivity.

## Stage 3: Real VLA / Official Benchmark Path

Run only after Stage 2 green.

Possible path:

- SmolVLA or OpenVLA/OFT-style adapter if feasible.
- LoRA or adapter training as implementation tool only.
- Official LIBERO/LIBERO-Para or ActionMap-compatible benchmark evidence.
- Multi-task and ideally multi-model table before paper claim.

Required comparisons:

- base VLA/OFT recipe;
- standard imitation/L1 adapter;
- ActionMap or closest official ActionMap baseline;
- direct single-point action-head injection if reproducible or approximated fairly;
- canonicalization-only for paraphrase/object lexical robustness;
- Target-Grounded ActionMap;
- oracle target only as upper bound.

## Paper-Grade Evidence Target

A paper-grade continuation would eventually need:

- official benchmark success or defensible SOTA-axis table;
- multiple tasks and datasets or suites;
- at least two model/backbone settings if feasible;
- ablations for target prior, conditioning mechanism, counterfactual consistency, paraphrase objective, and heatmap decoding;
- robustness to simple baselines and latest anchor-paper baselines;
- no leakage and no oracle target as method evidence.

## Current Decision

Do not run this plan yet as a method experiment.

Immediate required decision:

`NEED_ACTIONMAP_ANCHOR_REPRO_FIRST`
