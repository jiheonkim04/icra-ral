# Epoch 4 Cycle 34 Candidate Generation

Date: 2026-07-16 KST

Decision: `BRID_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

Candidate count: exactly `3`

Previous method: `AFID-VLA`

Previous decision: `AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE`

Governance: current post-COVI minimum-sufficient governance with one genuinely
new mechanism, LoRA only as implementation infrastructure, and the closest
external prior in the first serious comparison.

## Candidate 1: BRID-VLA

Full name: Base-Residual Implicit Diffusion for SmolVLA action chunks

Closest prior: Diffusion Policy

Primary sources:

- https://diffusion-policy.cs.columbia.edu/
- https://github.com/real-stanford/diffusion_policy

Positive prior: Diffusion Policy reports action-sequence denoising policies
that outperform prior robot learning methods across 12 tasks and four
benchmarks, with official code, configs, logs, and checkpoints.

Contribution type: `PRIOR_EXTENSION`

Mechanism: learn a Base-conditioned residual score field over action chunks.
Instead of replacing SmolVLA actions, BRID samples or denoises a bounded
residual initialized at zero around the frozen Base chunk. The residual model
is trained on discovery demonstrations and selected on validation only. At
inference it receives deployment-observable inputs and the Base chunk, never
demonstration actions, rewards, success flags, object poses, or future frames.

Minimal difference from prior: Diffusion Policy models raw action chunks as a
conditional denoising process. BRID models only the residual distribution
around a strong frozen VLA Base and includes an identity-preserving zero
residual path with explicit action-delta caps.

First serious comparison:

1. `smolvla_base`
2. `diffusion_policy_action_chunk_proxy`
3. `brid_full`
4. `brid_no_base_residual_ablation`
5. `standard_lora`

Scores:

- provisional novelty: `23 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `19 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `94 / 100`

Rationale: BRID directly addresses AFID's hard-gate brittleness by replacing
classification-gated edits with a continuous residual score field while
preserving exact Base identity. It has the strongest official prior and a
clean local proxy path from existing LIBERO demonstrations.

## Candidate 2: FART-VLA

Full name: Frequency-Action Residual Tokenization for Base-preserving SmolVLA

Closest prior: FAST / FAST+

Primary source: https://arxiv.org/abs/2501.09747

Positive prior: FAST reports that frequency-space action tokenization improves
VLA action modeling for dexterous, high-frequency tasks and can match diffusion
VLA performance with reduced training time.

Contribution type: `PRIOR_EXTENSION`

Mechanism: transform Base residual chunks into low-frequency DCT coefficients,
learn a sparse residual-token predictor, and reconstruct bounded action edits
through inverse DCT with exact zero-token Base passthrough.

First serious comparison:

1. `smolvla_base`
2. `fast_frequency_token_proxy`
3. `fart_full`
4. `fart_time_domain_ablation`
5. `standard_lora`

Scores:

- provisional novelty: `22 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `86 / 100`

Rationale: FART is promising and likely cheap to audit, but the strongest FAST
claim is about tokenizer-backed autoregressive VLA training. A small local
residual-token proxy may under-reproduce that prior.

## Candidate 3: RACT-VLA

Full name: Residual Action-Chunk Transformer for Base-preserving SmolVLA

Closest prior: ACT

Primary source: https://github.com/tonyzhaozh/act

Positive prior: ACT provides official chunked imitation-learning code and a
strong prior for temporal action-chunk modeling and temporal ensembling.

Contribution type: `CROSS_DOMAIN_MECHANISM_TRANSFER`

Mechanism: learn a lightweight chunk transformer over Base chunk, proprio
summary, and task language to predict bounded residual corrections with exact
zero-residual initialization.

First serious comparison:

1. `smolvla_base`
2. `act_action_chunk_proxy`
3. `ract_full`
4. `ract_no_base_condition_ablation`
5. `standard_lora`

Scores:

- provisional novelty: `18 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `16 / 20`
- technical mechanism quality: `15 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `79 / 100`

Rationale: RACT is feasible but weaker scientifically because it risks being
a generic residual sequence model rather than a distinct mechanism. ACT is
also less directly VLA-aligned than Diffusion Policy or FAST.

## Selection

Selected method: `BRID-VLA`

Selected score: `94 / 100`

Selection decision: `BRID_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

BRID is selected because it has the strongest positive prior anchor, the most
direct response to AFID's failure, identity-preserving integration, and a
decisive local Stage 0 path. Unknown empirical performance is not a rejection
reason. No BRID implementation, training, validation search, rollout, or
confirmatory-test access has happened.
