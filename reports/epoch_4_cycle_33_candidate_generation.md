# Epoch 4 Cycle 33 Candidate Generation

Date: 2026-07-16 KST

Decision: `AFID_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

Candidate count: exactly `3`

Previous method: `LCG-VLA`

Previous decision: `LCG_STAGE_0_DESIGN_FAILURE`

Governance: current post-COVI minimum-sufficient governance with one genuinely
new mechanism, LoRA only as implementation infrastructure, and the closest
external prior in the first serious comparison.

## Candidate 1: AFID-VLA

Full name: Action-Factor Instruction Densification for Base-preserving SmolVLA

Closest prior: FineVLA

Primary source: https://arxiv.org/html/2605.27284v1

Positive prior: FineVLA reports that fine-grained instruction supervision
improves steerable control and does not sacrifice goal-level success, with
fine-grained-only gains of `+1.4` to `+8.1` over raw-only and best mixed ratios
around `1:2` to `1:1`.

Contribution type: `PRIOR_EXTENSION`

Mechanism: derive bounded action-factor labels from discovery/validation
demonstrations, such as approach axis, dominant translation direction, gripper
event timing, rotation sign, and terminal motion class. Train or audit an
identity-preserving sparse factor predictor from deployment-observable
RGB/proprio/language/Base chunks. Use predicted factors to gate only the action
cells whose residual distribution is factor-conditioned and noncollapsed.

Minimal difference from prior: FineVLA augments policy training with
fine-grained instructions. AFID converts demonstration-derived action factors
into a Base-preserving residual-gate mechanism for SmolVLA, with exact Base
passthrough when factor confidence is low.

First serious comparison:

1. `smolvla_base`
2. `finevla_action_factor_proxy`
3. `afid_full`
4. `afid_no_factor_ablation`
5. `standard_lora`

Scores:

- provisional novelty: `22 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `90 / 100`

Rationale: AFID directly addresses the LCG failure by replacing broad language
contrast with sparse, auditable action factors. Required labels can be derived
from existing LIBERO demonstrations without privileged inference inputs.

## Candidate 2: ACR-VLA

Full name: Action-CoT Residual Rationale for Base-preserving SmolVLA

Closest prior: ACoT-VLA

Primary source: https://arxiv.org/html/2601.11404v2

Positive prior: ACoT-VLA reports strong LIBERO results with action-space
reasoning, including `98.5` average in its LIBERO table, and provides official
code at https://github.com/AgibotTech/ACoT-VLA.

Contribution type: `PRIOR_EXTENSION`

Mechanism: learn a compact sequence of action-rationale waypoints from
demonstration chunks and condition a bounded residual gate on agreement between
Base chunks and predicted action rationales.

First serious comparison:

1. `smolvla_base`
2. `acot_reference_trajectory_proxy`
3. `acr_full`
4. `acr_no_rationale_ablation`
5. `standard_lora`

Scores:

- provisional novelty: `18 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `84 / 100`

Rationale: ACR has the strongest action-reasoning prior, but it risks
collapsing into a renamed coarse-intent method already explored locally unless
the rationale representation is made sharply distinct.

## Candidate 3: GCF-VLA

Full name: Geometry-Canonical Factorization for Base-preserving SmolVLA

Closest prior: GEAR-VLA

Primary source: https://arxiv.org/html/2606.08530v2

Positive prior: GEAR-VLA reports `98.7%` LIBERO average and `88.7%`
zero-shot LIBERO-Plus average through geometry-aware action representations
and embodiment canonicalization.

Contribution type: `IMPLICIT_GAP_SOLUTION`

Mechanism: derive proprioceptive contact-frame factors from end-effector state
and Base chunks, then gate residuals in a canonicalized local frame rather than
raw action coordinates.

First serious comparison:

1. `smolvla_base`
2. `gear_geometry_proxy`
3. `gcf_full`
4. `gcf_raw_action_frame_ablation`
5. `standard_lora`

Scores:

- provisional novelty: `21 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `6 / 10`
- decisive experiment feasibility: `7 / 10`
- total: `82 / 100`

Rationale: GCF is attractive but weaker locally because current cached data do
not expose object-level geometry or depth. A proxy based only on proprioception
may be too under-observed to reproduce the prior's mechanism.

## Selection

Selected method: `AFID-VLA`

Selected score: `90 / 100`

Selection decision: `AFID_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

AFID is selected because it has the best combination of positive prior,
mechanism novelty, data feasibility, and decisive local Stage 0 auditability.
It uses a different mechanism axis from LCG: sparse action-factor supervision
instead of broad language-null contrast. It also preserves the prior-first
comparison by placing a FineVLA-style action-factor proxy as policy 2.
