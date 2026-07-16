# S2C-VLA Reviewer B Attack

Date: 2026-07-16 KST

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Method: `S2C-VLA`, Seam-Supervised Chunk Consistency for Base-preserving
SmolVLA execution.

Proposal: `reports/s2c_vla/researcher_proposal.md`

Proposal SHA-256:
`399A3960F9FF9AFA8EDA7C3F743A95C3FD4DC711644C2398630F1E68486DC5B3`

## Summary Judgment

Conditional pass to rebuttal.

S2C has a credible path because it targets a failure axis not tested by URF:
cross-chunk boundary inconsistency rather than Base-to-expert residual
magnitude. But novelty is narrow. ChunkFlow is extremely close, and SEAM is
also directly relevant. S2C must be framed as a frozen-SmolVLA,
Base-preserving overlap edit layer, not as a new seam-aware policy-learning
framework.

The proposal may continue only if Researcher A accepts the conditions below.

## Closest Prior Boundary

Closest prior: ChunkFlow

Primary source: `https://arxiv.org/html/2607.12992v1`

Secondary prior: SEAM, `https://arxiv.org/abs/2607.04609`

ChunkFlow is closer than SEAM because S2C proposes training/development
supervision for overlap masks and boundary consistency, not only a training-free
flow-step correction. SEAM remains the closest training-free previous-tail
reference.

S2C novelty is not:

- deterministic overlap blending;
- seam consistency loss by itself;
- first/second-order smoothness loss by itself;
- action-history conditioning by itself;
- a renamed ChunkFlow adapter;
- LoRA on SmolVLA.

Permitted novelty boundary:

`A Base-preserving, identity-initialized overlap edit layer for already decoded
SmolVLA chunks that learns where to allow tail-anchored boundary edits while
leaving unselected overlap cells and future-zone cells exactly at Base.`

## Required Policy Order

The first serious comparison must keep:

1. `smolvla_base`
2. `chunkflow_overlap_proxy` or official ChunkFlow if installed and verified
3. `s2c_full`
4. `s2c_no_learned_overlap_mask_ablation`
5. `standard_lora`

`chunkflow_overlap_proxy` must be a strong transparent proxy:

- deterministic overlap blending;
- frozen/editable/future zone semantics;
- first- and second-order continuity terms;
- no strawman fixed averaging if a stronger ChunkFlow-style weighting is
  feasible locally.

SEAM/RTC/ACT-style smoothing may be included as Stage 0 diagnostics, but may
not replace ChunkFlow as the closest prior unless official ChunkFlow assets are
unavailable and a later audit proves SEAM is actually closer to the implemented
S2C mechanism.

## Major Risks

### Risk 1: S2C Is Only ChunkFlow-Lite

If S2C uses the same zones, deterministic blending, boundary loss, and history
conditioning as ChunkFlow, then it is not novel. The only defensible difference
is the frozen-SmolVLA, Base-preserving, learned edit-mask constraint.

Required rebuttal: state precisely what S2C changes relative to ChunkFlow and
why that change is technically meaningful under the same SmolVLA backbone.

### Risk 2: Previous Tail Leakage

The previous tail at inference must be the previous Base/S2C executed or
committed tail, not an expert future tail from the demonstration. Expert future
actions may be used only as training labels or diagnostics.

Required rebuttal: define the exact deployment-time previous-tail construction
and prove no future observation, future action, reward, success, done flag, or
confirmatory identity enters inference.

### Risk 3: No Boundary Headroom

URF failed partly because Base residual headroom was small. S2C must not assume
boundary artifacts exist. Stage 0 must first prove Base adjacent-chunk boundary
disagreement is meaningful on development rows.

Required Stage 0 gate:

- Base boundary jump or overlap disagreement must exceed a preregistered
  threshold;
- deterministic ChunkFlow proxy must leave residual boundary or clean-retention
  headroom;
- no global smoothing-only win can pass.

### Risk 4: Smoothness Can Hurt Task Semantics

Boundary smoothing may erase legitimate abrupt gripper or contact actions.
S2C must preserve task-relevant discontinuities and gripper events.

Required Stage 0 diagnostics:

- translation, rotation, and gripper deltas separately;
- event-preservation for gripper transitions;
- no smoothing across discrete gripper changes unless explicitly allowed by
  action semantics;
- clean-retention on non-boundary cells;
- no future-zone drift.

### Risk 5: Adjacent SmolVLA Chunks May Be Expensive

S2C needs paired neighboring decoded chunks. Existing CCIF/URF caches cover many
single windows but may not contain exact adjacent replanning windows.

Required rebuttal: specify the bounded decoding plan, maximum row count, resume
keys, and whether existing Base caches can be reused. If adjacent chunks cannot
be obtained within the local budget, classify correctly as
`IMPLEMENTATION_OR_DATA_FAILURE`, not as a scientific kill.

### Risk 6: Standard LoRA Must Remain

Because S2C trains an adapter/head on demonstrations, matched standard LoRA is
a plausible explanation. It remains the fifth policy.

## Mathematical Audit Requirements

The mathematical audit must freeze:

- chunk horizon `H`;
- replanning stride `s`;
- overlap length `K`;
- frozen, editable, and future zone definitions;
- exact previous-tail variable and deployment construction;
- exact edit mask variable, shape, initialization, and support;
- bridge target formula;
- cap/clamp values by action group;
- continuity losses;
- boundary jump and high-frequency metrics;
- gradient paths;
- clean-retention objective;
- no deterministic-action KL.

If a KL term is proposed, reject it unless valid distributions, support,
direction, estimator, gradient flow, and comparison against JS/Wasserstein/MMD,
Huber/L2, vector-field consistency, and trajectory discrepancy are justified.

## Required Ablations

1. `s2c_no_learned_overlap_mask_ablation`
2. deterministic ChunkFlow overlap proxy
3. no-boundary-loss variant or mask-randomized diagnostic if cheap
4. standard LoRA
5. gripper-event preservation diagnostic

## Stage 0 Must Stop For

- no adjacent chunk boundary headroom;
- collapsed all-zero or all-one masks;
- S2C equivalent to deterministic ChunkFlow proxy;
- S2C equivalent to no-mask ablation;
- global smoothing across all cells;
- future-zone edits when not preregistered;
- gripper event destruction;
- invalid action semantics;
- identity or checkpoint reload failure;
- use of expert future tail at inference;
- any reward/success/done/confirmatory read.

## Conditions For Researcher A

Researcher A must accept:

1. ChunkFlow remains the closest prior and policy 2.
2. SEAM remains a secondary prior and optional diagnostic.
3. S2C novelty is narrowed to a frozen-SmolVLA Base-preserving learned overlap
   edit layer.
4. Previous-tail inference input is previous executed/committed Base/S2C tail,
   never expert future tail.
5. Stage 0 must first prove adjacent boundary headroom.
6. Deterministic blending alone cannot count as S2C.
7. Gripper transitions and legitimate discontinuities must be protected.
8. Standard LoRA remains required.
9. No deterministic-action KL is allowed.
10. URF and all previous methods remain closed.

If these conditions are accepted, proceed to Researcher A rebuttal. If not,
S2C must be rejected before implementation.
