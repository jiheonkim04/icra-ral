# S2C-VLA Researcher A Rebuttal

Date: 2026-07-16 KST

Decision: `S2C_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Proposal: `reports/s2c_vla/researcher_proposal.md`

Proposal SHA-256:
`399A3960F9FF9AFA8EDA7C3F743A95C3FD4DC711644C2398630F1E68486DC5B3`

Reviewer attack: `reports/s2c_vla/reviewer_attack.md`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Researcher A accepts all Reviewer B conditions. S2C remains live only under the
narrowed formulation below.

## Accepted Novelty Boundary

Researcher A accepts that S2C novelty is limited to:

`A Base-preserving, identity-initialized overlap edit layer for already decoded
SmolVLA chunks that learns where to allow tail-anchored boundary edits while
leaving unselected overlap cells and future-zone cells exactly at Base.`

S2C does not claim to invent seam-aware training, overlap blending, boundary
losses, smoothness regularization, or action-history conditioning. ChunkFlow is
the closest positive prior and remains policy 2 in the first serious
comparison. SEAM remains a secondary previous-tail consistency prior and may be
used as a diagnostic, but it does not replace ChunkFlow as the closest prior.

## Exact Difference From ChunkFlow

The technical difference that S2C may defend is not deterministic seam
smoothing. It is the combination of:

- frozen SmolVLA Base chunks;
- zero-initialized learned edit mask over overlap cells;
- bounded tail-anchored bridge applied only where the mask permits;
- exact passthrough for unselected overlap cells;
- exact passthrough for all future-zone cells unless a later mathematical
  audit explicitly freezes a bounded exception;
- no rollout reward, success, done, object-pose, or confirmatory identity input.

If the implemented method collapses to deterministic ChunkFlow-style blending,
S2C must stop as a design failure before validation search.

## Previous-Tail Construction

At deployment, the previous-tail input must be constructed only from actions
already committed by the running policy:

1. Let `B_t` be the current frozen SmolVLA Base chunk.
2. Let `A_{t-s}` be the previously committed action chunk from Base or S2C.
3. The previous tail `T_{t-s}` is the unexecuted overlap slice of
   `A_{t-s}` that still overlaps the current chunk head.
4. If no previous committed chunk exists, S2C uses exact Base passthrough and
   records the row as no-previous-tail.

Expert future actions may be used only as training labels or diagnostics on
development partitions. They may not enter deployment inference. Future
observations, object poses, reward, success, done flags, and confirmatory
identities are forbidden inference inputs.

## Accepted Policy Order

The first serious comparison remains exactly:

1. `smolvla_base`
2. `chunkflow_overlap_proxy` or official ChunkFlow if installed and verified
3. `s2c_full`
4. `s2c_no_learned_overlap_mask_ablation`
5. `standard_lora`

The ChunkFlow proxy must be a strong transparent proxy, not fixed averaging by
default. It must implement frozen, editable, and future zone semantics,
deterministic overlap blending, first-order continuity, second-order
continuity, and the same action-validity semantics as S2C where locally
feasible.

`standard_lora` remains required because S2C trains lightweight infrastructure
on demonstrations.

## Stage 0 Headroom Gate

Stage 0 must first prove adjacent-chunk boundary headroom on development rows.
It must measure:

- current Base head versus previous committed Base/S2C tail disagreement;
- boundary jump by translation, rotation, and gripper groups;
- first-order and second-order discontinuity;
- high-frequency energy around the boundary;
- clean non-boundary retention;
- residual headroom left by the ChunkFlow proxy.

No bounded validation search, training escalation, or rollout may proceed if
adjacent Base boundary disagreement is absent or if deterministic ChunkFlow
fully explains the available correction.

## Gripper And Legitimate Discontinuity Protection

S2C may not smooth away valid gripper or contact transitions. Stage 0 must
separate translation, rotation, and gripper diagnostics. Gripper sign changes
or threshold crossings define event cells. Those cells require either exact
Base preservation or an explicitly frozen event-preserving bridge rule in the
mathematical audit.

Any method that improves smoothness by destroying event cells or by editing
future-zone actions globally must stop before validation search.

## Bounded Decoding And Resume Plan

Stage 0 may reuse existing verified Base action feature caches only when they
contain the exact neighboring replanning windows needed for overlap tests. If
adjacent windows are missing, Stage 0 may decode only a bounded development
set, with row keys:

`(split, task_suite, task_id, demo_id, window_start, stride, previous_policy_source)`

The maximum Stage 0 row budget and missing-key resume behavior must be frozen
in the mathematical audit or preregistration before any expensive command.
Completed rows may not be repeated. If adjacent chunks cannot be obtained
within budget, the outcome is `IMPLEMENTATION_OR_DATA_FAILURE`, not a
scientific kill.

## Mathematical Audit Commitments

The mathematical audit must freeze:

- chunk horizon `H`;
- replanning stride `s`;
- overlap length `K`;
- frozen, editable, and future-zone indices;
- previous-tail tensor and deployment construction;
- mask tensor, shape, initialization, support, and noncollapse gates;
- bridge target formula;
- action-group caps and clamps;
- clean-retention objective;
- boundary jump, high-frequency, first-order, and second-order metrics;
- gradient paths;
- no deterministic-action KL.

If any KL term is later proposed, it must be rejected unless its arguments are
valid distributions with direction, support, estimator, gradient flow, and
alternatives justified.

## Accepted Stop Conditions

S2C must stop before validation search for:

- no adjacent boundary headroom;
- collapsed all-zero or all-one masks;
- equivalence to deterministic ChunkFlow proxy;
- equivalence to no-mask ablation;
- global smoothing across all cells;
- future-zone edits outside the frozen protocol;
- gripper event destruction;
- invalid action semantics;
- identity or checkpoint reload failure;
- expert future-tail inference;
- reward, success, done, or confirmatory-record reads;
- attempted rescue of URF or any previous closed method.

## Current Status

No S2C implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this rebuttal.

Immediate next stage: mathematical mechanism audit before preregistration,
prototype protocol, implementation, validation search, training, or rollout.
