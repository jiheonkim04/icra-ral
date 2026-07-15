# RAR-VLA Researcher A Rebuttal

Date: 2026-07-15 KST

Method: `RAR-VLA`

Proposal: `reports/rar_vla/researcher_proposal.md`

Proposal hash: `723C16C3885A974E2CA12D90BC36267FA6E86827AC9D2A1E0E0E475E16FB0E56`

Reviewer attack: `reports/rar_vla/reviewer_attack.md`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Rebuttal decision: `RAR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

## Accepted Narrowed Novelty

Researcher A accepts the narrowed novelty:

RAR-VLA is only a frozen-SmolVLA, identity-preserving, AR-style residual memory
adapter with development-gated source legality, residual-headroom, and
EMA-history baseline checks.

RAR will not claim broad novelty for:

- autoregressive action experts;
- action memory;
- re-anchoring;
- action smoothing;
- action chunking;
- masked chunk correction;
- temporal action selection;
- LoRA or SmolVLA attachment.

## Accepted Closest-Prior Status

`ar_vla_reanchored_expert_proxy` remains a transparent local proxy unless exact
official AR-VLA code/checkpoint equivalence is independently established.

RAR will not be described as official AR-VLA.

## REMAC And TAS Distinction

Researcher A accepts Reviewer B's warning that REMAC and TAS may be closer than
the proposal initially emphasized.

RAR must remain distinct from REMAC by not becoming:

- masked action chunk training;
- prefix-preserved sampling;
- a chunk repair objective whose only mechanism is intra-chunk correction.

RAR must remain distinct from TAS by not becoming:

- a cached chunk selector;
- a weighted blend over recent action chunks;
- a reactivity/consistency selector without a causal residual memory state.

Stage 0 and the mathematical audit must separately report:

- inter-chunk discontinuity diagnostics;
- intra-chunk inconsistency diagnostics;
- causal memory state inputs;
- whether re-anchoring changes predictions beyond EMA/history.

If implementation collapses to REMAC, TAS, or smoothing, stop before rollout.

## EMA Action-History Baseline

Researcher A accepts `ema_action_history_baseline` as the mandatory simple
reviewer-killer baseline.

Before validation search:

- RAR residual predictability must beat EMA and linear-history baselines by the
  preregistered margin;
- the full method must preserve exact Base passthrough at initialization;
- residual activation must be localized, not global;
- translation, rotation, and gripper deltas must be separately bounded.

If EMA/history is strongest in Stage 0, classify the result as `DESIGN_FAILURE`.

## Source Legality

RAR inference may use only:

- current RGB/proprioception/language or task instruction;
- current frozen SmolVLA Base action chunk;
- previous emitted actions;
- previous Base chunks;
- internally maintained causal memory derived from those legal values.

RAR inference may not use:

- future actions or future action segments;
- CALA latent labels;
- success/reward/failure labels;
- reset identity or manifest key;
- object pose, target placement, privileged simulator state;
- confirmatory-test outcomes.

## Identity-Preserving Integration

Researcher A accepts:

- zero-initialized residual branch;
- closed gate or Base-passthrough initialization;
- clean-retention gate before rollout;
- checkpoint save/reload before Stage A;
- no validation search, training, manifest freeze, or rollout before Stage 0
  passes.

## Remaining Disagreement

No unresolved disagreement remains before mathematical audit.

The decisive pre-rollout question is empirical and bounded:

Does legal causal action memory predict a useful bounded residual beyond
EMA/linear action-history baselines while preserving Base behavior?

If not, RAR stops before rollout.
