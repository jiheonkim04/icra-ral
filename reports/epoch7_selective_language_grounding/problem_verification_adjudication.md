# Epoch 7 selective-language-grounding problem verification

Decision date: 2026-07-20

Status: `PROBLEM_VERIFIED_STRONG_COMPARATOR_RESIDUAL`.

This status authorizes a bounded method-overlap and supervision-legality audit. It does not authorize training, Ours rollouts, confirmatory identities, or a paper claim.

## Frozen evidence

The unmodified X-VLA Base achieved 30/30 under canonical instructions and 19/30 under matched meaning-preserving paraphrases. The 36.7-point gap contains 11 canonical-success/paraphrase-failure pairs across seven tasks and all three paraphrase families. The identical canonical instruction recovers all 11 failures, establishing legal language headroom.

The lexical canonicalizer achieved 24/30. The frozen 22.7M-parameter MiniLM semantic canonicalizer achieved 25/30, a 20-point gain over raw paraphrases, while leaving five failures and a 16.7-point residual to canonical. All five failures were executed; none succeeded under the incorrectly retrieved task instruction. Exact reuse of the other 25 canonical Base episodes is valid because every execution input and seed is identical.

The reference-aligned CAG-TF Prior was correctly action-connected but not competent: 14/30 canonical and 11/30 paraphrase, with a 53.3-point canonical retention loss. It is retained as a negative external-method comparator, not the strong comparator supporting authorization.

## Gate-by-gate adjudication

1. Base competence: pass (30/30 canonical, ten tasks, finite executed actions).
2. Repeated claim-specific gap: pass (36.7 points; 11 pairs).
3. Multiple identities/tasks: pass (seven tasks; all families).
4. Relevant Prior/comparator correctly implemented: pass via the MiniLM Control; CAG itself is relevant and correctly implemented but empirically incompetent.
5. Strong-comparator residual: pass narrowly (five pairs; 16.7 points).
6. Legal recoverable headroom: pass (30/30 canonical; no privileged inference state).
7. Infrastructure/normalization/reset explanation: rejected by exact matched pairing, official success, finite actions, zero exceptions, one-live-environment execution, and zero swap.
8. Plausible method-level mechanism after overlap audit: not yet passed; this is the next hard gate.

## Skeptical reviewer view

The strongest simple Control already explains six of 11 Base failures and reaches 83.3%. The remaining instructions are unusually ambiguous, including indirect hints such as “The stove is still clear” and scene-dependent phrases such as “The wine bottle is still out.” A reviewer may reasonably argue that the residual is semantic retrieval or benchmark underspecification, not a VLA control contribution. RobustVLA, RoVLA, RSS, CAST, CAG, ProGAL-VLA, and step-wise language alignment occupy most generic solutions. Raw task demonstrations are not state-matched across intents, so a counterfactual action target cannot be invented.

## Adjudicator boundary

Proceed only to one-to-three bounded mechanism specifications and a focused equation/supervision/inference overlap matrix. A mechanism is eligible only if it:

- uses real retained demonstration supervision without synthetic or unverified counterfactual actions;
- models both within-intent equivalence and between-intent selectivity;
- is not plain augmentation, action consistency, retrieval, prompting, CAG, or a renamed archived route;
- changes executed actions through a locally trainable component;
- differs from every direct prior by at least two major dimensions;
- has a cheap Stage 0 falsifier and can beat the 25/30 semantic Control while retaining the 30/30 Base standard condition.

If no mechanism clears this boundary, rotate the thesis; do not weaken the contribution sentence.
