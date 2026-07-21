# Epoch 9D Existing-Trace Causal Diagnostic

Evidence: `RETROSPECTIVE_DEVELOPMENT_ONLY_DIAGNOSIS`

Decision: `FREEZE_ORIGINAL_PRIMARY_SCORE_AND_RUN_EXACT_STATE_MASS_SWAP_CAUSAL_PANEL`

Paper: `PAPER_NOT_AUTHORIZED`

No new simulator outcome, validation identity, or confirmation identity was accessed.

## Fixed primary response score

The primary score is the original frozen back-slot RGB response threshold. From the exact five-step response window, the back response is the peak positive expected-axis displacement reconstructed from ordinary RGB after subtracting the median of the three immediately preceding stored steps. The frozen threshold is `0.005219466062047 m`. Candidate scores are `front = threshold - back_response` and `back = back_response - threshold`; the smaller score predicts heavier. No secondary score is frozen and no neural model is used.

## Front/back asymmetry

The original panel is 12/12 when the front candidate is heavy and 10/12 when the back candidate is heavy. This is structural: every light-back response exceeds the fixed threshold, but two heavy-back responses also exceed it. Repair3 preserves mechanics yet falls to 12/12 front-heavy and 7/12 back-heavy because five heavy-back responses cross the unchanged absolute threshold under the new interior geometry. The raw miss rows, contact timing, controller error, lane margin, and response dynamics are in `reports/epoch9d_existing_trace_causal_diagnostic.json`.

## Residualization and observability

Across all 64 historical dynamic-nudge scenes, the unadjusted back-heavy-minus-back-light response contrast is `-0.008835 m`, with a source-demo-group bootstrap 95% interval `[-0.009568, -0.007937] m`; smaller response under heavy mass is the expected direction. After adjusting for position, lane, order, initial RGB localization, contact timing, admissible controller error, and campaign effects, the mass coefficient is `-0.009119 m`, with grouped interval `[-0.009389, 0.000441] m`. The leave-one-reset-group-out residual heavy-below-light AUC is `0.868`. These are retrospective diagnostics, not confirmation.

The RGB-derived peak agrees with the simulator-only displacement audit at Pearson `r = 0.976` and mean absolute error `0.000998 m`. The primary score itself uses no simulator field. Simulator displacement is evaluation-only and is not controlled away because displacement is the hypothesized physical signal.

## Oracle failure attribution

Original and Repair3 oracle failures are not assigned to a single cause by success count alone. Each row is classified using post-probe target lane margin and lift/release progress. All three original frozen-panel failures and all eight Repair3 failures are completion-only under this rule; one failure in the smaller development repairs implicates both probe state and completion. All continuous shifts, margins, lift heights, stage flags, and categories are retained in the JSON report.

## Causal implication

The retrospective signal remains positive enough to justify the preregistered exact-state intervention, but the absolute-threshold geometry sensitivity prevents a causal claim from historical panels. The original primary score and response window are therefore frozen unchanged for the 16-base-state, 32-assignment mass-swap panel. The secondary score cannot rescue a failure because none is frozen.
