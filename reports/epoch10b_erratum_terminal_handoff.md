# Epoch 10B manifest/adjudicator erratum terminal handoff

Terminal state: `EPOCH10B_ERRATUM_MANIFEST_COMPLETED_CERTIFIED_STAGE0_AUTHORIZED`

Branch: `codex/epoch10b-manifest-adjudicator-erratum`

## Decision

The frozen audit selected Path B: choice-free completion of one pre-registered primary panel. The original terminal `EPOCH10B_ICAE_ASSAY_INVALID_ROUTE_CLOSED` remains immutable and visible. The versioned superseding adjudication is `reports/epoch10b_erratum_superseding_adjudication.json`; it does not rewrite or conceal the original failure.

The raw 1,287-row log contained 16 complete reverse panels, not 15. The adjudicator converted the 128 manifest entries to a `state_id` dictionary. Spatial/task_0/demo_8 phase slots 6 and 7 had both serialized to frame 61; the later phase-7 record overwrote the earlier record and changed the joined `reverse_order_duplicate` flag from true to false. Joining that flag by logical OR counts the already executed panel and changes no scientific value.

An adjudicator-only correction was not sufficient because the original protocol required at least 128 candidate mechanics states and the raw log had only 127 distinct primary states. The preserved collision-corrected pre-probation preregistration (SHA-256 `2e5f2fe58153ed7dad17a88a11c41377275b7ebd56e7f71d989b30d073b8b6b6`) fixed exactly `libero_spatial|task_0|demo_8|frame_60|transport_goal`, with registered seed `1857632994`, state hash `83a767...9971`, expert-action hash `681bed...a9f7`, and nine primary keys. Its deterministic backward collision rule reads no score.

The Path B decision was frozen and hashed before source correction or execution. Its file SHA-256 is `cab840da177eaf99a9fa9f34b9814adc7464a273f512215bb7566ea7468a64a0`.

## Execution integrity

The original raw log remains 15,618,820 bytes with SHA-256 `a2f2992d03fae52177408f057ba311b4f522b3955d91ba78dcd74a165e55ced7`.

The erratum log is `runs/epoch10b_manifest_erratum/frame60_primary_panel.jsonl`, 9 rows, SHA-256 `e7f36aa798460d7a268d34c09b9ed4a401b9e0ebd7566298e8a572480b0fe03f`. It contains the exact frozen control order and no reverse row. All nine keys are unique, all rows are valid and finite, registered-state restore L2 is 0, all environments called `close`, and no row has an error.

The first guard attempt encountered a PowerShell integer-width error after four rows; it shut down WSL and left the fifth key unmaterialized. The pending marker was preserved in `infrastructure_attempts.jsonl`, no scientific zero was assigned, and the runner resumed exactly the five missing keys. A later monitor-aggregation property-access error occurred only after all nine rows were complete. No completed key was rerun. The final telemetry-only guard pass finalized the resource record from native wrapper samples plus every row's embedded before/after resource sample.

- Peak host RAM: 62.8621% (frozen soft threshold 80%).
- Peak WSL memory used: 1,932,533,760 bytes.
- Peak WSL swap used: 0 bytes.
- Peak GPU VRAM used: 2,224 MiB.
- Material NTFS/disk/WHEA/Kernel-Power events: 0.
- Protected rollout and adapter ledgers: unchanged; the adapter aggregate reproduces `12b46e...` under its native Windows case-insensitive sort.
- WSL teardown: unconditional; Ubuntu-22.04 is stopped.
- CHKDSK: still `UNAVAILABLE_INCONCLUSIVE`, not revisited and not used as evidence.

## Superseding re-adjudication

The union is logical only: 1,287 immutable original rows plus 9 separate erratum rows. The frozen expected ledger and observed union both contain 1,296 unique keys, with 0 missing and 0 extra.

- Primary: 128 states, 1,152 rows.
- Reverse: 16 states, 144 rows.
- Valid/finite: 1,296/1,296; environment close called: 1,296/1,296.
- Nominal twins: 64/64 within `1e-8`; maximum state L2 0; 16 passing pairs per suite.
- Reverse order: 16/16 panels; maximum matched L2 0.
- Selected original gate: H=4, bounded expert recovery cost.
- Nominal duplicate Spearman: 0.9999999999999998; ICC: 1.0.
- Responsive states: 114/128, with Spatial/Object/Goal/10 counts 30/24/30/30.
- Harmful-minus-nominal grouped mean: 0.0225665744; unchanged 95% cluster-bootstrap interval [0.0157758237, 0.0309970922].
- Medium-minus-small descriptive mean: 0.0192420123; 95% interval [0.0154499978, 0.0240277465].

All H=4, H=8, and H=16 twin gates pass; at least one frozen endpoint is eligible at each horizon. H=4 is selected by the unchanged lowest-passing-horizon rule. Full per-horizon grouped contrasts, confidence intervals, responsiveness counts, and twin statistics are in the superseding adjudication and mechanics reanalysis.

This procedural repair is not new empirical support for ICAE checkpoint ranking. It certifies only the original mechanics assay. The original invalid run and this manifest/adjudicator erratum must both be disclosed in any reproducibility record or paper.

## Sealed continuation

The exact original continuation instructions were recovered from `C:\Users\jiheo\Downloads\epoch10b_icae_fresh_controller_ral_continuation_prompt.md`, SHA-256 `24c2198d83ea262ff4133ffe7d44d63af65bd7ab93f237fdebcd4a47aeaefa66`. No replacement protocol was invented.

Checkpoint actions, checkpoint scores, adapter inference outcomes, closed-loop success labels, development results, held-out validation results, and confirmation results remained sealed throughout this erratum. The named terminal state authorizes the original hash-pinned Stage 0+ continuation under those instructions and unchanged thresholds. It does not itself establish `EPOCH10B_STAGE0_GO`, a prospective paper result, or a paper package.
