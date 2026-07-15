# COVI-VLA Stage 0 Adjudication

Date: `2026-07-15 KST`

Proposal hash: `338430D2C6CF1D82410C036D79102ED3F38B2367BB35B9AE2811161698A3E621`

Final decision: `COVI_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE_NO_SCIENTIFIC_KILL`

## Protocol Integrity

The frozen Stage 0 completed under its existing method, data partitions,
objective coefficients, comparator list, occlusion transform, sample counts,
and thresholds. It used `600` discovery-fit and `400` validation records,
represented `40` validation episodes, and decoded zero confirmatory-test
records. No validation search, closed-loop rollout, or confirmatory-test
tuning occurred.

The first run is preserved as `stage_0_result_invalid_v1.json`. It was
inadmissible because the executable omitted the preregistered
objective-specific gradient audit. The bounded implementation repair added
only that audit and reran the same frozen command. Historical values were not
deleted or reinterpreted.

## Decisive Failure

The repaired result is `stage_0_result.json` with decision
`IMPLEMENTATION_OR_DATA_FAILURE`:

- full COVI weighted objective-gradient ratio: `1345.9529990435792:1`;
- equal-area random-cutout control ratio: `1316.4080461878966:1`;
- frozen maximum: `100:1`;
- nonzero objectives at identity initialization: `2 / 5`;
- `L_clean`, `L_delta`, and `L_action` value and gradient: zero at the
  preregistered pretraining audit;
- checkpoint reload maximum difference: `0.0`;
- Base parameters updated: `0`;
- initial action delta p95: `0.0`;
- output-valid fraction under the frozen action bounds: `0.2`, below `1.0`.

No coefficient may be rescaled after observing this result. The failure is an
objective/integration failure of the frozen COVI configuration, not evidence
that all complementary-view methods are scientifically invalid.

## Diagnostic Evidence

The development proxy also showed no preregistered representation headroom:

- no-imagined-view normalized RMSE: `0.008841950533460463`, below the `0.02`
  practical-headroom threshold;
- COVI margin versus the strongest comparator: `-0.05313794852281253`;
- margin versus the transparent VIM proxy: `-0.030850944116179575`;
- margin versus equal-area random cutout: `0.002058002119155732`;
- episode-bootstrap interval versus the strongest comparator:
  `[-0.10168302523568187, -0.004541124240623629]`.

These values are retained as diagnostics. Because the objective and action
validity gates failed, they are not promoted to
`ROBUST_EMPIRICAL_DESIGN_FAILURE` and do not permanently kill the scientific
family.

## Continuation

The sealed one-check set is not opened: the frozen one-check path requires
valid implementation, headroom, identity, and safety gates. The six-config
validation search and closed-loop rollout are forbidden. COVI Cycle 14 closes
as a non-scientific implementation/optimization stop without rescue.

The corrected LoRA/QLoRA and minimum-sufficient-comparison governance becomes
active only after this frozen COVI Stage 0 closure. It applies to the next
unfrozen method cycle and does not alter this result or its comparator list.
