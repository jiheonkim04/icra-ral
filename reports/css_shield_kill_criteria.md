# CSS-Shield Kill Criteria

Kill CSS-Shield if any of these persist after bounded diagnostics:

- no real simulator/rollout diagnostic metric can be produced,
- only synthetic offline unit metrics exist,
- full shield is equivalent to clipping-only,
- full shield safe-stops almost everything,
- utility preservation collapses,
- baseline actions are already perfect and no realistic failure cases exist,
- wrong-target and unsafe rates do not decrease,
- method requires full VLA retraining to show any signal,
- evidence would require paper-grade claims before rollout support exists.

If killed, create a CSS-Shield kill report rather than expanding planners.

## Current Gate Status After State 1.5 / State 2

Current bounded diagnostics do not trigger the immediate kill gate:

- real simulator metrics were produced,
- intended and distractor objects were resolvable from instruction text plus visible scene names,
- wrong-target metric was computable,
- full CSS-Shield beat safety-only on semantic wrong-target rate in the controlled State 1.5 gate,
- full CSS-Shield beat safety-only on semantic wrong-target rate in the 20-trial State 2 randomized batch,
- full CSS-Shield did not stop all actions.

Remaining kill/reframe checks:

- evidence is still diagnostic-only and not paper-grade,
- State 3 must verify novelty against recent VLA safety/semantic grounding work,
- later trials must confirm the failures are realistic and not an artifact of synthetic proposal construction,
- if stronger native/simulator action sources do not show semantic failures, reframe or kill the RA-L route.

