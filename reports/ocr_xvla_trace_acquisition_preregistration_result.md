# OCR-XVLA Bounded Trace-Acquisition Preregistration

- Decision: `OCR_TRIGGER_TRACE_ACQUISITION_PREREGISTERED`
- Natural-reset residual search is already closed as `NATURAL_RESET_SEARCH_SATURATED` at `ddbe62c2a537cda2fd84de9027334096943669ab`.
- This does not reopen SGL-XVLA or rename any closed method.
- This preserves the frozen OCR-XVLA mechanism and only tests whether its no-progress trigger is observable from legal traces.

## Frozen trace pass

- Policy: frozen official `X-VLA-Libero`.
- Task: `libero_spatial/task5`, “pick up the black bowl on the ramekin and place it on the plate”.
- Discovery residual identities: `20260727`, `20260730`, `20260733`.
- Discovery clean-retention identities: `20260731`, `20260732`.
- Held-out identities not used: `20260734`, `20260735`, `20260736`, `20260737`.
- Episode count: 5, lower than another 30-episode broad sweep.

Allowed trace fields are per-step RGB/wrist RGB, EEF proprioception, issued 7D actions, timestamps, chunk indices, and task/reset/frozen-prior metadata. Reward, done/success oracle, simulator object/contact state, privileged pose, future observation, and reset identity as a trigger feature are forbidden.

## Preregistered observability test

The attempt window starts at the first positive gripper command, with fallback to the first post-step20 EEF z-rise greater than 1.5 cm, and spans 120 steps.

`OCR_TRIGGER_OBSERVABILITY_PASS` requires all five traces to complete and at least one legal RGB/proprio feature to strictly separate residual failures from clean-retention successes above a trivial action-history-only baseline. Otherwise the one permitted reconsideration returns `OCR_TRIGGER_OBSERVABILITY_FAIL`.
