# A2C2 official-action-semantics continuation authority

Date: `2026-07-19 KST`

Authority SHA-256: `CDC674DB9E0EFDC85F3529FA7387D4E3A9BD31DF91A66B0D9B87C1279DA6C0B0`

Active state: `A2C2_OFFICIAL_ACTION_SEMANTICS_CORRECTION_AUTHORIZED`

The historical `CORRECTED_A2C2_EVALUATION_INVALID` record remains unchanged
as `HISTORICAL_LOCAL_STRICT_RAW_BOUND_GATE_RESULT`. This authority permits
exactly one evaluation-semantics correction: validity follows the released
evaluator and native LIBERO/robosuite controller path instead of treating any
pre-controller `[-1,1]` exceedance as automatically invalid.

This is not method rescue. Checkpoints, model graph, tasks, resets, delays,
timeouts, success criteria, and scientific decision rules remain unchanged.
No external wrapper clip is permitted.
