# ECHO Horizon And Success Semantics

## Previous Gate

- previous decision: `NO_ECHO_CANDIDATE_HEADROOM`
- previous success metric: `CandidateRecord.success was not downstream task success. The first gate set it to compatibility_score(realized_effect, phase) > 0.05 after a four-step intervention.`
- policy continuation after candidate: `False`
- materially better definition: `local realized-effect compatibility: oracle.success or oracle.compatibility > default.compatibility + 0.25`
- classification: `SHORT_HORIZON_LOCAL_EFFECT_PROXY_WITHOUT_CONTINUATION_INSUFFICIENT`

## Final Gate

- candidate-only effect horizons: `[4, 8, 16]`
- downstream success metric: `official LIBERO task success after candidate intervention plus frozen SmolVLA continuation to normal bounded episode termination`
- continuation horizon: `16`
- local physical progress is diagnostic only; final GO requires downstream official task success.
