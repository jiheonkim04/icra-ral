# CVLR-XVLA Exact Scientific Status

- Campaign state: `AUTONOMOUS_CAMPAIGN_PAUSED_RESUMABLE`
- Latest frozen decision: `CVLR_XVLA_STAGE0_DESIGN_FAILURE`
- Active training or rollout worker: none
- Next authorized empirical stage: none

RIFA v1 and CVLR v1 are both archived as not Stage-A-ready. CVLR produced a strong held-out latent-reconstruction result and exact clean passthrough, but its direct dropout-time token insertion violated semantic-aware safety on every live identity. No CVLR closed-loop evidence exists, so this result neither proves closed-loop harm nor rules out the broader cross-view reconstruction family.

The current steer does not authorize Stage A, a v1 retune, a threshold change, a replacement candidate, or a return to broad search. Resumption requires a new preregistered direction that preserves the frozen RIFA and CVLR decisions.
