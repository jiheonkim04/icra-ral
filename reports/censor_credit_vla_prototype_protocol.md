# CensorCredit-VLA Prototype Protocol

Date: 2026-07-11 KST

CensorCredit-VLA trains two temporal trust models from short exact-state interventions: an uncensored recovered-outcome ablation and a censored prefix-credit model. At deployment it emits one action per policy step by blending with the previous action when the learned margin says the current prefix should not receive future recovery credit.

- tasks: `[('libero_spatial', 4), ('libero_10', 4)]`
- training identities: `[20260711]`
- eval identities: `[20260712]`
- variants: `['frozen_smolvla', 'vla_corrector_jump_proxy', 'simple_temporal_ema', 'uncensored_recovery_ablation', 'censor_credit_full']`

GO/KILL follows the same Route A/Route B thresholds as PhaseBarrier, with `uncensored_recovery_ablation` as the key ablation.
