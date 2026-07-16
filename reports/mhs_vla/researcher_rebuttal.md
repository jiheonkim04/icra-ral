# MHS-VLA Researcher A Rebuttal

Date: 2026-07-16 KST

Decision: `MHS_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Proposal: `reports/mhs_vla/researcher_proposal.md`

Reviewer attack: `reports/mhs_vla/reviewer_attack.md`

Proposal hash:
`BBDF67AE3EC4BD9D025707A8BB3A5008BAB5EB5C691D02D44516157802A87BF3`

## Response

Researcher A accepts Reviewer B's conditional pass in full.

The MHS claim is narrowed to:

`frozen SmolVLA Base + deployment-observable recurrent history state +
identity-initialized bounded residual gate`.

MHS will not claim broad novelty for Mamba imitation learning, full-history
policies, memory in VLAs, stateful policies, or LoRA adaptation. The publishable
axis, if any, is whether history-state residual integration can improve a
strong frozen VLA while preserving exact Base behavior by default.

## Accepted Conditions

1. Novelty is narrowed to Base-preserving history-state residual integration.
2. MTIL or a transparent MTIL proxy remains policy 2 in the first serious
   comparison.
3. The exact history-window construction will be frozen before Stage 0.
4. The ambiguity/usefulness label `m_t` and history target `z_target_t` will be
   frozen before training.
5. Label and residual noncollapse across tasks is required before training.
6. History predictability must beat trivial and current-frame-only baselines.
7. `mhs_no_history_state_ablation` is the key ablation.
8. `standard_lora` is the single simple reviewer-killer baseline.
9. Exact Base passthrough is required at initialization and disk reload.
10. Bounded action deltas and clean retention are required before rollout.
11. Rewards, success flags, done flags, object poses, simulator state, future
    observations, demonstration actions at inference, and confirmatory-test
    identities are forbidden.
12. Failed label, headroom, or history-observability gates will be classified
    as `DATA_OR_SUPERVISION_FAILURE` or `NO_USABLE_HEADROOM`, not as a
    closed-loop scientific result.

## Clarified Prior Boundary

Closest prior remains MTIL.

If official MTIL code cannot be run in the local SmolVLA/LIBERO scaffold, the
comparison policy will be named `mtil_history_state_proxy`, not official MTIL
reproduction. The proxy must preserve MTIL's essential mechanism: recurrent
state-space history encoding and history-conditioned action prediction from
demonstration supervision.

MHS differs only by keeping SmolVLA Base fixed and using history state as a
selective residual/gate rather than replacing the full policy.

## Clarified Data Rules

MHS may use only deployment-observable history:

- previous observations;
- previous proprioception;
- previous executed actions;
- current observation and proprioception;
- instruction;
- frozen SmolVLA Base chunk.

Training targets may use aligned demonstration actions on discovery and
validation identities. Inference may not use demonstration actions.

The Stage 0 audit must prove zero train/validation/test overlap and must report
duplicate-key checks before any result is accepted.

## Clarified Objective Rules

The mathematical audit must freeze:

- all tensor shapes;
- exact `m_t` construction;
- exact `z_target_t` construction;
- loss-term formulas;
- loss scales and gradient paths;
- coefficient search bounds;
- identity tolerance;
- action caps;
- no-history ablation construction;
- MTIL proxy construction;
- standard-LoRA construction.

No deterministic-action KL is permitted.

## Clarified Stop Classes

Before rollout, the following stop classes apply:

- collapsed or privileged labels: `DATA_OR_SUPERVISION_FAILURE`;
- no history ambiguity/headroom: `NO_USABLE_HEADROOM`;
- nonacting gradients or wrong checkpoint: `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- global gate activation or destructive action deltas: `DESIGN_FAILURE`;
- current-frame or no-history ablation explains MHS: `KEY_COMPONENT_NOT_USEFUL`;
- standard LoRA explains MHS: `SIMPLE_BASELINE_EXPLAINS_METHOD`.

Only a valid frozen closed-loop comparison can support a scientific method
decision.

## Next Stage

Proceed to mathematical mechanism audit. No MHS implementation, training,
validation search, rollout, simulator access, or confirmatory-test tuning has
happened before this rebuttal.
