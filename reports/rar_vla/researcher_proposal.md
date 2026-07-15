# RAR-VLA Researcher A Proposal

Date: 2026-07-15 KST

Method: `RAR-VLA`, Re-Anchored Autoregressive Residuals for frozen SmolVLA.

Decision: `RAR_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Closest external prior: AR-VLA, https://arxiv.org/abs/2603.10126.

Secondary priors:

- ReactVLA, https://arxiv.org/abs/2606.14255
- DSWAM, https://arxiv.org/abs/2607.04927
- ABot-M0 Action Manifold Learning, https://arxiv.org/abs/2602.11236

## Research Claim

Frozen chunk-based SmolVLA can fail when consecutive action chunks are
temporally inconsistent or stale relative to the evolving control state. A
causal action-memory residual, re-anchored whenever a new Base chunk is
observed, may improve closed-loop task success while preserving Base behavior
by default.

RAR-VLA tests whether an identity-preserving, frozen-backbone approximation of
AR-VLA's causal action expert can improve SmolVLA beyond:

1. frozen SmolVLA;
2. a transparent AR-VLA-style re-anchored expert proxy;
3. the key no-reanchor-memory ablation;
4. a simple EMA/action-history baseline.

This is not a claim that RAR reproduces official AR-VLA.

## Mechanism

At timestep `t`, let:

- `o_t`: current deployment observation containing RGB streams, proprioception,
  and language/task instruction;
- `a_base_t`: frozen SmolVLA action chunk or emitted 7D action;
- `h_t`: causal memory state built only from previous emitted actions,
  previous Base chunks, proprioception, and task identity;
- `r_t`: bounded residual action or hidden-adapter update predicted from
  `(o_t, a_base_t, h_t)`;
- `g_t`: residual gate initialized closed;
- `a_rar_t = a_base_t + g_t * r_t`.

When a new Base chunk arrives, `h_t` is re-anchored to the new chunk identity.
Between chunk refreshes, `h_t` evolves autoregressively from emitted actions.

The initial policy must be exactly or numerically indistinguishably Base:

- residual branch initialized to zero;
- gate initialized to zero or Base passthrough;
- all action deltas audited before rollout;
- Base action bounds preserved.

## Allowed Sources

At inference RAR may use only:

- current RGB observations available to SmolVLA;
- current proprioception/state;
- language or task instruction;
- current frozen SmolVLA action chunk or emitted Base action;
- previous emitted actions from the same rollout;
- previous Base chunks from the same rollout;
- internally maintained causal memory computed from those legal values.

At inference RAR may not use:

- future actions or future action segments;
- latent labels from CALA;
- future observations;
- success, reward, or failure labels;
- reset identity, manifest key, or held-out outcome;
- simulator object pose, target placement, privileged object state, or oracle
  phase labels;
- confirmatory-test outcomes for tuning.

## Supervision

Development-only supervision may use ordered demonstration records with Base
actions and target actions to learn:

- next-action or next-chunk residuals;
- chunk-boundary discrepancy targets;
- causal memory update targets;
- action discontinuity and jerk diagnostics;
- clean-retention losses.

Future target actions may be used only as labels during discovery/validation.
They are not inference inputs.

## Stage 0 Development Audit

Stage 0 must run before validation search, final training, manifest freeze, or
rollout.

Required checks:

1. Causal source gate:
   - inference feature list contains only legal current or past deployment
     values;
   - no CALA future-action latent labels are used;
   - no reset identity or confirmatory manifest information is used.

2. Action-history headroom:
   - Base has measurable temporal-discontinuity, chunk-boundary, or
     residual-prediction headroom on development identities;
   - a diagnostic upper bound shows a plausible maximum gain.

3. Residual observability:
   - RAR causal features predict residual structure above `ema_action_history`
     and simple linear-history baselines by the preregistered margin;
   - if EMA/history is strongest, the correct stop is `DESIGN_FAILURE`.

4. Identity and disruption:
   - initial action delta p95 at most `1e-6`;
   - Base action validity equals `1.0`;
   - translation, rotation, and gripper deltas are separately bounded;
   - residual activation is localized rather than global.

5. Gradient and implementation smoke:
   - expected residual and memory parameters receive finite nonzero gradients;
   - checkpoint save/reload must work before any closed-loop rollout.

Allowed Stage 0 decisions:

- `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`
- `DATA_OR_SUPERVISION_FAILURE`
- `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE`
- `DESIGN_FAILURE`
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

Stage 0 stops are pre-rollout development outcomes, not closed-loop scientific
kills.

## Bounded Validation Search

Run only if Stage 0 passes.

Maximum six configurations:

1. `rar_h4_s003_linear`: history horizon `4`, residual scale `0.03`, linear head
2. `rar_h8_s003_linear`: history horizon `8`, residual scale `0.03`, linear head
3. `rar_h16_s003_linear`: history horizon `16`, residual scale `0.03`, linear head
4. `rar_h4_s006_mlp`: history horizon `4`, residual scale `0.06`, one-hidden-layer head
5. `rar_h8_s006_mlp`: history horizon `8`, residual scale `0.06`, one-hidden-layer head
6. `rar_h16_s006_mlp`: history horizon `16`, residual scale `0.06`, one-hidden-layer head

No other architecture, horizon, scale, coefficient, seed, source variant, or
baseline may be added before confirmatory testing.

Validation score:

`S = 0.25 * residual_predictability + 0.20 * clean_retention + 0.20 * bounded_action_validity + 0.15 * localized_activation + 0.15 * ema_baseline_margin + 0.05 * efficiency`

## First Serious Comparison

Exactly five policies:

1. `frozen_smolvla`
2. `ar_vla_reanchored_expert_proxy`
3. `rar_full`
4. `rar_no_reanchor_memory_ablation`
5. `ema_action_history_baseline`

The AR proxy is a faithful transparent local proxy unless official equivalence
is independently established.

## GO And Kill Logic

RAR can become a paper candidate only if:

- `rar_full` beats frozen SmolVLA;
- `rar_full` beats the AR-style proxy on the matched claim axis;
- `rar_full` beats the no-reanchor-memory ablation;
- the EMA/action-history baseline does not explain the gain;
- clean behavior is retained;
- residual and memory diagnostics support the intended mechanism;
- no privileged inference signal is used.

Kill or stop when:

- EMA/action-history baseline matches or beats RAR;
- no discontinuity or residual headroom exists;
- residual prediction is not observable from legal causal inputs;
- the method globally changes actions rather than activating locally;
- clean retention fails;
- the method requires future actions, reset identities, success labels, object
  pose, or confirmatory outcomes at inference.

## Non-Claims

- RAR is not official AR-VLA.
- RAR is not LoRA novelty.
- RAR is not generic action smoothing.
- RAR is not adaptive chunk scheduling, frame retention, candidate ranking,
  future-action latent prediction, or output-action clipping.
- RAR does not rescue CALA, G3P, EAC, PESA, MARC, DAGR, MTF, RAC, FANG, CAVM,
  PSE, RCV, or earlier killed formulations.
