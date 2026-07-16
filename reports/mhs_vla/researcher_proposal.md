# MHS-VLA Researcher A Proposal

Date: 2026-07-16 KST

Decision: `MHS_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `MHS-VLA`, Mamba History State for Base-preserving SmolVLA.

Cycle: Epoch 4 Cycle 35

Previous method: `BRID-VLA`

Previous fixed result: `BRID_STAGE_0_NO_RESIDUAL_HEADROOM`

BRID is preserved unchanged. No BRID repair, rescue, threshold change, proxy
change, task change, memory construction change, objective change, or
reinterpretation is allowed.

## Claim

SmolVLA may fail on sequential LIBERO states where the current observation,
instruction, proprioception, and short Base action chunk are not sufficient to
disambiguate the correct stage of the episode. A deployment-observable
full-history state can encode past observations and executed actions as a
compact non-Markovian context. If that context predicts an ambiguity-relevant
residual, MHS can apply a small bounded correction around the frozen SmolVLA
Base chunk; otherwise it should return exact Base behavior.

MHS should improve closed-loop success only when the history state changes the
action distribution in states whose current-frame inputs are ambiguous. It
should not act globally, replace Base, or depend on privileged inference
signals.

## Closest Prior

Closest prior: MTIL, Mamba Temporal Imitation Learning.

Primary sources:

- `https://arxiv.org/abs/2505.12410`
- `https://arxiv.org/html/2505.12410v3`
- `https://github.com/yulinzhouZYL/MTIL`

Positive result: MTIL reports full-history state-space imitation learning that
outperforms ACT and Diffusion Policy on ACT, Robomimic, LIBERO, and real-world
sequential manipulation tasks. The official repository identifies the method as
accepted in IEEE Robotics and Automation Letters and provides an implementation
for MTIL-style agents.

MTIL's relevant mechanism is a recurrent state-space history representation.
The policy updates a compact state from the whole observation/action history
and conditions action prediction on that state plus the current observation.

## What Is New

MHS is not ordinary LoRA, not a new VLA backbone, not raw MTIL, not a generic
RNN policy, and not a diffusion residual method.

MHS adds one mechanism to frozen SmolVLA:

`legal history + current legal input + frozen Base chunk -> recurrent history
state -> ambiguity-aware residual gate -> bounded action correction -> exact
Base passthrough when inactive`.

The mechanism is distinct from MTIL because:

- MTIL trains a full imitation policy that directly predicts action chunks.
- MHS keeps SmolVLA Base frozen and uses history only to decide whether a
  bounded residual around Base is justified.
- MHS includes exact zero-residual, zero-gate Base passthrough as the default
  behavior.
- MHS requires a no-history-state ablation to test whether the recurrent
  history representation matters.
- MHS restricts inference to deployment-observable history: previous
  observations, previous executed actions, current observation/proprioception,
  instruction, and frozen Base chunk.
- MHS forbids reward, success flags, object poses, simulator states, future
  observations, demonstration actions at inference, and confirmatory-test
  identities.

LoRA or a lightweight adapter may parameterize the history encoder or residual
head. The scientific method is full-history state conditioned residual
integration, not LoRA itself.

## Mechanism Sketch

Let `H_t = ((o_1, q_1, a_1), ..., (o_{t-1}, q_{t-1}, a_{t-1}))` be the legal
deployment history before replanning step `t`, where `o` is RGB observation,
`q` is proprioception, and `a` is the actually executed 7D action. Let
`x_t = (o_t, q_t, l_t)` be the current legal input with task instruction
`l_t`.

The frozen SmolVLA Base produces:

`B_t = pi_base(x_t) in R^{50 x 7}`.

MHS encodes history with a recurrent state-space update:

`h_t = f_phi(h_{t-1}, e_phi(o_t, q_t, a_{t-1}, l_t)) in R^d`.

For a development demonstration, let:

- `E_t in R^{50 x 7}` be the aligned demonstration action chunk;
- `R_t = E_t - B_t` be the Base residual target;
- `m_t in {0, 1}` be a development-only ambiguity/usefulness label derived
  from residual headroom and history-only contrast diagnostics.

MHS predicts:

- residual proposal `Delta_theta(h_t, x_t, B_t) in R^{50 x 7}`;
- gate `g_theta(h_t, x_t, B_t) in [0, 1]` or `[0, 1]^{50 x 7}`.

The deployed chunk is:

`A_t = B_t + g_theta(h_t, x_t, B_t) * clip_group(Delta_theta, rho_trans,
rho_rot, rho_gripper)`.

At initialization and after disk reload:

`g_theta = 0`, `Delta_theta = 0`, and `A_t = B_t`

within the frozen identity tolerance.

## Development Data

Allowed development inputs:

- existing LIBERO discovery and validation demonstrations;
- task instructions from discovery and validation partitions;
- legal RGB observations and proprioception;
- previous executed actions in the same trajectory;
- frozen SmolVLA Base chunks;
- aligned demonstration action chunks for training targets;
- deterministic history-window identities generated from development row keys.

Forbidden inputs:

- rollout reward;
- success or done flags;
- object poses or simulator privileged state;
- future observations relative to the inference step;
- demonstration action chunks at inference;
- confirmatory-test task or reset identities;
- confirmatory-test failures or partial outcomes.

## Training Objective

All objectives operate on development partitions only.

For a development row, let `B_t, E_t, R_t, h_t, Delta_t, g_t` be as defined
above, and let:

`A_t = B_t + g_t * clip_group(Delta_t)`.

The default objective is:

`L = L_res + lambda_gate L_gate + lambda_hist L_hist + lambda_clean L_clean +
lambda_valid L_valid`.

Where:

- `L_res = mean(Huber(A_t - E_t))` on residual-active development rows;
- `L_gate = BCE(g_t, m_t)` or focal BCE if `m_t` is imbalanced;
- `L_hist = mean(Huber(z_theta(h_t) - z_target_t))` for a simple
  development-only history contrast target, such as residual-mode or phase
  proxy labels that are unavailable and unnecessary at inference;
- `L_clean = mean(Huber(A_t - B_t))` on clean-retention rows where the frozen
  validation rule says MHS should not intervene;
- `L_valid` penalizes action-bound violations after official SmolVLA
  postprocessing.

Expected gradient paths:

- `L_res` updates the history-conditioned residual head, gate, and any trainable
  history adapter;
- `L_gate` updates the gate and history features used by the gate;
- `L_hist` updates the history encoder only through deployment-observable
  inputs;
- `L_clean` updates the gate and residual head toward Base passthrough;
- `L_valid` updates only trainable MHS parameters;
- frozen SmolVLA Base receives no gradients.

All terms use Huber/L1/BCE penalties on valid deterministic variables or valid
binary targets. MHS does not compute KL divergence directly between
deterministic 7D actions and does not assume SmolVLA flow vectors are
probability distributions.

Before nontrivial training, MHS must estimate loss-term magnitudes and gradient
norms on a small development batch. Coefficients may be selected only by
bounded validation search.

## Evidence Partitions

`DISCOVERY`: inspect Base history ambiguity, residual distributions,
history-window construction, ambiguity labels, and mechanism observability.

`VALIDATION`: select one bounded configuration from at most six total
configurations, including history horizon, hidden dimension, residual cap,
clean-retention coefficient, gate threshold, and adapter size.

`CONFIRMATORY_TEST`: used once only after method, configuration, checkpoint,
tasks, reset identities, metrics, thresholds, policy list, and ablations are
frozen.

No confirmatory-test outcome may be used to retune MHS.

## First Serious Comparison

The first serious comparison must include exactly:

1. `smolvla_base`
2. `mtil_history_state_proxy`
3. `mhs_full`
4. `mhs_no_history_state_ablation`
5. `standard_lora`

Policy 2 is the closest-prior proxy. If official MTIL code cannot be run
directly under the local SmolVLA/LIBERO scaffold, the proxy must be transparent
and matched: train a history-state action-chunk policy or residual proxy using
the same development data, legal inputs, action semantics, split, and comparable
compute budget, but without SmolVLA Base passthrough as the claimed mechanism.

The key ablation removes recurrent history state while preserving current
observation, instruction, Base chunk, residual head, gate architecture,
parameter budget, training rows, and inference budget. It tests whether the
history-state mechanism matters beyond ordinary current-frame residual
adaptation.

Standard LoRA is required because MHS trains on demonstrations and a reviewer
could argue that ordinary adaptation with the same data and compute explains
any gain.

## Bounded Validation Search

Maximum search budget: `6` total configurations.

Allowed factors:

- history horizon or compression schedule;
- history-state dimension;
- residual cap scale;
- clean-retention coefficient;
- gate threshold;
- adapter or LoRA rank.

Selection score must combine validation closed-loop success when available or
the closest feasible proxy, clean retention, history mechanism activation,
action validity, full-versus-ablation distinction, and compute overhead. Do not
select purely by offline action L2.

After selection, freeze the single configuration, checkpoint, policy list,
metrics, thresholds, and confirmatory manifest before any confirmatory use.

## Stage 0 Development Audit

Stage 0 is development-only. It is not a closed-loop scientific result.

Required checks:

- proposal hash and source artifacts;
- discovery/validation/test identity separation;
- no reward/success/done/object-pose/confirmatory record access;
- Base chunks and demonstration chunks align under official 7D action
  semantics;
- history windows contain enough valid previous observations/actions;
- ambiguity/usefulness labels are finite, noncollapsed, and task-covered;
- a history probe predicts the ambiguity/usefulness target above trivial
  majority and current-frame-only baselines;
- diagnostic headroom exists for history-conditioned residual intervention;
- MTIL proxy leaves meaningful residual failure relative to the MHS claim axis;
- initialized and disk-reloaded MHS equals Base within tolerance;
- expected MHS parameters receive finite nonzero gradients;
- frozen SmolVLA parameters receive no gradients;
- MHS differs from Base, MTIL proxy, and no-history ablation after a small
  development fit;
- edits are bounded by translation, rotation, and gripper caps;
- clean-retention rows preserve Base behavior;
- gate activation is concentrated in history-ambiguous states rather than
  everywhere;
- action postprocessing remains valid.

Do not proceed to bounded validation if:

- history labels or residual targets are collapsed;
- no current-frame ambiguity or history headroom exists;
- history targets cannot be inferred above trivial baselines from legal inputs;
- MTIL proxy dominates MHS on the matched claim axis;
- the no-history ablation explains the effect;
- standard LoRA explains the effect;
- the module globally changes all actions;
- clean retention fails;
- any privileged inference input or confirmatory-test identity is used.

## Expected Evidence If It Works

MHS should show:

- noncollapsed history-window coverage across development tasks;
- a history probe above trivial and current-frame-only baselines;
- validation improvement over Base and the MTIL history-state proxy on the
  matched claim axis;
- improvement over the no-history-state ablation;
- bounded action deltas with preserved action validity;
- clean retention where history ambiguity is absent;
- mechanism evidence tying gate activation to history-ambiguous states;
- no confirmatory-test tuning.

## Current Status

No MHS implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this proposal.

Immediate next stage: Reviewer B attack on novelty, MTIL prior boundary,
history-label viability, objective scale, identity preservation, and decisive
experiment feasibility.
