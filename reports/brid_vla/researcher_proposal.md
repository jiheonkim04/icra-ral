# BRID-VLA Researcher A Proposal

Date: 2026-07-16 KST

Decision: `BRID_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `BRID-VLA`, Base-Residual Implicit Diffusion for SmolVLA action
chunks.

Cycle: Epoch 4 Cycle 34

Previous method: `AFID-VLA`

Previous fixed result: `AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE`

AFID is preserved unchanged. No AFID repair, rescue, threshold change, proxy
change, task change, objective rescaling, or reinterpretation is allowed.

## Claim

SmolVLA may fail when a single deterministic decoded action chunk lies on a
poor local mode even though the legal observation, instruction, proprioception,
and frozen Base chunk contain enough information to define a better bounded
residual neighborhood. A Base-conditioned diffusion residual model can learn
the distribution of demonstration residual action chunks while preserving the
frozen Base action as the default behavior. The method should improve
closed-loop success only when the learned residual score field identifies a
useful bounded correction, and should otherwise return exact Base behavior.

## Closest Prior

Closest prior: Diffusion Policy.

Primary sources:

- `https://diffusion-policy.cs.columbia.edu/`
- `https://github.com/real-stanford/diffusion_policy`

Positive result: Diffusion Policy formulates visuomotor policy learning as
conditional denoising over action sequences. The official project reports
consistent outperformance across 12 tasks from four manipulation benchmarks,
with an average success-rate improvement of `46.9%`, and provides official
code, experiment configs, logs, and checkpoints.

Diffusion Policy's relevant mechanism is action-sequence denoising: a model
learns to iteratively denoise noisy action chunks conditioned on current
observations.

## What Is New

BRID is not ordinary LoRA, not a new VLA backbone, not generic behavior
cloning, and not raw Diffusion Policy attached beside SmolVLA.

BRID adds one mechanism to frozen SmolVLA:

`legal current input + frozen Base chunk -> residual diffusion score field ->
bounded residual denoising -> action-delta cap -> exact Base passthrough when
inactive`.

The mechanism is distinct from Diffusion Policy because:

- Diffusion Policy models raw action chunks directly.
- BRID models only the residual distribution around a strong frozen VLA Base
  chunk.
- BRID conditions explicitly on the frozen Base action chunk and treats zero
  residual as the identity action.
- BRID is initialized and validated so that the default policy is exact Base,
  not a newly sampled action policy.
- BRID applies translation, rotation, and gripper caps to residual edits before
  official action postprocessing.
- BRID may decline to act when residual score confidence, validation-selected
  denoising consistency, or action-validity checks fail.

LoRA or a lightweight adapter may parameterize the residual score network. The
scientific method is Base-residual implicit diffusion with identity-preserving
integration, not LoRA itself.

## Mechanism Sketch

Let `x_t = (o_t, q_t, l_t)` be the legal deployment input at replanning time
`t`, with RGB observations `o_t`, proprioception `q_t`, and task instruction
`l_t`.

The frozen SmolVLA Base produces:

`B_t = pi_base(o_t, q_t, l_t) in R^{50 x 7}`.

For a development demonstration, let:

- `E_t in R^{50 x 7}` be the aligned demonstration action chunk;
- `R_t = E_t - B_t` be the Base residual target.

BRID trains a residual denoiser:

`epsilon_theta(r_k, k, x_t, B_t) -> epsilon_hat`,

where `r_k = sqrt(alpha_bar_k) R_t + sqrt(1 - alpha_bar_k) epsilon`, diffusion
step `k` is sampled from a fixed development schedule, and
`epsilon ~ N(0, I)` has the same shape as `R_t`.

At inference, BRID starts from either zero residual plus bounded validation
noise or a validation-frozen deterministic residual seed, denoises for a small
fixed number of steps, and applies:

`Delta_t = clip_group(D_theta(x_t, B_t), rho_translate, rho_rotate,
rho_gripper)`.

The deployed action chunk is:

`A_t = B_t + g_theta(x_t, B_t, D_theta) * Delta_t`,

with `g_theta in [0, 1]` or `[0, 1]^{50 x 7}` initialized to zero. If the gate,
score confidence, denoising consistency, or action-validity audit fails, BRID
returns `A_t = B_t`.

Initialization and disk reload must satisfy:

`g_theta = 0`, `Delta_t = 0`, and `A_t = B_t`

within the frozen identity tolerance.

## Development Data

Allowed development inputs:

- existing LIBERO discovery/validation demonstrations;
- task instructions from discovery/validation partitions;
- legal RGB observations and proprioception;
- frozen SmolVLA Base chunks;
- aligned demonstration action chunks for residual supervision;
- fixed diffusion step identities and noise identities generated only from
  discovery/validation row keys.

Forbidden inputs:

- rollout reward;
- success or done flags;
- object poses or simulator privileged state;
- future observations;
- confirmatory-test task or reset identities;
- confirmatory-test failures or partial outcomes;
- demonstration action chunks at inference.

## Training Objective

All objectives operate on development partitions only.

Let `R_t = E_t - B_t` be the residual chunk, `epsilon` the sampled diffusion
noise, `k` the diffusion step, and `epsilon_theta(r_k, k, x_t, B_t)` the
predicted noise.

The default objective is:

`L = L_score + lambda_rec L_rec + lambda_clean L_clean + lambda_gate L_gate +
lambda_valid L_valid`.

Where:

- `L_score = mean(Huber(epsilon_theta(r_k, k, x_t, B_t) - epsilon))`;
- `L_rec = mean(Huber(B_t + Delta_t - E_t))` on residual-active development
  rows;
- `L_clean = mean(Huber(A_t - B_t))` on rows where the validation-frozen
  confidence rule says BRID should not intervene;
- `L_gate = mean(g_theta)`;
- `L_valid` penalizes action-bound violations after official SmolVLA
  postprocessing.

Expected gradient paths:

- `L_score` updates the residual denoiser and conditioning adapter;
- `L_rec` updates the denoiser output head and gate through `Delta_t`;
- `L_clean` updates the gate and residual head toward Base passthrough;
- `L_gate` discourages global activation;
- `L_valid` updates only trainable BRID parameters;
- frozen SmolVLA Base receives no gradients.

All terms use Huber/L1 or action-bound penalties on valid deterministic
vectors. Diffusion noise prediction is trained with a valid density/noise
model, but BRID does not compute KL divergence directly between deterministic
7D action vectors.

Before nontrivial training, BRID must estimate loss-term magnitudes and
gradient norms on a small development batch. Coefficients may be selected only
by bounded validation search.

## Evidence Partitions

`DISCOVERY`: inspect Base residual distributions, choose the residual
parameterization, build the transparent Diffusion Policy action-chunk proxy,
and debug residual/noise/action-validity health.

`VALIDATION`: select one bounded configuration from at most six total
configurations, including diffusion step count, residual cap, clean-retention
coefficient, score/loss coefficient, and adapter size.

`CONFIRMATORY_TEST`: used once only after method, configuration, checkpoint,
tasks, reset identities, metrics, thresholds, policy list, and ablations are
frozen.

No confirmatory-test outcome may be used to retune BRID.

## First Serious Comparison

The first serious comparison must include exactly:

1. `smolvla_base`
2. `diffusion_policy_action_chunk_proxy`
3. `brid_full`
4. `brid_no_base_residual_ablation`
5. `standard_lora`

Policy 2 is the closest-prior proxy. If official Diffusion Policy code cannot
be run directly under the local SmolVLA/LIBERO scaffold, the proxy must be
transparent and matched: train a raw action-chunk diffusion denoiser on the
same development data, same legal inputs, same action semantics, same split,
and comparable compute budget, but without Base-residual conditioning or exact
Base passthrough.

The key ablation removes Base-residual conditioning and zero-residual identity
integration while preserving the denoising objective, parameter budget,
training rows, and inference budget. It tests whether the Base-residual
mechanism matters beyond simply training a local diffusion action model.

Standard LoRA is required because BRID trains on demonstrations and a reviewer
could argue that ordinary adaptation with the same data and compute explains
any gain.

## Bounded Validation Search

Maximum search budget: `6` total configurations.

Allowed factors:

- residual cap scale;
- denoising step count;
- clean-retention coefficient;
- score/loss coefficient;
- adapter or LoRA rank;
- deterministic versus validation-frozen stochastic residual seed rule.

Selection score must combine validation closed-loop success when available or
the closest feasible proxy, clean retention, residual mechanism activation,
action validity, and compute overhead. Do not select purely by offline action
L2.

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
- residual targets are finite and noncollapsed across tasks and phases;
- residual headroom exists relative to frozen Base and the raw Diffusion Policy
  proxy;
- diffusion noise identities are deterministic and partition-safe;
- score target prediction beats a trivial zero-noise or mean-noise baseline on
  validation;
- initialized and disk-reloaded BRID equals Base within tolerance;
- expected BRID parameters receive finite nonzero gradients;
- frozen SmolVLA parameters receive no gradients;
- BRID differs from Base, the raw diffusion proxy, and the no-Base-residual
  ablation after a small development fit;
- edits are bounded by translation, rotation, and gripper caps;
- clean-retention rows preserve Base behavior;
- gate or intervention activation is residual-relevant rather than everywhere;
- action postprocessing remains valid.

Do not proceed to bounded validation if:

- residual targets are collapsed;
- diffusion score prediction cannot beat a trivial baseline;
- no residual headroom exists;
- the raw Diffusion Policy proxy dominates BRID;
- the no-Base-residual ablation explains the effect;
- standard LoRA explains the effect;
- the module globally changes all actions;
- clean retention fails;
- any privileged inference input or confirmatory-test identity is used.

## Expected Evidence If It Works

BRID should show:

- noncollapsed Base residual chunks on development data;
- validation score/noise prediction above trivial baselines;
- a validation improvement over Base and the raw Diffusion Policy proxy on the
  residual action-generation claim axis;
- an improvement over the no-Base-residual ablation;
- bounded action deltas with preserved action validity;
- clean retention where residual confidence is low;
- mechanism evidence tying intervention activation to useful residual states;
- no confirmatory-test tuning.

## Current Status

No BRID implementation, training, validation search, rollout, simulator
access, or confirmatory-test tuning has happened before this proposal.

Immediate next stage: Reviewer B attack on novelty, Diffusion Policy prior
boundary, residual data viability, objective scale, identity preservation, and
decisive experiment feasibility.
