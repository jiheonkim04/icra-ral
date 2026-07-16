# LCG-VLA Researcher A Proposal

Date: 2026-07-16 KST

Decision: `LCG_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `LCG-VLA`, Language-Contrastive Guidance for Base-preserving SmolVLA
actions.

Cycle: Epoch 4 Cycle 32

Previous method: `S2C-VLA`

Previous fixed result: `S2C_STAGE_0_DATA_OR_SUPERVISION_FAILURE`

S2C is preserved unchanged. No S2C repair, rescue, threshold change, proxy
change, task change, or reinterpretation is allowed.

## Claim

SmolVLA may fail on visually familiar but linguistically different situations
because the decoded action chunk follows a vision shortcut more strongly than
the instruction. A frozen-Base, identity-initialized language-contrast gate can
route bounded action-cell edits only where original-instruction versus
language-null action contrast indicates that language conditioning matters,
while preserving Base exactly when the contrast is absent or unreliable.

## Closest Prior

Closest prior: Counterfactual Action Guidance.

Primary source: `https://arxiv.org/abs/2602.17659`

Positive result: the paper introduces LIBERO-CF for counterfactual VLA
language following and reports that Counterfactual Action Guidance improves
language-following accuracy and task success on under-observed LIBERO-CF tasks,
with additional real-world counterfactual failure reductions.

CAG's relevant mechanism is a dual-branch comparison between a standard
language-conditioned VLA policy and a language-unconditioned vision-action
branch. That comparison is used at inference to reduce reliance on visual
shortcuts.

## What Is New

LCG is not ordinary LoRA, not a new VLA backbone, not counterfactual label
augmentation, and not a direct CAG reimplementation.

LCG adds one mechanism to frozen SmolVLA:

`original-instruction Base chunk + language-null Base chunk -> language
contrast features -> identity-initialized action-cell gate -> bounded residual
edits -> Base passthrough elsewhere`.

The mechanism is distinct from CAG because:

- CAG is primarily a dual-branch inference-time guidance rule.
- LCG learns a deployment-observable gate from existing demonstrations, rather
  than applying the language contrast directly everywhere.
- LCG keeps the original SmolVLA Base action chunk as the default action and
  requires exact Base passthrough at initialization, after disk reload, and
  whenever the gate is inactive.
- LCG edits individual time/action cells under group caps rather than replacing
  or globally steering the entire action chunk.
- LCG uses no reward, success flag, done flag, object pose, future observation,
  or confirmatory-test identity at training or inference.

LoRA or a lightweight adapter may parameterize the gate or residual head. The
scientific method is language-contrastive Base-preserving action gating, not
LoRA itself.

## Mechanism Sketch

Let `x_t = (o_t, q_t, l_t)` be the legal deployment input at replanning time
`t`, with observation `o_t`, proprioception `q_t`, and instruction `l_t`.

Let `l_null` be the frozen language-null instruction used only to remove
instruction-specific conditioning while preserving the same observation and
proprioception.

The frozen SmolVLA Base produces:

- `B_t = pi_base(o_t, q_t, l_t) in R^{50 x 7}`;
- `N_t = pi_base(o_t, q_t, l_null) in R^{50 x 7}`.

LCG forms a language-contrast feature:

`C_t = group_norm(B_t - N_t)`.

The LCG adapter predicts:

- a residual proposal `Delta_theta(x_t, B_t, N_t, C_t) in R^{50 x 7}`;
- a bounded edit gate `G_theta(x_t, B_t, N_t, C_t) in [0, 1]^{50 x 7}`.

The action chunk is:

`A_t = B_t + G_theta * clip_group(Delta_theta, rho_translate, rho_rotate,
rho_gripper)`.

Initialization must satisfy:

`G_theta = 0` and `A_t = B_t`

within the frozen identity tolerance before any training and after disk reload.

The gate is allowed to activate only when deployment-observable language
contrast is nontrivial. If `B_t` and `N_t` are effectively identical, LCG must
default to Base.

## Development Data

Allowed development inputs:

- existing LIBERO discovery/validation demonstrations;
- original task instructions from discovery/validation partitions;
- legal language-null instructions;
- optional counterfactual instruction swaps drawn only from discovery and
  validation task text, never from confirmatory-test identities;
- frozen SmolVLA Base chunks under original and null instructions;
- observation-derived SmolVLA features, proprioception, current Base chunk,
  and demonstration action chunks.

Forbidden inputs:

- rollout reward;
- success or done flags;
- object poses or simulator privileged state;
- future observations;
- confirmatory-test task or reset identities;
- confirmatory-test failures or partial outcomes;
- language alternatives mined from confirmatory-test labels.

## Training Objective

Let `E_t in R^{50 x 7}` be the demonstration action chunk aligned to `B_t`.

Let `R_t = E_t - B_t`.

Let `M_lang` be a development-only language-contrast mask derived from
`C_t`, with noncollapsed thresholds selected on validation only.

The default objective is:

`L = L_res + lambda_clean L_clean + lambda_gate L_gate + lambda_valid L_valid`.

Where:

- `L_res = mean(M_lang * Huber(A_t - E_t))`;
- `L_clean = mean((1 - M_lang) * Huber(A_t - B_t))`;
- `L_gate = mean(G_theta)`;
- `L_valid` penalizes action-bound violations after official SmolVLA
  postprocessing.

All terms operate on deterministic action vectors with Huber/L1 distances.
There is no KL divergence between deterministic 7D actions.

Before any nontrivial training, term magnitudes and gradient norms must be
estimated on a small development batch. Coefficients may be chosen only through
the bounded validation search.

## Evidence Partitions

`DISCOVERY`: inspect Base/null contrast, construct language-contrast masks,
debug label health, inspect failures, and build transparent CAG proxy.

`VALIDATION`: choose one bounded configuration from at most six configurations,
including any gate threshold, residual cap, clean-retention coefficient, and
adapter size.

`CONFIRMATORY_TEST`: used once only after method, configuration, checkpoint,
tasks, reset identities, metrics, thresholds, policy list, and ablations are
frozen.

No confirmatory-test outcome may be used to retune LCG.

## First Serious Comparison

The first serious comparison must include exactly:

1. `smolvla_base`
2. `counterfactual_action_guidance_proxy`
3. `lcg_full`
4. `lcg_no_language_contrast_ablation`
5. `standard_lora`

Policy 2 is the closest prior. If official CAG code or checkpoint is not
locally available, the proxy must be transparent: use the same frozen SmolVLA
Base, the same original and language-null chunks, and a predeclared
training-free guidance form such as:

`A_CAG = B_t + beta * clip_group(B_t - N_t, rho_translate, rho_rotate,
rho_gripper)`.

`beta` and caps must be selected on validation only, not confirmatory test.

The key ablation removes language contrast from the gate input while keeping
parameter count, labels, optimizer budget, and action caps matched.

Standard LoRA is required because LCG trains a small module on demonstrations.

## Stage 0 Development Audit

Stage 0 is development-only. It is not a closed-loop scientific result.

Required checks:

- proposal hash and source artifacts;
- discovery/validation/test identity separation;
- no reward/success/done/object-pose/confirmatory record access;
- original/null Base chunks decode and align with action semantics;
- Base/null contrast is finite and noncollapsed across tasks and phases;
- demonstration residual targets are finite and noncollapsed;
- language-contrast mask is not all-zero or all-one;
- CAG proxy leaves residual headroom for LCG on validation;
- initialized and disk-reloaded LCG equals Base within tolerance;
- expected LCG parameters receive finite nonzero gradients;
- frozen SmolVLA parameters receive no gradients;
- LCG differs from Base and the no-language-contrast ablation after a small
  development fit;
- edits are bounded by translation, rotation, and gripper caps;
- clean-retention rows preserve Base behavior;
- gate activation is concentrated in language-sensitive states rather than
  everywhere;
- action postprocessing remains valid.

Do not proceed to bounded validation if:

- Base/null contrast is collapsed;
- residual labels are collapsed;
- no residual headroom remains after the CAG proxy;
- the no-language-contrast ablation explains the effect;
- standard LoRA explains the effect;
- the module globally changes all actions;
- clean retention fails;
- any privileged inference input or confirmatory-test identity is used.

## Expected Evidence If It Works

LCG should show:

- noncollapsed Base/null action contrast on development data;
- a validation improvement over Base and the CAG proxy on the language-contrast
  claim axis;
- an improvement over the no-language-contrast ablation;
- bounded action deltas with preserved action validity;
- clean retention where language contrast is absent;
- mechanism evidence tying gate activation to language-sensitive states;
- no confirmatory-test tuning.

## Current Status

No LCG implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this proposal.

Immediate next stage: Reviewer B attack on novelty, CAG prior boundary,
language-contrast data viability, objective scale, identity preservation, and
decisive experiment feasibility.
