# AFID-VLA Researcher A Proposal

Date: 2026-07-16 KST

Decision: `AFID_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `AFID-VLA`, Action-Factor Instruction Densification for
Base-preserving SmolVLA.

Cycle: Epoch 4 Cycle 33

Previous method: `LCG-VLA`

Previous fixed result: `LCG_STAGE_0_DESIGN_FAILURE`

LCG is preserved unchanged. No LCG repair, rescue, threshold change, proxy
change, task change, or reinterpretation is allowed.

## Claim

SmolVLA may fail when a goal-level instruction under-specifies the local
motion factors that determine the useful action residual: approach direction,
dominant translation axis, gripper event timing, rotation sign, and terminal
motion class. A sparse action-factor predictor trained only from
discovery/validation demonstrations can expose those factors from legal
deployment inputs and use them to gate bounded residual edits, while preserving
the frozen Base action exactly whenever the factor signal is absent or
uncertain.

## Closest Prior

Closest prior: FineVLA.

Primary source: `https://arxiv.org/html/2605.27284v1`

Positive result: FineVLA reports that fine-grained instruction supervision
improves steerable VLA control without sacrificing goal-level task completion.
Its RoboTwin analysis reports fine-grained-only gains over raw-only of `+1.4`
to `+8.1` success-rate points, and best mixed fine-grained/raw ratios around
`1:2` to `1:1`.

FineVLA's relevant mechanism is process-level language supervision: it adds
fine-grained control descriptions to raw task instructions so policies learn
execution constraints that are not explicit in the goal sentence.

## What Is New

AFID is not ordinary LoRA, not a new VLA backbone, not instruction rewriting
at inference, and not direct access to demonstration actions at inference.

AFID adds one mechanism to frozen SmolVLA:

`legal current input + frozen Base chunk -> predicted sparse action factors ->
confidence-gated action-cell mask -> bounded residual edits -> exact Base
passthrough elsewhere`.

The mechanism is distinct from FineVLA because:

- FineVLA supervises policies with fine-grained natural-language descriptions.
- AFID derives compact action-factor labels from existing LIBERO action and
  proprioception traces on discovery/validation partitions.
- AFID predicts those factors at inference from deployment-observable
  RGB/proprio/language/Base chunks, not from future actions or privileged
  simulator state.
- AFID integrates the factor prediction as an identity-preserving residual
  gate around frozen SmolVLA Base, rather than replacing Base behavior.
- AFID edits only factor-conditioned, noncollapsed time/action cells under
  translation, rotation, and gripper caps.

LoRA or a lightweight adapter may parameterize the factor predictor, gate, or
residual head. The scientific method is sparse action-factor densification and
Base-preserving residual gating, not LoRA itself.

## Mechanism Sketch

Let `x_t = (o_t, q_t, l_t)` be the legal deployment input at replanning time
`t`, with RGB observation `o_t`, proprioception `q_t`, and task instruction
`l_t`.

The frozen SmolVLA Base produces:

`B_t = pi_base(o_t, q_t, l_t) in R^{50 x 7}`.

For a development demonstration, let:

- `E_t in R^{50 x 7}` be the aligned demonstration action chunk;
- `R_t = E_t - B_t` be the Base residual target.

From `E_t`, `R_t`, and allowed proprioception traces, AFID constructs bounded
development-only factor labels:

- `z_axis in {x, y, z, none}` for dominant approach axis;
- `z_dir in {-1, 0, +1}^3` for dominant translation signs;
- `z_grip in {open, close, hold, none}` for gripper event timing;
- `z_rot in {-1, 0, +1}^3` for rotation signs;
- `z_term in {approach, align, grasp, transport, release, settle}` for
  terminal motion class.

These labels are supervision only. At inference, AFID must predict them from
`x_t`, SmolVLA features, and `B_t`; it may not read `E_t`, `R_t`, future
observations, success flags, done flags, object poses, or simulator state.

Let `Z_t` denote the factor-label vector and `M_factor in {0, 1}^{50 x 7}`
the development-only factor-conditioned action-cell mask. The mask is derived
only from discovery/validation residual statistics and must be noncollapsed:
not all zero, not all one, and covered across tasks and phases.

AFID predicts:

- factor probabilities `P_theta(Z_t | x_t, B_t)`;
- confidence `c_theta in [0, 1]`;
- residual proposal `Delta_theta(x_t, B_t) in R^{50 x 7}`;
- gate `G_theta(x_t, B_t, P_theta, c_theta) in [0, 1]^{50 x 7}`.

The action chunk is:

`A_t = B_t + G_theta * clip_group(Delta_theta, rho_translate, rho_rotate,
rho_gripper)`.

Initialization and disk reload must satisfy:

`G_theta = 0` and `A_t = B_t`

within the frozen identity tolerance. If `c_theta < tau_conf`, if predicted
factor entropy is too high, or if the factor mask is absent for the current
state, AFID must return exact Base.

## Development Data

Allowed development inputs:

- existing LIBERO discovery/validation demonstrations;
- task instructions from discovery/validation partitions;
- legal RGB observations and proprioception;
- frozen SmolVLA Base chunks;
- observation-derived SmolVLA features;
- aligned demonstration action chunks for supervision;
- factor labels derived from demonstration actions and proprioception.

Forbidden inputs:

- rollout reward;
- success or done flags;
- object poses or simulator privileged state;
- future observations;
- confirmatory-test task or reset identities;
- confirmatory-test failures or partial outcomes;
- factor labels at inference.

## Training Objective

All objectives operate on development partitions only.

Let `P_theta` be the predicted factor distribution, `Z_t` the derived factor
label, `A_t` the AFID action chunk, `B_t` the frozen Base chunk, `E_t` the
demonstration chunk, and `M_factor` the noncollapsed factor-conditioned mask.

The default objective is:

`L = L_factor + lambda_res L_res + lambda_clean L_clean + lambda_gate L_gate +
lambda_valid L_valid`.

Where:

- `L_factor = CE(P_axis, z_axis) + BCE(P_dir, z_dir) + CE(P_grip, z_grip) +
  BCE(P_rot, z_rot) + CE(P_term, z_term)`;
- `L_res = mean(M_factor * Huber(A_t - E_t))`;
- `L_clean = mean((1 - M_factor) * Huber(A_t - B_t))`;
- `L_gate = mean(G_theta)`;
- `L_valid` penalizes action-bound violations after official SmolVLA
  postprocessing.

Expected gradient paths:

- `L_factor` updates the factor predictor only;
- `L_res` updates the residual head and gate through `A_t`;
- `L_clean` updates the gate and residual head toward Base passthrough;
- `L_gate` discourages global activation;
- `L_valid` updates only trainable AFID parameters;
- frozen SmolVLA Base receives no gradients.

All terms use cross-entropy, binary cross-entropy, Huber/L1, or bound penalties
on valid variables. There is no KL divergence between deterministic 7D action
vectors.

Before nontrivial training, AFID must estimate term magnitudes and gradient
norms on a small development batch. Coefficients may be selected only by the
bounded validation search.

## Evidence Partitions

`DISCOVERY`: derive candidate factor labels, inspect Base residuals, debug
factor/mask health, and build the transparent FineVLA-style proxy.

`VALIDATION`: select one bounded configuration from at most six total
configurations, including any factor confidence threshold, residual cap,
factor loss coefficient, clean-retention coefficient, and adapter size.

`CONFIRMATORY_TEST`: used once only after method, configuration, checkpoint,
tasks, reset identities, metrics, thresholds, policy list, and ablations are
frozen.

No confirmatory-test outcome may be used to retune AFID.

## First Serious Comparison

The first serious comparison must include exactly:

1. `smolvla_base`
2. `finevla_action_factor_proxy`
3. `afid_full`
4. `afid_no_factor_ablation`
5. `standard_lora`

Policy 2 is the closest-prior proxy. If official FineVLA code or compatible
checkpoints are not locally available, the proxy must be transparent and
matched: use the same frozen SmolVLA Base, the same development partitions,
and the same derived action-factor labels, but apply them only as
fine-grained instruction text or training metadata without AFID's residual
gate.

The key ablation removes predicted action factors from the gate while keeping
parameter count, labels, optimizer budget, and action caps matched.

Standard LoRA is required because AFID trains a small module on demonstrations.

## Bounded Validation Search

Maximum search budget: `6` total configurations.

Allowed factors:

- factor-confidence threshold `tau_conf`;
- residual cap scale;
- `lambda_res` or `lambda_clean`;
- factor latent dimension;
- lightweight adapter or LoRA rank.

Selection score must combine validation proxy improvement, clean retention,
factor predictability, mechanism activation locality, action validity, and
compute overhead. Do not select purely by offline action L2.

After selection, freeze the single configuration, checkpoint, policy list,
metrics, thresholds, and confirmatory manifest before any confirmatory use.

## Stage 0 Development Audit

Stage 0 is development-only. It is not a closed-loop scientific result.

Required checks:

- proposal hash and source artifacts;
- discovery/validation/test identity separation;
- no reward/success/done/object-pose/confirmatory record access;
- factor labels parse and are noncollapsed;
- positive/negative factor examples cover tasks and phases;
- `M_factor` is not all-zero or all-one;
- factor prediction beats trivial majority baselines on validation;
- Base residual headroom exists for factor-conditioned cells;
- FineVLA action-factor proxy leaves residual headroom for AFID;
- initialized and disk-reloaded AFID equals Base within tolerance;
- expected AFID parameters receive finite nonzero gradients;
- frozen SmolVLA parameters receive no gradients;
- AFID differs from Base, the FineVLA proxy, and the no-factor ablation after
  a small development fit;
- edits are bounded by translation, rotation, and gripper caps;
- clean-retention rows preserve Base behavior;
- gate activation is sparse and factor-relevant rather than everywhere;
- action postprocessing remains valid.

Do not proceed to bounded validation if:

- factor labels are collapsed;
- factor prediction cannot beat a trivial baseline;
- no factor-conditioned residual headroom exists;
- the FineVLA proxy dominates AFID;
- the no-factor ablation explains the effect;
- standard LoRA explains the effect;
- the module globally changes all actions;
- clean retention fails;
- any privileged inference input or confirmatory-test identity is used.

## Expected Evidence If It Works

AFID should show:

- noncollapsed action-factor labels on development data;
- validation factor prediction above trivial baselines;
- a validation improvement over Base and the FineVLA-style proxy on the
  action-factor claim axis;
- an improvement over the no-factor ablation;
- bounded action deltas with preserved action validity;
- clean retention where factor confidence is low;
- mechanism evidence tying gate activation to predicted factor states;
- no confirmatory-test tuning.

## Current Status

No AFID implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this proposal.

Immediate next stage: Reviewer B attack on novelty, FineVLA prior boundary,
factor-label viability, objective scale, identity preservation, and decisive
experiment feasibility.
