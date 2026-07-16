# MCI-VLA Mathematical Mechanism Audit

Date: 2026-07-16 KST

Decision: `MCI_MATHEMATICAL_AUDIT_PREREGISTERED`

Method: `MCI-VLA`, Multi-Consistency Invariance for Base-preserving SmolVLA.

Proposal: `reports/mci_vla/researcher_proposal.md`

Reviewer attack: `reports/mci_vla/reviewer_attack.md`

Researcher rebuttal: `reports/mci_vla/researcher_rebuttal.md`

Proposal SHA-256:
`88CB11CC6236D19BA05602217C65C1819A68BEA53B041E17BA12796403BA0B9A`

## Audit Scope

This audit freezes the mathematical object that may be implemented in Stage 0.
No implementation, validation search, rollout, simulator evaluation, or
confirmatory-test tuning has happened before this audit.

MCI proceeds only as Base-preserving multi-consistency invariance for frozen
SmolVLA action chunks. LoRA or another small adapter may implement the
trainable low-compute surface, but it is not the scientific mechanism.

## Fixed Variables And Shapes

For a batch of development rows:

- `N`: batch size.
- `H = 50`: SmolVLA action-chunk horizon.
- `D = 7`: official continuous LIBERO action dimension.
- `d_z in {16, 32}`: validation-search latent consistency-code dimension.
- `x = (o, p, l, B)`: legal deployment input plus frozen Base chunk.
- `o`: current RGB observation or cached legal visual feature from the current
  observation.
- `p in R^{N x 8}`: proprioception.
- `l`: task string or deterministic task/language feature.
- `B in R^{N x H x D}`: frozen SmolVLA Base action chunk.
- `Y in R^{N x H x D}`: demonstration action chunk, development training and
  audit only.
- `T_k`: task-preserving transformation from family
  `k in {instruction, observation_proprioception, action_evolution}`.
- `x_k = T_k(x)`: transformed legal pair.
- `m_k in {0,1}^N`: valid-pair mask for transformation family `k`.
- `z = z_phi(o, p, l, B) in R^{N x d_z}`: consistency code.
- `z_k = z_phi(x_k) in R^{N x d_z}`: transformed-pair consistency code.
- `r = r_theta(o, p, l, B, z) in R^{N x H x D}`: residual proposal.
- `u = u_eta(o, p, l, B, z) in R^{N x H x D}`: gate logits.
- `G = sigmoid(u) in [0,1]^{N x H x D}`: soft gate.
- `Delta in R^D_+`: frozen groupwise action-delta cap vector.
- `R = Delta * tanh(r) in R^{N x H x D}`: capped residual.
- `A_raw = B + G * R in R^{N x H x D}`: raw emitted action chunk.
- `A = postprocess(A_raw) in R^{N x H x D}`: official action after local
  postprocessing.

At inference, MCI may use only current observations, proprioception,
task/language input, the frozen Base chunk, learned adapter parameters, and
validation-frozen constants. It may not use object pose, simulator state,
reward, success, done, timeout, reset identity, future observation, future
expert action, confirmatory identities, or confirmatory outcomes.

## Transformation Families

All transformation generators must be frozen in preregistration before any
confirmatory access.

Instruction transformations are deterministic task-preserving paraphrase
templates or official equivalent strings. No unlogged LLM prompt iteration is
allowed after validation.

Observation/proprioception transformations are bounded RGB brightness,
contrast, crop/resize, low-amplitude image noise, and bounded proprioceptive
jitter inside deployment-valid ranges.

Action-evolution transformations perturb the current Base chunk or legal
action-generation feature path with small valid action or flow-step noise. They
may enforce intent consistency but may not create new action targets from
confirmatory outcomes.

If a transformation changes task semantics, action semantics, or legal
deployment observability, Stage 0 stops as `DATA_OR_SUPERVISION_FAILURE` or
`DESIGN_FAILURE`.

## Forward Formula And Identity

The emitted action is:

`A = postprocess(B + sigmoid(u_eta(o,p,l,B,z_phi(o,p,l,B))) * Delta * tanh(r_theta(o,p,l,B,z_phi(o,p,l,B))))`.

Identity initialization is mandatory:

- frozen SmolVLA weights receive no gradients;
- the residual head final weights and bias are initialized to zero;
- therefore `R = 0`, `A_raw = B`, and `A = postprocess(B)`;
- the gate is still logged, but it cannot change actions while `R = 0`;
- disk reload must reproduce the same Base passthrough.

If initialization or reload changes any action beyond numerical tolerance,
Stage 0 stops as `IMPLEMENTATION_FAILURE`.

## Objective Terms

For a valid transformed pair `x, x_k`, define normalized consistency codes:

`h = layernorm(z)` and `h_k = layernorm(z_k)`.

Consistency-code loss:

`L_code = mean_k mean_i m_{k,i} * 0.5 * (||h_i - stopgrad(h_{k,i})||_2^2 / d_z + ||h_{k,i} - stopgrad(h_i)||_2^2 / d_z)`.

Action-consistency loss:

`L_act = mean_k mean_i m_{k,i} * Huber(A_i - A_{k,i}; beta_act)`.

Demonstration fit loss:

`L_fit = mean_i Huber(A_i - Y_i; beta_fit)`.

Base-retention loss:

`L_keep = mean_i Huber(A_i - B_i; beta_keep)`.

Representation variance floor:

`L_var = mean_j relu(gamma_var - std_batch(h[:,j]))^2`.

Action-bound penalty before postprocessing:

`L_bound = mean relu(|A_raw| - 1)^2`.

Total objective:

`L = lambda_c * (L_code + L_act) + lambda_fit * L_fit + lambda_keep * L_keep + lambda_var * L_var + lambda_bound * L_bound`.

Initial small-batch audit coefficients:

- `lambda_c = 0.5`;
- `lambda_fit = 1.0`;
- `lambda_keep = 1.0`;
- `lambda_var = 1.0`;
- `lambda_bound = 1.0`;
- `beta_act = beta_fit = beta_keep = 0.05`;
- `gamma_var = 0.5`.

These initial coefficients are for small-batch term and gradient inspection.
The only validation-search coefficient is `lambda_c`.

## Units And Scale

`L_code` and `L_var` are unitless representation losses. `L_act`, `L_fit`, and
`L_keep` are normalized SmolVLA action units. `L_bound` is squared normalized
action-bound violation before postprocessing.

Before training, Stage 0 must report:

- each unweighted objective value;
- each weighted objective value;
- gradient norm by objective and parameter group;
- weighted gradient norm ratios;
- action deltas by translation, rotation, and gripper groups;
- gate activation fraction by transformation family and task;
- representation standard deviation and collapse indicators;
- postprocessed action validity.

No objective may dominate another by more than `100:1` in weighted gradient
norm without documented normalization and validation-only coefficient
selection. If the ratio remains above `100:1`, Stage 0 stops as
`IMPLEMENTATION_FAILURE` or objective-scale failure before rollout.

## Gradient Paths

- `L_code` updates `phi` through `z_phi` on the non-stopped side of each
  symmetric term.
- `L_act` updates `phi`, `theta`, and `eta` through `z`, `r`, `G`, `R`, and
  `A`; transformations are data-generation operations and receive no gradient.
- `L_fit` updates `phi`, `theta`, and `eta` through the emitted action.
- `L_keep` updates `phi`, `theta`, and `eta` to preserve Base behavior.
- `L_var` updates `phi` and prevents constant-code collapse.
- `L_bound` updates `phi`, `theta`, and `eta` through `A_raw`.
- Frozen SmolVLA Base parameters receive no gradients.

Expected nonzero gradients before training:

- consistency encoder parameters from `L_code` and `L_var`;
- residual and gate adapter parameters from `L_act`, `L_fit`, `L_keep`, and
  `L_bound` after the identity smoke intentionally enables a nonzero residual
  probe;
- zero gradients to frozen SmolVLA parameters.

NaN, infinite, or unexpected zero gradients stop as `IMPLEMENTATION_FAILURE`.

## Simpler Alternative And Required Ablation

The simpler alternative is ordinary adaptation to the same legal augmented
development data without a learned multi-consistency code. It is policy 5:
`augmentation_only_lora_killer`.

The required mechanism ablation is policy 4:
`mci_no_consistency_code_ablation`. It removes the learned consistency code and
consistency losses while preserving the same adapter surface, data, action
caps, clean-retention objective, and training budget.

If either policy explains the gain, MCI is not a paper candidate.

## Required Comparisons

The first serious comparison remains exactly:

1. `smolvla_base`
2. `rovla_multiconsistency_proxy`
3. `mci_full`
4. `mci_no_consistency_code_ablation`
5. `augmentation_only_lora_killer`

Policy 2 must first attempt official RoVLA compatibility. If exact local
execution is unavailable, it must be labeled a transparent local proxy and
must preserve instruction, observation/proprioception, and action-evolution
consistency. It may not be a generic augmentation baseline renamed as RoVLA.

## Bounded Validation Search

The maximum validation search is six configurations:

- `lambda_c in {0.25, 0.50, 1.00}`;
- `d_z in {16, 32}`.

No other architecture, task, reset identity, transformation family, objective,
threshold, policy order, baseline, or confirmatory metric may be searched in
this method cycle.

The validation score is:

`S_val = 0.30 * success_or_best_legal_proxy
       + 0.20 * clean_retention
       + 0.20 * consistency_activation
       + 0.15 * action_validity
       + 0.10 * prior_and_ablation_margin
       + 0.05 * compute_overhead`.

All terms are scaled to `[0,1]`. Ties break by clean retention, then lower
Base-relative action delta, then lower adapter parameter count.

## No KL Between Deterministic Actions

MCI does not compute KL divergence between deterministic `7D` actions, action
chunks, or SmolVLA flow vectors.

Any future KL proposal is invalid unless it defines valid probability
distributions, support, KL direction, estimator, gradient flow, and why KL is
preferred over Huber/L2, JS, Wasserstein, MMD, Mahalanobis distance,
vector-field consistency, or trajectory discrepancy. The frozen MCI protocol
uses Huber action losses and representation consistency, not
deterministic-action KL.

## Stage 0 Required Diagnostics

Stage 0 must prove:

- proposal hash verification;
- discovery/validation/test separation;
- legal transformation generators are frozen and logged;
- no privileged inference input;
- action shape `[50, 7]`;
- duplicate-key and split-overlap checks;
- noncollapsed transform pairs, labels, and consistency codes;
- consistency signal above trivial task, phase, action-magnitude, and
  augmentation-family baselines;
- Base and RoVLA proxy leave development headroom;
- exact Base passthrough at initialization and after disk reload;
- finite nonzero expected gradients;
- bounded translation, rotation, and gripper action deltas;
- normalized and postprocessed action validity;
- clean validation retention;
- `mci_full` differs from Base, RoVLA proxy, no-code ablation, and
  augmentation-only LoRA on mechanism diagnostics.

## Stage 0 Stop Classes

- `DATA_OR_SUPERVISION_FAILURE`: collapsed transformations, collapsed
  representation targets, duplicate or overlap failure, insufficient task/demo
  coverage, or invalid transformation semantics.
- `NO_HEADROOM`: Base and RoVLA proxy leave no meaningful development headroom
  on the claimed robustness axis.
- `IMPLEMENTATION_FAILURE`: checkpoint reload failure, nonfinite or missing
  gradients, Base identity failure, wrong action shape, invalid actions, or
  objective-scale ratio above `100:1`.
- `DESIGN_FAILURE`: consistency signal is not observable from legal inputs,
  the adapter acts everywhere destructively, the mechanism is exactly trivial,
  or the ablation/simple killer explains the method before rollout.

None of these pre-rollout stops is a closed-loop scientific kill.

## Current Status

This audit freezes the mathematical mechanism only. No MCI implementation,
training, validation search, rollout, simulator access, or confirmatory-test
access has happened.
