# MHS-VLA Mathematical Mechanism Audit

Date: 2026-07-16 KST

Decision: `MHS_MATHEMATICAL_AUDIT_PREREGISTERED`

Method: `MHS-VLA`, Mamba History State for Base-preserving SmolVLA.

Proposal: `reports/mhs_vla/researcher_proposal.md`

Reviewer attack: `reports/mhs_vla/reviewer_attack.md`

Researcher rebuttal: `reports/mhs_vla/researcher_rebuttal.md`

Proposal hash:
`BBDF67AE3EC4BD9D025707A8BB3A5008BAB5EB5C691D02D44516157802A87BF3`

## Fixed Constants

- action horizon `K = 50`;
- action dimension `D = 7`;
- history length `L = 8`;
- history state dimension `d_h = 128`;
- residual hidden dimension `d_r = 128`;
- Huber delta `delta_h = 0.01`;
- history usefulness margin `tau_hist = 0.01`;
- Base residual activity threshold `tau_base = 0.02`;
- translation residual cap `rho_trans = 0.02`;
- rotation residual cap `rho_rot = 0.05`;
- gripper residual cap `rho_grip = 0.25`;
- identity tolerance `epsilon_id = 1e-7`;
- gradient norm ratio alert threshold `20.0`;
- bounded validation search maximum `6` configurations.

All constants are development constants. They may be changed only by a new
method cycle or by a preregistered validation-only search factor where allowed.

## Variables And Shapes

For batch size `N`:

- current RGB observation: `O_0 in R^{N x C x H_img x W_img}` when available;
- history RGB observations: `O_hist in R^{N x L x C x H_img x W_img}` when
  available;
- current proprioception: `Q_0 in R^{N x q}`;
- history proprioception: `Q_hist in R^{N x L x q}`;
- previous executed or demonstration actions for history: `U_hist in R^{N x L x D}`;
- instruction embedding or token summary: `T in R^{N x d_l}`;
- frozen SmolVLA Base chunk: `B in R^{N x K x D}`;
- aligned demonstration chunk: `E in R^{N x K x D}`;
- Base residual target: `R = E - B in R^{N x K x D}`;
- history feature sequence: `X_hist in R^{N x L x d_x}`;
- current feature: `X_0 in R^{N x d_x}`;
- recurrent history state: `h in R^{N x d_h}`;
- residual proposal before caps: `P in R^{N x K x D}`;
- capped residual proposal: `Delta in R^{N x K x D}`;
- scalar gate: `g in R^{N x 1 x 1}`;
- deployed chunk: `A in R^{N x K x D}`;
- ambiguity/usefulness label: `m in {0, 1}^{N}`;
- history auxiliary target: `z in R^{N x 4}`.

If RGB observations are unavailable in the local HDF5 demonstrations, Stage 0
must record the missing modality and use only legal proprio/action/instruction
history for the development audit. If the target is not predictable from those
legal modalities above baselines, the outcome is `DATA_OR_SUPERVISION_FAILURE`
or `NO_USABLE_HEADROOM`, not a closed-loop scientific result.

## Feature Construction

For each history step `l`, construct:

`x_l = concat(f_o(O_l), f_q(Q_l), f_a(U_l), f_l(T))`.

Where:

- `f_o` is frozen SmolVLA-compatible visual feature extraction when available;
- `f_q` is an affine projection of proprioception;
- `f_a` is an affine projection of previous executed/demonstration action;
- `f_l` is a frozen instruction embedding or deterministic task embedding.

At inference, `U_l` is the robot's own executed action. In development row
construction, `U_l` may come from the demonstration trajectory because it is the
training analogue of the executed history. Demonstration actions are forbidden
at inference.

## History State

MHS uses a recurrent state-space update:

`h_0 = 0`

`h_l = SSM_phi(h_{l-1}, x_l)` for `l = 1 ... L`

`h = h_L`.

The local implementation may use a Mamba-style selective state-space block or a
minimal transparent recurrent state-space proxy when official MTIL/Mamba
components are unavailable. If a proxy is used, it must be labeled as a proxy
and must preserve the essential claim: recurrent history state, not
current-frame-only residual regression.

## Frozen Label Construction

Labels are development-only and may use `B`, `E`, `R`, and legal history
features. They are not inference inputs.

Define a robust Huber chunk error:

`err(Y, Z) = mean_{k,d} Huber_delta_h(Y_{k,d} - Z_{k,d})`.

For row `i`:

`e_base(i) = err(B_i, E_i)`.

Build two leave-one-out development proxies within the same split and task:

- `j_cur(i)`: nearest neighbor to row `i` by current-frame signature
  `c_i = summary(X_0_i, B_i, T_i)`;
- `j_hist(i)`: nearest neighbor to row `i` by history signature
  `r_i = summary(X_hist_i, U_hist_i, T_i)`.

The summaries are frozen deterministic vectors consisting of means and standard
deviations of the available legal features plus first/last/mean Base action
chunk statistics. The exact serialized summary function must be saved before
Stage 0 training.

Define:

`e_cur(i) = err(E_{j_cur(i)}, E_i)`

`e_hist(i) = err(E_{j_hist(i)}, E_i)`

`benefit(i) = e_cur(i) - e_hist(i)`.

The ambiguity/usefulness label is:

`m_i = 1[e_base(i) >= tau_base and benefit(i) >= tau_hist]`.

Rows with no valid current/history neighbor are masked out of `L_gate` and
reported separately. If unmasked labels are all zero, all one, task-collapsed,
or predictable only from task id, Stage 0 stops as data/supervision failure.

The auxiliary history target is:

`z_i = [clip(e_base(i), 0, 1), clip(benefit(i), -1, 1),
        mean_abs(R_i[:, 0:3]), grip_switch(E_i)]`.

Each component is normalized by discovery-set median and interquartile range
before training. Validation uses discovery normalization only.

This construction deliberately combines Base residual activity with a
history-over-current-frame nearest-neighbor advantage. It is not allowed to
reduce `m_i` to action L2 alone.

## Residual And Gate

Let:

`y = concat(h, X_0, summary(B), T)`.

The residual proposal is:

`P = reshape(W_2 sigma(W_1 y + b_1) + b_2, K, D)`.

The last residual layer is initialized:

`W_2 = 0`, `b_2 = 0`,

so `P = 0` at initialization while gradients into `W_2` are nonzero under
`L_res`.

The capped residual is:

`Delta[:,:,0:3] = rho_trans * tanh(P[:,:,0:3])`

`Delta[:,:,3:6] = rho_rot * tanh(P[:,:,3:6])`

`Delta[:,:,6] = rho_grip * tanh(P[:,:,6])`.

The gate is:

`g = sigmoid(w_g^T y + b_g)`,

with `b_g = -8` at initialization. Base identity at initialization is exact
because `Delta = 0`. Gate values are still reported because broad high gate
activation after training is a design failure.

The deployed chunk is:

`A = B + g * Delta`.

## Objective

The development objective is:

`L = L_res + lambda_gate L_gate + lambda_hist L_hist + lambda_clean L_clean +
lambda_valid L_valid`.

Terms:

`L_res = mean_i m_i * err(A_i, E_i) / max(mean_i m_i, epsilon)`.

`L_gate = BCE(g_i, m_i)` over rows with valid `m_i`.

`L_hist = mean_i Huber(z_hat_i - z_i)`, where `z_hat = W_z h + b_z`.

`L_clean = mean_i (1 - m_i) * err(A_i, B_i) / max(mean_i (1 - m_i), epsilon)`.

`L_valid` is the mean postprocessing bound violation:

`L_valid = mean ReLU(abs(A_post) - action_bound)`.

Default coefficients before validation search:

- `lambda_gate = 0.25`;
- `lambda_hist = 0.10`;
- `lambda_clean = 1.0`;
- `lambda_valid = 10.0`.

The only coefficient family allowed for bounded validation search is
`lambda_clean in {0.5, 1.0, 2.0}` combined with at most two architecture choices
and at most six total configurations. If a later preregistration narrows the
search further, the narrower rule controls.

## Gradient Paths

- `L_res` updates the residual head, history state encoder, and gate through
  `A = B + g * Delta`; frozen Base receives no gradients.
- `L_gate` updates the gate and history representation.
- `L_hist` updates the history encoder and auxiliary head only.
- `L_clean` updates gate and residual head toward Base passthrough on clean
  rows.
- `L_valid` updates only trainable MHS parameters through `A_post`.

Before any nontrivial training, Stage 0 must report finite nonzero gradient
norms for the history encoder, residual head, and gate, and zero gradients for
frozen SmolVLA parameters.

## Required Magnitude Audit

On a small development batch, report:

- `L_res`, `L_gate`, `L_hist`, `L_clean`, `L_valid`;
- gradient norms for history encoder, residual head, gate, and auxiliary head;
- maximum gradient norm ratio;
- action delta p50/p95/max by translation, rotation, and gripper groups;
- gate p50/p95/max;
- clean row action delta p95;
- action-bound violation count.

If any single objective term or gradient path dominates beyond the frozen ratio
without justification, stop as `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` before
rollout.

## Closest Alternatives And Ablations

Closest mathematical alternatives:

- current-frame residual regression without history;
- frame-stacked transformer or RNN imitation;
- MTIL-style full policy replacement;
- nearest-neighbor history retrieval;
- standard LoRA adaptation.

Required ablation:

`mhs_no_history_state_ablation`.

It must keep residual/gate capacity, data, optimizer, steps, action caps, and
selection metric matched while replacing `h` with current-frame features only.

Required prior proxy:

`mtil_history_state_proxy`.

It must use recurrent history state and action prediction, but it must not use
MHS's Base-preserving residual/gate identity mechanism.

Required simple reviewer-killer:

`standard_lora`.

It tests whether ordinary adaptation with the same training rows and compute
explains MHS.

## Identity-Preserving Integration Audit

Before any rollout:

- initialized MHS must return `A = B` within `epsilon_id`;
- disk-reloaded MHS must return `A = B` within `epsilon_id`;
- Base action, MHS action, residual norm, gate value, changed dimensions, and
  activation context must be reported;
- action deltas must satisfy the group caps;
- clean validation behavior must be retained;
- intended mechanism activation must concentrate on rows with `m_i = 1`;
- no privileged inference input may appear in the exported policy interface.

## Stop Conditions

Stop before bounded validation when:

- labels are collapsed or task-only;
- current-frame baseline matches history predictability;
- no Base residual activity exists;
- no history-over-current-frame headroom exists;
- expected parameters receive zero or nonfinite gradients;
- MHS is identical to no-history ablation after a small fit;
- MHS globally changes actions;
- clean retention fails;
- action validity fails;
- privileged inference inputs or confirmatory identities are used.

Classify these as `DATA_OR_SUPERVISION_FAILURE`,
`NO_USABLE_HEADROOM`, `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, or
`DESIGN_FAILURE`, not as a closed-loop scientific result.

## Current Status

No MHS implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this audit.

Immediate next stage: preregistration.
