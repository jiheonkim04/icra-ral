# PESA-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Method: `PESA-VLA`, Prior-Expert Spectral Adaptation for frozen SmolVLA 7D policies.

Proposal hash: `B05B1ACF7CD3514365B418E25C7E995604FCA8C117CDC0F3384F1046BAF26B63`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Rebuttal decision: `PESA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Audit decision: `PESA_MATHEMATICAL_AUDIT_PREREGISTERED`

## Variables And Shapes

- `B`: batch size.
- `D = 7`: normalized LIBERO action dimension.
- `x_t`: deployment observation processed by the official SmolVLA input path. Shape is processor-defined.
- `s_t in R^8`: robot proprioceptive state when available from local prediction artifacts.
- `u_t`: language instruction or transparent task-language proxy in development diagnostics.
- `rho_t in R^1`: normalized phase or chunk-position feature when available.
- `f_t in R^F`: deployment-observable feature vector extracted from frozen Base features, robot state, language proxy, phase, and `a_base_t`; `F` is fixed by implementation before Stage 0.
- `a_base_t in R^D`: frozen SmolVLA prior action in normalized 7D units.
- `a_exp_t in R^D`: expert demonstration action in the same units.
- `A_psi(f_t) in R^D`: adaptation expert action proposal.
- `a_adapt_t in R^D`: shorthand for `A_psi(f_t)`.
- `q_phi(f_t, a_base_t, a_adapt_t) in [0,1]`: prior-query gate.
- `alpha in R_+`: validation-selected action-delta cap.
- `d_t in R^D`: bounded adaptation delta.
- `a_pesa_t in R^D`: emitted PESA action.

For each spectral adapter layer `l`:

- `W_l in R^{d_out_l x d_in_l}`: frozen or adapter target weight shape for the selected local adapter layer.
- `r_l <= min(d_out_l, d_in_l)`: maximum spectral rank.
- `U_l in R^{d_out_l x r_l}`: learned or initialized left basis.
- `V_l in R^{r_l x d_in_l}`: learned or initialized right basis.
- `z_l(f_t) in R^{r_l}`: raw spectral logits.
- `s_l(f_t) in R_+^{r_l}`: nonnegative singular-like scores.
- `p_l(f_t) in [0,1]^{r_l}`: normalized score energy.
- `E_l(k; f_t) in [0,1]`: cumulative energy through rank `k`.
- `k_l(f_t) in {1,...,r_l}`: smallest active rank whose cumulative energy reaches threshold `eta`.
- `m_l(f_t) in {0,1}^{r_l}`: hard active-rank mask.
- `Delta W_l(f_t) in R^{d_out_l x d_in_l}`: input-conditioned spectral update.

Forbidden inference inputs:

- terminal success;
- reward;
- reset identity;
- future action;
- future state;
- object pose unless it is part of Base's deployment observation;
- confirmatory-test labels, task/reset identities, or rollout outcomes.

## Spectral Energy Construction

Raw scores:

`s_l(f_t) = softplus(z_l(f_t) / tau_s) + eps`

where:

- `tau_s > 0` is fixed before Stage 0 or selected only in bounded validation search;
- `eps = 1e-8` for numerical stability.

Energy distribution:

`p_l,i(f_t) = s_l,i(f_t)^2 / (sum_j s_l,j(f_t)^2 + eps)`

Cumulative energy:

`E_l(k; f_t) = sum_{i=1}^k p_l,i(f_t)`

Active rank:

`k_l(f_t) = min { k : E_l(k; f_t) >= eta }`

Hard mask:

`m_l,i(f_t) = 1[i <= k_l(f_t)]`

Input-conditioned spectral update:

`Delta W_l(f_t) = U_l diag(stopgrad(m_l(f_t)) * s_l(f_t)) V_l`

The hard rank boundary is not differentiated through. Gradients flow through active scores `s_l`, basis parameters `U_l`, `V_l`, feature projection parameters, adaptation expert parameters, and prior-query parameters. The hard mask is treated as a development-selected capacity selector, not a differentiable relaxation.

If hard masking causes nonacting or unstable gradients in Stage 0, the result is `IMPLEMENTATION_FAILURE` or `DESIGN_FAILURE` unless a single preregistered validation-only soft-mask fallback was explicitly included before training. No confirmatory outcome may select the mask rule.

## Action Formula

Action proposal:

`a_adapt_t = A_psi(f_t)`

Raw adaptation displacement:

`r_adapt_t = a_adapt_t - stopgrad(a_base_t)`

L2-clipped displacement:

`d_t = r_adapt_t * min(1, alpha / (||r_adapt_t||_2 + eps))`

Emission:

`a_pesa_t = clip_action(a_base_t + q_phi(f_t, a_base_t, a_adapt_t) * d_t)`

Initial condition:

- the final delta projection is zero-initialized or the prior-query bias is initialized closed;
- therefore `a_pesa_t = a_base_t` up to numerical tolerance before training;
- Stage 0 must report initial action delta mean, p95, max, and per-group deltas.

Per-group action deltas:

- translation: `||a_pesa_t[0:3] - a_base_t[0:3]||_2`
- rotation: `||a_pesa_t[3:6] - a_base_t[3:6]||_2`
- gripper: `|a_pesa_t[6] - a_base_t[6]|`

## Query-Label Construction

The prior-query label is optional and must be rejected if unhealthy.

Development-only residual magnitudes:

- Base error: `e_base_t = ||a_exp_t - a_base_t||_1`
- simple-adapter error: `e_simple_t = ||a_exp_t - a_simple_t||_1`

where `a_simple_t` is produced by a development-only standard LoRA or adapter trained only on train identities.

Material improvement margin:

`mu_q = quantile_train(e_base_t - e_simple_t, 0.60)`

Label:

`y_q,t = 1[(e_base_t - e_simple_t) > max(mu_q, delta_min)]`

Default `delta_min = 0.01` in normalized action L1 units, subject only to bounded validation search before confirmatory testing.

Stage 0 must reject query BCE if:

- `y_q` is all-zero or all-one;
- validation majority baseline cannot be beaten by a query probe;
- positives are concentrated in one task family or one phase shortcut;
- labels use confirmatory closed-loop outcomes.

If query BCE is rejected but the rest of PESA remains viable, the gate may be trained only through emitted-action and retention losses; that fallback must be declared before validation search and compared against the key ablation.

## Objective

Action scale:

`scale_j = median_train(|a_exp_t[j] - median_train(a_exp[:, j])|) + eps`

Adaptation imitation:

`L_adapt = mean_t Huber_delta((a_adapt_t - a_exp_t) / scale)`

Emitted-action imitation:

`L_emit = mean_t Huber_delta((a_pesa_t - a_exp_t) / scale)`

Clean retention:

Let `c_t in {0,1}` be a development-only clean-retention indicator. By default:

`c_t = 1[y_q,t = 0]`

when healthy query labels exist; otherwise `c_t` is the validation-defined low-headroom indicator based only on train/validation Base-vs-simple-adapter residuals.

`L_ret = mean_t c_t * Huber_delta((a_pesa_t - stopgrad(a_base_t)) / scale)`

Delta regularization:

`L_delta = mean_t ||q_phi_t * d_t||_2^2`

Spectral concentration:

`L_spec = mean_{t,l} H(p_l(f_t)) / log(r_l)`

where:

`H(p_l) = -sum_i p_l,i log(p_l,i + eps)`

This term encourages concentrated spectral energy. It is not a closed-loop performance metric.

Optional prior-query loss:

`L_query = mean_t BCEWithLogits(logit_q_t, y_q,t)`

Full objective:

`L = L_adapt + lambda_emit L_emit + lambda_ret L_ret + lambda_delta L_delta + lambda_spec L_spec + lambda_query L_query`

Default development coefficients before validation:

- `lambda_emit = 1.0`
- `lambda_ret = 0.50`
- `lambda_delta = 0.10`
- `lambda_spec = 0.05`
- `lambda_query = 1.0` when healthy query labels exist, otherwise `0.0`

Any coefficient change must occur only inside the bounded validation search and must be frozen before confirmatory testing.

## Gradient Path

Gradients flow into:

- adaptation expert parameters `psi`;
- prior-query gate parameters `phi`;
- spectral score parameters;
- active spectral basis parameters `U_l` and `V_l`;
- optional small feature projection parameters.

No gradients flow into:

- frozen SmolVLA Base;
- persisted Base actions;
- query-label construction;
- clean-retention label construction;
- train/validation/test split assignment;
- confirmatory-test identities or outcomes.

The stop-gradient on `a_base_t` is mandatory. PESA may use Base action as an inference feature and prior action source, but it may not update Base or treat Base action labels as trainable targets for Base itself.

## Small-Batch Magnitude And Gradient Audit

Before expensive training, report on a development-only batch:

- `L_adapt`;
- `L_emit`;
- `L_ret`;
- `L_delta`;
- `L_spec`;
- `L_query` when active;
- adaptation parameter gradient norm;
- query parameter gradient norm;
- spectral score gradient norm;
- `U_l` and `V_l` gradient norms for adapted layers;
- ratio of largest to smallest finite nonzero gradient norm;
- active rank distribution per layer;
- spectral entropy mean and p95;
- query positive and negative counts;
- emitted delta mean, p95, max;
- translation, rotation, and gripper delta summaries;
- action validity rate;
- initial Base-equality delta.

Hard stop when expected parameters receive zero, nonfinite, or catastrophically imbalanced gradients and the issue cannot be attributed to a narrow implementation defect.

## Simpler Alternatives

Closest-prior proxy:

- `priorvla_style_proxy`: frozen Base prior action plus standard adaptation expert and prior-query or retention gate, without spectral capacity allocation. It is a faithful transparent local proxy, not an official PriorVLA reproduction.

Key ablation:

- `pesa_no_spectral_no_prior_query_ablation`: same available development data and comparable adapter budget where feasible, but no spectral energy selection and no prior-query gate.

Simple killer:

- `standard_lora_or_clean_retention_baseline`: one strongest standard fixed-rank 7D LoRA/adapter or clean-retention LoRA mixture selected on validation before confirmatory testing.

PESA is killed if any of these explains the result under the frozen five-policy comparison.

## Stage 0 Development Audit Requirements

Stage 0 must report:

- train/validation/reserved split sizes;
- duplicate sample and frame keys;
- overlap across train, validation, reserved, task, episode, and reset identities;
- fixed Base checkpoint and preprocessing path;
- fixed 7D action-label availability;
- Base, simple adapter, and PriorVLA-style proxy development headroom;
- query-label balance and validation predictability when query BCE is active;
- spectral activation fractions and rank distributions;
- full-versus-proxy action distinction;
- full-versus-ablation action distinction;
- full-versus-simple-killer action distinction;
- clean-retention deltas;
- action validity;
- checkpoint save/reload identity.

Do not proceed to rollout when:

- labels are collapsed;
- no headroom exists;
- spectral/query modules are nonacting;
- PESA globally perturbs clean Base actions;
- intended mechanism cannot be inferred from deployment inputs;
- the PriorVLA-style proxy or simple baseline cannot be fairly constructed.

Classify such failures as `DATA_OR_SUPERVISION_FAILURE`, `NO_HEADROOM`, `IMPLEMENTATION_FAILURE`, or `DESIGN_FAILURE`, not as closed-loop scientific kills.

## Why Not KL

PESA does not compute KL divergence between deterministic 7D actions. `a_base_t`, `a_exp_t`, `a_adapt_t`, `d_t`, and `a_pesa_t` are deterministic vectors in normalized action units, not probability distributions.

The spectral energy vector `p_l(f_t)` is a valid distribution over adapter rank directions, but the audit does not require a KL term there. Entropy over `p_l` is sufficient for the intended concentration pressure and is easier to interpret. If a future variant proposes KL over spectral energies, it must name `p` and `q`, support, direction, estimator, gradient flow, and the reason KL is preferred over entropy, JS, MMD, Wasserstein, Mahalanobis, Huber/L2, or trajectory discrepancy. That would be a new method variant unless preregistered before confirmatory testing.

## Audit Verdict

PESA-VLA may proceed to preregistration and Stage 0 planning only under this audited objective:

- no deterministic-action KL;
- exact Base-passthrough initialization;
- frozen Base with mandatory stop-gradient;
- development-only query labels and validation-only coefficient/search selection;
- explicit closest-prior proxy and single simple killer;
- mandatory action-distinction, headroom, label-health, gradient, clean-retention, and action-validity checks before rollout.

Decision: `PESA_MATHEMATICAL_AUDIT_PREREGISTERED`.
