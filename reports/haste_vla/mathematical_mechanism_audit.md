# HASTE-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Proposal hash:
`5415BC1533A24EC55CC511DDEB014BB11D9C19F603C59D1F1D3E151E15B930A6`.

Decision: `HASTE_MATHEMATICAL_AUDIT_PREREGISTERED`

## Variables And Shapes

- batch size: `B`;
- event horizon: `H_e in {20,50}`;
- postprocessed actions: `A in R^(B,L,7)`;
- arm actions: `A_arm in R^(B,L,6)`;
- gripper commands: `G in R^(B,L)`;
- audited SmolVLA representation: `Z in R^(B,960)`;
- valid future interval count: `M in {1,...,H_e}^B`;
- event offset: `Tau in {1,...,M} union {null}`;
- event indicator per interval: `Y in {0,1}^(B,H_e)`;
- valid survival mask: `S in {0,1}^(B,H_e)`;
- normalized relative displacement: `D in R^(B,6)` for uncensored rows;
- hazard logits and probabilities: `L_h,H in R^(B,H_e)`;
- predicted displacement: `D_hat in R^(B,6)`.

## Label Construction

For row `b`, `S[b,k]=1` only when future interval `k` is observed within the
demonstration and `k <= H_e`. `Y[b,k]=1` only at the first observed gripper
command transition; later positions are masked. If no event is observed,
`Y=0` on valid intervals and the row is right-censored.

For an event at `tau`,

`D_raw[b] = sum_(j=0)^tau A_arm[b,t+j]`.

Discovery statistics `mu in R^6` and `sigma in R^6` use an elementwise floor
`sigma >= 1e-6`:

`D[b] = (D_raw[b]-mu)/sigma`.

Units remain normalized postprocessed action coordinates. No SE(3), metric
pose, contact, or probability-density claim is made.

## Hazard Loss

`H = sigmoid(L_h)`.

For each valid pre-event or censored interval, survival contributes
`-log(1-H[b,k])`. The first event interval contributes `-log(H[b,tau])`.
Positions after the first event and beyond the data boundary are masked.

Use numerically stable BCE-with-logits and divide each row by its valid
likelihood term count, then average rows:

`L_haz = mean_b(sum_k S_eff[b,k] BCELogit(L_h[b,k],Y[b,k]) / N_b)`.

Support is Bernoulli event occurrence conditional on survival at each observed
interval. Gradient flows through hazard head and shared LoRA representation.

## Displacement Loss

For uncensored set `U`:

`L_disp = mean_(b in U, j in 1..6) Huber_1(D_hat[b,j]-D[b,j])`.

Censored rows provide no displacement gradient. Gradient flows through the
displacement head and shared LoRA representation.

## Flow And Retention

`L_flow` is the repository's ordinary conditional flow-matching objective on
legal action examples.

For identical noisy action state `x_s`, flow time `s`, and policy inputs:

`L_ret = mean Huber_1(v_theta(x_s,s)-stopgrad(v_base(x_s,s)))`.

It has the same normalized action-flow units as the policy vector field.
Gradient flows only to LoRA policy parameters.

## Total Objective

`L_total = L_flow + lambda_h L_haz + L_disp + L_ret`,

where `lambda_h in {0.25,0.5,1.0}` and all other coefficients equal `1.0`.

No KL, JS, Wasserstein, MMD, or Mahalanobis term is used. Hazard BCE is valid
because it operates on Bernoulli conditional event probabilities. Actions and
flow vectors are not normalized distributions.

## Closest-Prior Proxy

The transparent StaKe proxy uses the same `Z`, LoRA, action rows, retention
rows, and optimizer. It replaces HASTE heads with:

- binary stage BCE;
- Huber regression to the normalized absolute action at the next command-event
  keyframe.

The proxy is faithful to the public method description but is not official
code.

## Required Ablations

- no-hazard: remove `L_haz`, retain relative `L_disp`;
- StaKe proxy: binary stage plus absolute keyframe target;
- standard LoRA: remove both auxiliary heads;
- Base: no adaptation.

The hazard contribution is supported only if HASTE beats no-hazard under
matched training. The representation contribution is supported only if HASTE
beats the StaKe proxy. Generic adaptation is excluded only if HASTE beats
standard LoRA.

## Pretraining Audit

On one fixed discovery microbatch, persist:

- unweighted means and standard deviations of all objective terms;
- LoRA gradient L2 norms per term;
- auxiliary-head gradient norms;
- pairwise LoRA-gradient cosine matrix;
- finite fractions;
- event/censor and task composition;
- initialized Base/HASTE flow and action maximum errors.

Do not proceed on nonfinite values, zero required gradients, more than `100x`
unexplained nonzero LoRA-gradient ratio, identity failure, or missing masks.
