# PESA-VLA Researcher A Proposal

Date: 2026-07-15 KST

Method: `PESA-VLA`, Prior-Expert Spectral Adaptation for frozen SmolVLA 7D policies.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: PriorVLA, https://arxiv.org/abs/2605.10925.

Secondary priors: LoRA-SP, https://arxiv.org/abs/2603.07404; VLA-GSE, https://arxiv.org/abs/2605.06175 and https://github.com/YuhuaJiang2002/VLA-GSE.

## Claim

PESA-VLA tests whether an identity-preserving, prior-expert spectral adapter can improve frozen SmolVLA 7D closed-loop manipulation success beyond Base, a PriorVLA-style proxy, a no-spectral/no-prior-query ablation, and one strongest simple fixed-rank LoRA or clean-retention adaptation baseline.

The method is not a rescue of MARC, DAGR, MTF, RAC, CAVM, PSE, or any earlier fixed-protocol result. It does not retune MARC checkpoints, change MARC thresholds, reuse MTF frame-selection or retention coefficients, revive DAGR route-gated residuals, or reinterpret any previous Stage A/B outcome. It starts a new method cycle with a different prior, representation, objective, and adaptation mechanism.

The narrow local claim is:

> Explicitly preserving frozen SmolVLA as a read-only prior expert, while learning a spectral-capacity adaptation expert with Base-passthrough initialization and clean-retention gates, can improve official closed-loop success when standard local LoRA has adaptation headroom but risks disrupting good Base behavior.

## Positive Prior Anchor

PriorVLA reports that preserving a pretrained prior as a read-only expert while training a downstream adaptation expert improves VLA adaptation, including strong LIBERO, RoboTwin, few-shot, OOD, and real-world results. The useful design principle is not "use an adapter"; it is "keep the pretrained policy as an explicit prior source instead of letting downstream training overwrite it."

LoRA-SP reports that VLA adaptation has task- and input-dependent rank needs, and that fixed low-rank LoRA can be a poor capacity match for robot transfer. It uses an SVD-style basis and an energy threshold to select the active adapter capacity.

VLA-GSE reports that spectral generalized/specialized experts can improve parameter-efficient VLA adaptation and provides official code. PESA does not claim generic expert routing as novelty; VLA-GSE makes that crowded. PESA uses the spectral-expert evidence as a capacity-allocation prior while making the frozen local action prior explicit and auditable.

## Falsifiable Mechanism

Problem condition:

- Frozen SmolVLA is often strong on official LIBERO paired resets but still leaves residual task failures.
- Repeated local cycles show that small learned action wrappers can become harmful when they globally perturb Base actions.
- Standard LoRA or adapter training can improve offline action fit while losing closed-loop retention if Base behavior is overwritten.

Intermediate failure mechanism:

- A fixed-rank or always-active adaptation module may allocate too little capacity to hard states and too much update pressure to states where Base is already good.
- The pretrained policy becomes only an initialization, so training can damage useful motor priors.

Policy behavior:

- Base should remain the default action source.
- Adaptation should be allowed only when the learned prior-query state indicates useful headroom and the spectral adapter's active directions remain bounded.

Closed-loop failure:

- Unnecessary action shifts in approach, grasp, transport, or release can turn otherwise successful Base trajectories into failures, as seen in prior local cycles.

Proposed method:

- Keep frozen SmolVLA as a read-only prior expert that emits `a_base`.
- Train a 7D adaptation expert with spectral low-rank basis directions and per-input/layer energy scores.
- Train or calibrate a prior-query gate from deployment-observable inputs and Base/adaptation diagnostics.
- Emit exact Base at initialization and in low-confidence/clean-retention states.

Intended internal change:

- Adapter capacity concentrates in the input-relevant spectral directions instead of using a single fixed rank everywhere.
- The frozen prior action remains available as an explicit action source during inference.

Expected action behavior:

- Base-like actions on clean or uncertain states.
- Bounded adaptation when validation data shows that the adaptation expert improves action quality without destroying clean retention.

Expected closed-loop improvement:

- Higher task-balanced success than Base and fixed-rank LoRA on states where adaptation helps, with less degradation than unconstrained adapters on states where Base is already good.

## Data And Supervision

Discovery and validation may use existing official SmolVLA stable prediction artifacts, local fixed 7D adapter data, and development split records. Confirmatory identities remain held out until Stage A/B manifests are frozen.

Required records:

- frozen Base 7D action chunk;
- expert 7D action chunk;
- observation/state features available to existing local runners;
- task key, sample key, frame key, and split identity for audit only;
- adapter loss and gradient diagnostics from train/validation only;
- no reward, success, reset identity, future observation, future action, object pose, or confirmatory-test identity at inference.

Training targets:

- adaptation target: expert 7D action chunk under fixed train-split normalization;
- prior retention target: frozen Base 7D action chunk on clean-retention records;
- spectral capacity target: concentrate active singular-like scores until cumulative energy `E(k) >= eta`;
- optional prior-query label: train-only indicator that standard adaptation improves over Base by a material action margin, used only if noncollapsed and validation-predictable.

Stage 0 must reject before expensive training or rollout if:

- fixed 7D action labels, Base actions, or adapter features are missing;
- train/validation/reserved split overlap is nonzero;
- standard LoRA or a PriorVLA-style proxy has no development headroom over Base and trivial baselines;
- the prior-query label, if used, collapses to all-zero/all-one or cannot beat majority validation;
- spectral energy scores are nonacting or active everywhere;
- initial emitted action is not equal to Base up to numerical tolerance;
- validation action deltas are globally destructive;
- action validity is below `1.0` on development validation.

## Method Sketch

Let:

- `x_t`: deployment-observable policy features for timestep/chunk `t`;
- `a_base_t in R^7`: frozen SmolVLA prior action;
- `a_exp_t in R^7`: expert action;
- `A_psi(x_t) in R^7`: adaptation expert action;
- `U_l, V_l`: spectral adapter basis for layer `l`;
- `s_l(x_t) >= 0`: input-conditioned singular-like scores;
- `k_l(x_t)`: smallest active index where cumulative energy `E_l(k) >= eta`;
- `q_phi(x_t, a_base_t, A_psi(x_t)) in [0,1]`: prior-query adaptation gate;
- `alpha`: validation-selected action delta cap.

Spectral adapter for layer `l`:

`Delta W_l(x_t) = U_l diag(mask_k(s_l(x_t)) * s_l(x_t)) V_l`

where `mask_k` keeps only the active directions up to `k_l(x_t)`.

Action proposal:

`a_adapt_t = A_psi(x_t)`

Bounded adaptation delta:

`d_t = clip_l2(a_adapt_t - stopgrad(a_base_t), alpha)`

Emission:

`a_pesa_t = clip_action(a_base_t + q_phi(x_t, a_base_t, a_adapt_t) * d_t)`

Initial condition:

- adaptation emission is initialized to exact Base passthrough by setting the query gate closed or the final delta projection to zero;
- initial emitted action must match Base within numerical tolerance before training.

Training objective:

- action imitation: Huber/L1 between `a_adapt_t` and `a_exp_t`;
- prior retention: Huber/L1 between emitted action and `a_base_t` on clean-retention records;
- spectral concentration: encourage energy concentration in the active spectral set without selecting capacity on confirmatory data;
- delta regularization: penalize large emitted `q_phi d_t`;
- optional query BCE only if train-only labels are noncollapsed and validation-predictable.

No KL is computed between deterministic 7D action vectors.

## First Serious Comparison

Exactly five policies:

1. `frozen_smolvla`
2. `priorvla_style_proxy`
3. `pesa_full`
4. `pesa_no_spectral_no_prior_query_ablation`
5. `standard_lora_or_clean_retention_baseline`

`priorvla_style_proxy` is a faithful transparent local proxy for the prior-expert/adaptation-expert claim axis, not an official PriorVLA reproduction.

`standard_lora_or_clean_retention_baseline` is the one strongest simple reviewer-killer baseline. It will be selected before confirmatory testing from development validation only, and will normally be either standard fixed-rank 7D LoRA or a simple validation-selected clean-retention LoRA mixture.

## Bounded Development Search

Default budget:

- at most `6` total configurations;
- at most `2` prior-query head shapes;
- at most `3` spectral energy thresholds or one critical coefficient;
- at most `2` lightweight seeds only if cheap enough;
- no combinatorial grid;
- no confirmatory-test identity use.

Validation score should combine:

- standard LoRA/prior-proxy development headroom;
- full-versus-proxy and full-versus-ablation action distinction;
- clean-retention action delta;
- action validity;
- spectral activation not collapsed and not everywhere;
- finite nonzero gradients;
- latency and VRAM overhead.

Do not select purely by offline action L2.

## Required Ablations And Baselines

Closest-prior proxy:

- `priorvla_style_proxy`: frozen Base prior expert plus standard adaptation expert and prior query, without spectral capacity allocation.

Key ablation:

- `pesa_no_spectral_no_prior_query_ablation`: same data and adapter budget where feasible, but no spectral energy selection and no prior-query gate.

Simple killer:

- `standard_lora_or_clean_retention_baseline`: best simple fixed-rank 7D LoRA or clean-retention adapter chosen on validation before confirmatory testing.

Additional development diagnostics:

- Base action;
- adaptation action;
- emitted action;
- action delta L2 and per-group translation/rotation/gripper deltas;
- active spectral rank and energy threshold;
- query gate value;
- clean-retention delta;
- action validity;
- latency and VRAM.

## Stop Rules

Classify failures before rollout as:

- `DATA_OR_SUPERVISION_FAILURE` if labels, Base actions, split integrity, or query labels fail;
- `NO_HEADROOM` if Base, standard LoRA, and PriorVLA-style proxy leave no plausible improvement target;
- `IMPLEMENTATION_FAILURE` if gradients, checkpoint reload, adapter wiring, or 7D action validity fail;
- `DESIGN_FAILURE` if full PESA is equivalent to the PriorVLA-style proxy, standard LoRA, or no-spectral ablation.

Only a valid closed-loop Stage A/B result with active mechanism, frozen five-policy manifest, and no confirmatory-test tuning can kill the scientific current formulation.
