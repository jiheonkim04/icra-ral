# MARC-VLA Researcher A Proposal

Date: 2026-07-15 KST

Method: `MARC-VLA`, Median-Anchored Regression Correction for frozen SmolVLA flow actions.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: OpenVLA-OFT, https://arxiv.org/abs/2502.19645 and https://openvla-oft.github.io/.

Secondary priors: ReactVLA and SnapFlow.

## Claim

MARC-VLA tests whether a lightweight, identity-preserving median action anchor can improve frozen SmolVLA closed-loop manipulation success beyond Base, an OpenVLA-OFT-style L1 continuous-action proxy, a no-disagreement-gate ablation, and a static Base/L1 mixture simple baseline.

The method is not a rescue of DAGR, MTF, RAC, CAVM, PSE, or any earlier fixed-protocol result. It does not retune DAGR route thresholds, MTF retained-frame settings, RAC consequence calibration, or previous task/reset identities. It starts a new method cycle with a different prior, representation, supervision, objective, and action-generation mechanism.

## Positive Prior Anchor

OpenVLA-OFT reports that an optimized continuous-action fine-tuning recipe using parallel decoding, action chunking, and L1 regression substantially improves LIBERO success and speed over base OpenVLA. The paper and project discussion also motivate a robust regression interpretation: L1 action prediction can avoid reproducing noisy or suboptimal demonstration modes that expressive diffusion-style policies may imitate.

ReactVLA and SnapFlow provide secondary support that flow-action generation is not sacred: calibrating or simplifying the flow path can improve performance or preserve success while reducing latency. MARC does not reproduce these architectures. It uses them as design constraints: keep inference lightweight, avoid multiple expensive policy calls, and report latency.

## Falsifiable Mechanism

Problem condition:

- SmolVLA emits continuous 7D flow-action chunks.
- The flow action can be locally plausible but closed-loop brittle when demonstrations contain noisy or suboptimal modes or when iterative flow denoising lands off the robust action center.

Intermediate failure mechanism:

- Base action generation may preserve multimodal or noisy behavior where a median-like action would be safer.
- Fully replacing Base is risky because frozen SmolVLA is already strong on many states.

Policy behavior:

- Base should be preserved by default.
- Corrections should occur only when a development-learned disagreement signal predicts that a robust median anchor is useful.

Closed-loop failure:

- Small action errors in approach, insertion, grasp, or release compound into missed manipulation success.

Proposed method:

- Train a robust median anchor `m_t in R^7` using L1/Huber loss on expert 7D actions.
- Train a disagreement gate from train-only base/expert action disagreement labels.
- Emit a clipped, gate-scaled correction from `a_base_t` toward `m_t`.

Intended internal change:

- The anchor captures a robust central action tendency.
- The gate restricts intervention to states where Base and the robust anchor are meaningfully different and predictable from deployment inputs.

Expected action behavior:

- Exact Base action at initialization and in low-disagreement states.
- Bounded move toward the median anchor in high-disagreement states.

Expected closed-loop improvement:

- Better manipulation success where Base flow actions are off-mode, while clean behavior and action validity are retained.

## Data And Supervision

Discovery and validation data may use existing official SmolVLA stable prediction artifacts and development split records. Confirmatory identities remain held out until Stage A/B manifests are frozen.

Required records:

- frozen Base 7D action chunk;
- expert 7D action chunk;
- task key, sample key, frame key, split identity;
- deployment-observable state/features used by existing local runners;
- normalized phase or chunk position only when allowed by the stable artifact;
- no reward, success, reset identity, future observation, future action, or object pose at inference.

Training labels:

- median-anchor target: `a_exp_t`;
- disagreement label: `1[||a_exp_t - a_base_t||_2 > tau_disagree]`, where `tau_disagree` is computed from train-only disagreement magnitudes;
- optional per-dimension materiality labels for diagnostics only, not policy selection.

Stage 0 must reject before rollout if:

- disagreement labels collapse to all-zero or all-one;
- L1 proxy does not beat trivial mean-action or fails action validity;
- MARC full is indistinguishable from the L1 proxy, no-gate ablation, or static mixture on validation action behavior;
- train/validation/test identity overlap is nonzero;
- initial emitted action is not equal to Base up to numerical tolerance;
- validation action deltas are globally destructive;
- the gate cannot beat a trivial majority baseline on validation.

## Method Sketch

Let:

- `a_base_t in R^7`: frozen SmolVLA action;
- `a_exp_t in R^7`: expert action;
- `m_theta(x_t) in R^7`: robust median anchor;
- `g_phi(x_t) in [0,1]`: disagreement gate;
- `alpha`: correction cap selected only on validation.

Correction:

`c_t = clip_l2(m_theta(x_t) - stopgrad(a_base_t), alpha)`

Emission:

`a_marc_t = clip_action(a_base_t + g_phi(x_t) * c_t)`

Initial condition:

- correction projection is zero-initialized or gate is initialized closed;
- the initial emitted action equals Base up to numerical tolerance.

Training objective:

- anchor loss: per-dimension L1/Huber between `m_theta(x_t)` and `a_exp_t`;
- gate loss: BCE on train-only disagreement labels;
- delta regularizer: `||g_phi c_t||_2^2`;
- clean-retention loss on low-disagreement records;
- optional scale normalization fixed from train split.

No KL is computed between deterministic 7D action vectors.

## First Serious Comparison

Exactly five policies:

1. `frozen_smolvla`
2. `openvla_oft_l1_proxy`
3. `marc_full`
4. `marc_no_disagreement_gate_ablation`
5. `static_l1_mixture_baseline`

`openvla_oft_l1_proxy` is a faithful transparent local proxy for the continuous L1/action-chunking claim axis, not an official OpenVLA-OFT reproduction.

`static_l1_mixture_baseline` is the one strongest simple reviewer-killer baseline: a validation-selected static convex mixture of Base and the L1 proxy with no learned state-dependent gate.

## Bounded Development Search

Default budget:

- at most `6` total configurations;
- at most `2` gate architectures;
- at most `3` correction caps;
- at most `2` lightweight seeds only if training is cheap enough;
- no combinatorial grid;
- no confirmatory-test use.

Validation score should combine:

- L1 proxy strength and MARC full-versus-proxy distinction;
- gate predictability above majority baseline;
- full-versus-no-gate and full-versus-static-mixture separation;
- clean action retention;
- action validity;
- bounded intervention frequency and latency.

## Required Ablations And Baselines

Closest-prior proxy:

- `openvla_oft_l1_proxy`: continuous L1/Huber action adapter, same data and 7D action semantics, no state-dependent MARC gate.

Key ablation:

- `marc_no_disagreement_gate_ablation`: same anchor, correction cap, and training data, but no learned disagreement gate.

Simple killer:

- `static_l1_mixture_baseline`: static validation-selected mixture `a = (1 - beta) a_base + beta a_l1_proxy`.

## Stop Rules

Classify failures before rollout as:

- `DATA_OR_SUPERVISION_FAILURE` if labels collapse, split integrity fails, or the gate is unobservable;
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` if gradients, checkpoint reload, or target wiring fails;
- `DESIGN_FAILURE` if MARC full is equivalent to the L1 proxy, no-gate ablation, or static mixture;
- `CONDITION_TOO_SEVERE_OR_NO_HEADROOM` if Base, L1 proxy, static mixture, and diagnostics leave no plausible improvement target.

Only a valid closed-loop Stage A/B result with active mechanism and fair baselines can kill the scientific current formulation.
