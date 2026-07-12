# Epoch 2 Candidate Generation

Date: 2026-07-12 KST

Governance: `reports/current_research_governance.md`

Epoch 2 must change at least two core dimensions relative to DICD, FEDO, and GCAP. The candidate set below avoids post-hoc delay adapters, low-level feedback residual correction, image hold-last or edge repair, selector/ranker/verifier routes, barrier/filter/damping, generic progress/value/confidence heads, generic DPO, and simple action reweighting.

## Candidate 1: PTC-VLA

Name: `PTC-VLA`, Posterior-Transition Conservative VLA

Hidden assumption: short-horizon proprioceptive transition latents expose execution intent that a lightweight policy-generation head can use without relying on privileged simulator state or post-hoc residual correction.

Precise novelty: train a small stochastic policy head to generate robot actions jointly with a transition-latent context derived from recent observed state changes and training-time paired state transitions. Unlike latent-action pretraining papers, this is a local, no-video-pretraining, conservative transition-head prototype on top of an official frozen VLA rollout substrate.

Equations:

- state transition: `dz_t = s_{t+1} - s_t`
- feature: `x_t = [s_t, dz_context_t, phase(t), task_code]`
- stochastic head: `mu_t, log_sigma_t = f_theta(x_t)`
- training loss: `L = ||mu_t - a_t||_2^2 + beta * mean(log_sigma_t^2) + lambda * ||mu_t - a_mean(phase, task)||_2^2`
- inference: `a_t = clip(mu_t, -1, 1)`

Representation: policy-input proprioceptive state, recent transition delta, phase code, and task code.

Objective: action negative log-likelihood/MSE with conservative regularization against phase/task mean actions.

Supervision: frozen official SmolVLA-LIBERO rollouts on training identities, using observed policy-input state transitions and generated action chunks.

Inference: generate actions directly from the learned transition-conditioned head; no action residual, no candidate ranking, no image repair.

Required data: official SmolVLA rollout traces on training identities, already locally executable.

Closest five papers:

| Paper | Link | Overlap | Difference |
| --- | --- | --- | --- |
| ALAM | https://arxiv.org/html/2605.10819v1 | latent transitions for VLA policy learning | no action-free video pretraining or joint flow matching; local transition head only |
| Conservative Offline Robot Policy Learning via Posterior-Transition Reweighting | https://arxiv.org/html/2603.16542v1 | posterior transition signal for conservative offline policy learning | no sample reweighting of an offline policy; trains a direct small head |
| LaWAM | https://arxiv.org/html/2606.15768 | latent world-action model conditions action generation | no learned world model or predicted visual subgoal |
| From Pixels to Tokens | https://arxiv.org/html/2605.04678v1 | latent action supervision taxonomy | local proprioceptive transition supervision rather than image/action latent benchmark |
| RotVLA | https://arxiv.org/html/2605.13403v1 | continuous latent action/flow head | no large VLM flow expert or SO(n) latent action pretraining |

Exact overlap matrix:

| Axis | PTC-VLA | Closest overlap? |
| --- | --- | --- |
| problem | local rollout instability from under-modeled short transitions | partial with latent-action works |
| representation | policy-input state transition latent | distinct from visual latent action/video latents |
| supervision | paired state/action transitions from frozen-policy traces | distinct from action-free videos and posterior reweighting |
| objective | conservative stochastic action head | partial with policy-distribution learning |
| policy component | small direct action generator | distinct from residual/wrapper/ranker |
| inference | transition-conditioned direct action | distinct from candidate selection |
| data | local official SmolVLA-LIBERO traces | distinct from large-scale video pretraining |
| claim | prototype headroom for transition-latent generation | narrower than latent-action SOTA claims |

Direct baseline: state-only/no-transition PTC ablation.

Simple killer baseline: phase/task mean-action policy.

Ablation: zero transition latent, same architecture.

Implementation plan: add `tca_map/smolvla/ptc_vla.py`, `scripts/run_ptc_vla_prototype.py`, focused tests, synthetic mechanism smoke, real trace training, and Stage A.

Prototype tasks: `libero_spatial/task_4` and `libero_10/task_4`, paired identities already used in Epoch 1 for comparability.

Second-backbone path: if Stage B/GO exists, test as a state/action head attached to Quantized OpenVLA-OFT INT4 action traces.

Second-condition path: apply to a language-retargeted or aliasing condition rather than visual occlusion or action fault.

Compute estimate: synthetic smoke under 1 minute; real trace training under 10 minutes; Stage A roughly 40-60 episodes, expected under 1 hour.

Failure risk: high; a state-only or mean-action baseline may explain all performance.

## Candidate 2: CAST-Lite VLA

Name: `CAST-Lite VLA`, Counterfactual Action-Semantics Transition Lite

Hidden assumption: instruction counterfactuals can be converted into local action-direction constraints without requiring new human correction chunks.

Precise novelty: create paired language counterfactual labels from local task instructions and train an action generator to preserve action components invariant to the counterfactual while changing controllable target-conditioned components.

Equations:

- counterfactual pair: `(o_t, l, a_t)` and `(o_t, l_cf, a_cf_proxy)`
- invariant mask: `m_inv = 1 - m_target(l, l_cf)`
- loss: `L = ||a_l - a_t||^2 + alpha ||m_inv * (a_l - a_cf)||^2 - gamma ||(1-m_inv)*(a_l-a_cf)||^2`

Representation: instruction pair embedding, policy-input state, and action component masks.

Objective: counterfactual language/action separation without generic DPO.

Supervision: synthetic counterfactual instruction labels plus frozen action traces.

Inference: direct language-conditioned action head.

Required data: local task instructions, frozen actions, and generated counterfactual labels.

Closest five papers:

| Paper | Link | Overlap | Difference |
| --- | --- | --- | --- |
| CAST | https://arxiv.org/html/2508.13446v2 | counterfactual labels improve instruction following | this is local action-component supervision, not full dataset augmentation |
| When Vision Overrides Language / CAG | https://arxiv.org/abs/2602.17659 | counterfactual VLA failures and language conditioning | not dual-branch VA/VLA guidance |
| FineVLA | https://arxiv.org/html/2605.27284v1 | fine-grained instruction alignment | no large fine-grained instruction dataset |
| ReSteer | https://arxiv.org/html/2603.17300v1 | trajectory steerability | no trajectory distribution quantification framework |
| Diagnosing Semantic Grounding | https://arxiv.org/html/2606.02277v1 | action-prediction semantic grounding gaps | aims at a direct local intervention |

Direct baseline: CAG-style language-unconditioned comparison proxy.

Simple killer baseline: instruction canonicalization only.

Ablation: no counterfactual separation loss.

Implementation plan: build local counterfactual labels and train a small action head.

Prototype tasks: LIBERO counterfactual/aliasing-style task pairs if locally available.

Second-backbone path: run on Quantized OpenVLA-OFT INT4 traces if a positive Stage B exists.

Second-condition path: LIBERO-CF-style instruction swaps.

Compute estimate: low for offline; closed-loop counterfactual condition may require more setup.

Failure risk: high overlap risk with CAST/CAG and possible weak local counterfactual labels.

## Candidate 3: CBF-VLA

Name: `CBF-VLA`, Controllability-Basis Flow VLA

Hidden assumption: internal controllable feature directions can be learned from small paired perturbations and used as a compact policy-generation basis rather than a generic confidence/progress head.

Precise novelty: learn a low-rank controllability basis that maps changes in policy-input state to action-generation basis coefficients, then train a flow-like head over coefficients instead of raw actions.

Equations:

- basis: `B in R^{d_action x k}`
- coefficient head: `c_t = g_phi(s_t, l, phase_t)`
- action: `a_t = B c_t`
- transition consistency: `L = ||Delta s_t - H B c_t||^2 + ||a_t - B c_t||^2`

Representation: low-rank controllability basis over action/state transition pairs.

Objective: coefficient-space action generation with transition consistency.

Supervision: paired state/action transitions from frozen traces.

Inference: direct coefficient-conditioned action generation.

Required data: same as PTC-VLA plus enough transition diversity to identify a basis.

Closest five papers:

| Paper | Link | Overlap | Difference |
| --- | --- | --- | --- |
| Observing and Controlling Features in VLA Models | https://arxiv.org/html/2603.05487v1 | controllability and feature steering | no internal VLA-feature steering, only external low-rank basis |
| ALAM | https://arxiv.org/html/2605.10819v1 | structured transition representation | low-rank action basis rather than latent transition pretraining |
| LaST0 | https://arxiv.org/html/2601.05248v1 | latent spatiotemporal reasoning | no latent CoT architecture |
| LaWAM | https://arxiv.org/html/2606.15768 | dynamics-aware latent policy | no world model |
| IntentVLA | https://arxiv.org/abs/2605.14712 | short-horizon intent representation | coefficient basis rather than history-conditioned VLA |

Direct baseline: PCA action basis without transition consistency.

Simple killer baseline: ridge regression on current state.

Ablation: no transition consistency term.

Implementation plan: learn basis from local traces, train coefficient head, then closed-loop evaluate.

Prototype tasks: same paired tasks as PTC-VLA.

Second-backbone path: fit basis on Quantized OpenVLA-OFT INT4 traces.

Second-condition path: aliasing/short-horizon intent shifts.

Compute estimate: low for offline; closed-loop similar to PTC-VLA.

Failure risk: may collapse to ridge regression and fail novelty/utility.

## Selection

Selected method: `PTC-VLA`.

Reason: PTC-VLA best satisfies the Epoch 2 requirement to change core dimensions while remaining implementable immediately. It changes representation to transition latents and changes policy generation to a stochastic direct head, without using residual correction, image repair, rankers, barriers, or confidence/progress heads. Its direct and simple baselines are straightforward and likely to kill it if the mechanism is trivial.
