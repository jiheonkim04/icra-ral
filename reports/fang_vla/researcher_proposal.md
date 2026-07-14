# Researcher A Proposal: FANG-VLA

Date: 2026-07-14 KST

Method name: `FANG-VLA`, Identity-Preserving Failure-Aware Negative Guidance for Frozen VLAs.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: AFIL, Failing Forward: Adaptive Failure-Informed Learning for Vision-Language-Action Models, https://arxiv.org/abs/2605.08434.

## Core Claim

Failure-aware negative guidance is a positive-prior mechanism for flow and diffusion VLA policies, but direct failure guidance can disrupt a strong pretrained policy when applied locally to a frozen backbone. FANG tests a constrained version:

learn success and failure action fields from non-confirmatory frozen SmolVLA traces, then convert those fields into a bounded residual at inference. The inference path is initialized and calibrated so the default behavior is exactly the base action unless validation evidence supports a bounded residual.

The claim is not "failure trajectories are useful" in general. AFIL already establishes that prior. The claim is that identity-preserving, validation-calibrated failure guidance can make a frozen VLA safer to adapt than an unconstrained AFIL-style local proxy and stronger than success-only or nearest-success memory baselines.

## Problem Evidence

CAVM-VLA produced a valid but non-GO result: full CAVM reached `24 / 58`, nearest-success replay reached `23 / 58`, and the paired confidence interval against the strongest baseline was unresolved. This suggests outcome contrast can be active, but non-parametric memory was too weak to support a paper-ready direction.

AFIL provides a stronger external prior: a learned success/failure generator can steer a flow or diffusion VLA away from failure-prone regions. FANG therefore changes the mechanism from local memory replay to parametric dual action-field guidance, while preserving base SmolVLA actions by construction.

## Evidence Partitions

`DISCOVERY`:

- Prior local reports through CAVM adjudication.
- Existing non-confirmatory CAVM acquisition/calibration traces from identities `20260901..20260916`.
- Literature mechanism map and candidate generation reports.

`VALIDATION`:

- Development split carved only from non-confirmatory identities.
- Bounded validation search over at most six configurations.
- Clean retention, action-delta, gradient, label-health, and mechanism-activation diagnostics.

`CONFIRMATORY_TEST`:

- New held-out identities not used by CAVM acquisition/calibration or FANG validation.
- Used only after method, configuration, baselines, ablation, tasks, reset identities, metrics, and thresholds are frozen.
- Confirmatory outcomes may not be used to retune FANG.

## Data And Supervision

Training examples use only deployment-available inputs plus terminal success labels after the episode ends:

- task key;
- official 8D proprioceptive state `q_t`;
- frozen 7D base action `a_t`;
- previous executed 7D action `a_{t-1}`;
- chunk-index fraction `rho_t`;
- terminal episode success label `y_episode`.

The terminal success label is privileged for training only. It is never available at inference.

Input feature:

`x_t = [q_t, a_t, a_{t-1}, rho_t, task_one_hot]`.

Feature dimension for the two-task prototype:

`8 + 7 + 7 + 1 + 2 = 25`.

## Mechanism

FANG trains a lightweight shared trunk with two action-field heads:

- `m_plus(x_t)` predicts a success-conditioned 7D action field;
- `m_minus(x_t)` predicts a failure-conditioned 7D action field;
- `g(x_t)` predicts a bounded reliability gate trained from a discovery-only success/failure field-separation target.

The frozen base action remains the anchor. The derived guidance direction is:

`u_t = (m_plus(x_t) - a_t) + beta * (m_plus(x_t) - m_minus(x_t))`.

The full action is:

`a'_t = clip_action(a_t + alpha * G_t * clip_delta(u_t))`.

`G_t` is a validation-calibrated gate. At initialization, `G_t = 0`, so `a'_t = a_t` regardless of the untrained action-field heads.

## Training Objective

Let `a_obs_t` be the logged executed frozen action from a trace row. FANG trains action-field heads, not oracle corrective residual targets. This avoids the collapse that would occur if every logged action were subtracted from itself.

Let `c_t in [0, 1]` be a reliability target computed on discovery data from same-task success/failure neighbor action-field separation. It is high only when both classes have nearby support and their action fields separate beyond the preregistered minimum. It is not available at inference; the gate head predicts it from deployment features.

For success-labeled rows:

`L_success = Huber(m_plus(x_t), stopgrad(a_obs_t))`.

For failure-labeled rows:

`L_failure = Huber(m_minus(x_t), stopgrad(a_obs_t))`.

Clean retention and identity preservation:

`L_delta = mean(||alpha * G_t * clip_delta((m_plus(x_t) - a_t) + beta * (m_plus(x_t) - m_minus(x_t)))||_2^2)`.

Gate sparsity:

`L_gate = mean(G_t)`.

Gate reliability:

`L_gate_fit = BCEWithLogits(s_gate(x_t), stopgrad(c_t))`.

Total objective:

`L = L_success + L_failure + lambda_delta L_delta + lambda_gate_fit L_gate_fit + lambda_gate_sparse L_gate`.

The mathematical mechanism audit must check term magnitudes and gradient norms before training beyond the smoke stage.

## Bounded Validation Search

Maximum six configurations:

1. `alpha=0.10`, `lambda_delta=0.10`
2. `alpha=0.20`, `lambda_delta=0.10`
3. `alpha=0.35`, `lambda_delta=0.10`
4. `alpha=0.10`, `lambda_delta=0.30`
5. `alpha=0.20`, `lambda_delta=0.30`
6. `alpha=0.35`, `lambda_delta=0.30`

Fixed choices:

- one architecture: MLP trunk width `64`, depth `2`;
- one random seed per configuration for the initial smoke;
- no combinatorial search beyond the six listed configurations;
- `beta = 0.50`;
- `lambda_gate_fit = 1.0`;
- `lambda_gate_sparse = 0.01`;
- action residual clip fixed before validation.

Validation score:

`S = mechanism_activation + clean_retention + action_validity - disruption_penalty + validation_proxy_improvement`.

The exact numeric validation score must be frozen in the preregistration before use. It may not use confirmatory identities or confirmatory outcomes.

## Required First Comparison

Exactly five policies for the first serious rollout:

1. `base_smolvla`: unmodified frozen SmolVLA.
2. `afil_local_proxy`: dual success/failure action-field guidance without the identity-preserving validation gate.
3. `fang_full`: full FANG.
4. `fang_no_failure_ablation`: success residual and gate only; no failure residual.
5. `nearest_success_replay`: strongest simple reviewer-killer baseline inherited from CAVM.

## Falsifiable Chain

Observed problem:

Frozen SmolVLA has mixed success/failure behavior on hard local LIBERO tasks, and non-parametric outcome contrast was active but too weak.

Intermediate failure mechanism:

Local successful and failed trajectories contain class-conditional action fields, but nearest-neighbor memory is sparse and coarse.

Policy representation/action behavior:

FANG learns smooth success/failure action fields and applies a small derived residual only when the validation-calibrated gate indicates reliable separation.

Closed-loop consequence:

Held-out task-balanced success improves over Base, AFIL local proxy, no-failure ablation, and nearest-success replay while retaining clean/base behavior.

## GO Criteria

Prototype GO requires:

- `fang_full` beats the strongest prototype baseline and `fang_no_failure_ablation`;
- full-minus-strongest baseline reaches the active governance useful-improvement rule for Stage B or its preregistered expansion;
- mechanism activates in relevant states and not everywhere;
- action validity is preserved;
- clean validation behavior is retained;
- inference uses no privileged success, simulator object pose, reward, BDDL predicate, or held-out identity membership.

## Kill Criteria

Kill the current formulation if:

- the pre-experiment audit finds collapsed labels, no headroom, or no predictable success/failure signal;
- the checkpoint does not persist/reload or expected parameters lack finite nonzero gradients;
- FANG globally changes most actions or violates action bounds;
- `afil_local_proxy`, `fang_no_failure_ablation`, or `nearest_success_replay` matches or beats `fang_full` in Stage B;
- full is clearly worse than Base;
- clean retention fails;
- confirmatory outcomes would be needed to retune the method.

## Non-Claims

FANG is not an official AFIL reproduction.

FANG is not a CAVM rescue, CAVM retune, or memory reconstruction change.

FANG is not a generic residual action head. Its paper claim is only valid if the failure-aware residual term and identity-preserving gate beat the AFIL proxy, no-failure ablation, nearest-success baseline, and Base under a frozen protocol.
