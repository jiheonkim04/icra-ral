# Epoch 4 Cycle 17 Prior Mechanism Map

Date: 2026-07-15 KST

Purpose: select the first method after IARC Stage 0A closed as
`IARC_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`. IARC remains an honest fixed
result. This cycle does not clip or reinterpret IARC actions, alter its frozen
range gate, run its one-check, or reuse its confirmatory identities.

## Local Evidence Boundary

The next method must not cosmetically re-enter a closed local axis:

- output-action correction, residual repair, EMA, chunk scheduling, queue
  control, or action-history heuristics;
- progress, future-state, waypoint, action-latent, value, confidence, memory,
  selector, or ranker methods;
- spatial or 3D labels without a verified noncollapsed supervision source;
- occlusion completion or complementary-view prediction;
- inference-time flow guidance, counterfactual action mixing, or a LIFT/IARC
  action-bound rescue;
- LoRA or QLoRA presented as the scientific contribution.

The official local checkpoint is `lerobot/smolvla_libero`, trained on the
`lerobot/libero` dataset. The dataset card reports `40` tasks and `1,693`
episodes. The local raw official LIBERO download also contains `libero_90`,
which supplies held-out task demonstrations and simulator definitions not used
by that checkpoint. This creates a legitimate new-skill adaptation axis rather
than another intervention on the already tested 40-task action surface.

## Positive Prior 1: RETAIN

Full title: Robust Finetuning of Vision-Language-Action Robot Policies via
Parameter Merging.

Primary sources:

- paper: https://arxiv.org/abs/2512.08333
- official code: https://github.com/yajatyadav/RETAIN_code
- project: https://retain.yajatyadav.com/

Positive result:

- RETAIN linearly interpolates a pretrained generalist policy and a policy
  finetuned on limited demonstrations of a new task;
- it reports that the merged policy outperforms the pretrained and finetuned
  endpoints on OOD target-task variants while retaining generalist skills;
- the paper evaluates about `45` demonstrations per LIBERO target task and
  explicitly studies vision, language, and action parameter groups;
- official code exposes checkpoint interpolation and LIBERO OOD evaluation.

Limitation to extend:

- a global or manually selected modality coefficient treats parameter distance
  as a proxy for behavioral disruption;
- equal parameter movement can have very different action effects across
  modules and states;
- the merge coefficient therefore does not directly enforce a functional
  clean-retention budget.

Local faithful proxy:

- train one standard low-rank task-adaptation endpoint on held-out LIBERO-90
  demonstrations;
- keep the effective task vector and parameter scaffold identical across
  RETAIN, Ours, and ablation;
- implement RETAIN as transparent interpolation of the same effective task
  vector, never as an official reproduction of the openpi training stack.

## Mechanism Prior: Functional Model Merging

Full title: Merging Models with Fisher-Weighted Averaging.

Primary source:

- paper and official code link: https://arxiv.org/abs/2111.09832

Positive result:

- Fisher merging improves over simple parameter averaging in robust
  fine-tuning and model combination;
- it demonstrates that parameter importance can make model merging more
  faithful than isotropic averaging.

Difference from the selected opportunity:

- Fisher merging estimates likelihood-space parameter precision;
- the Cycle 17 opportunity measures the actual VLA action response of inserting
  each task-vector group and solves a constrained action-function merge;
- no probabilistic KL or invalid action-distribution interpretation is used.

## Positive Prior 2: SimpleVLA-RL

Full title: SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning.

Primary sources:

- paper: https://arxiv.org/abs/2509.09674
- official code: https://github.com/PRIME-RL/SimpleVLA-RL

Positive result:

- online outcome-based RL applied to OpenVLA-OFT reports state-of-the-art
  LIBERO performance and stronger task generalization than SFT;
- the official repository includes LIBERO RL launch support;
- the paper reports behavior beyond the demonstration strategy, establishing
  that reward-directed VLA adaptation can add capability rather than merely
  reduce imitation loss.

Local limitation:

- a fair on-policy implementation needs many simulator trajectories and
  noncollapsed successes and failures before a bounded consumer-GPU prototype
  can estimate stable paired-reset advantages;
- this axis is viable but materially more expensive and less data-certain than
  post-finetuning model retention.

## Positive Prior 3: FlowPolicy

Full title: FlowPolicy: Enabling Fast and Robust 3D Flow-based Policy via
Consistency Flow Matching for Robot Manipulation.

Primary sources:

- paper: https://arxiv.org/abs/2412.04987
- official code: https://github.com/zql-kk/FlowPolicy

Positive result:

- consistency flow matching reports a `7x` inference-speed improvement while
  retaining competitive manipulation success;
- the method constrains velocity fields from different time states to reach a
  consistent action endpoint.

Local limitation:

- FlowPolicy is a 3D point-cloud policy rather than a VLA adaptation method;
- a SmolVLA proxy would replace or materially retrain the action generator;
- prior local LIFT, RAR, EAC, and action-head results make disruption risk high,
  even though training-time consistency is not identical to those methods.

## Cycle 17 Opportunity

The strongest bounded opportunity is `FAMR-VLA`, Function-Aware Model
Retention for VLA policies.

Let `theta_0` be the frozen 40-task SmolVLA checkpoint and `theta_ft` a
new-task finetuned endpoint. Partition the effective task vector
`Delta = theta_ft - theta_0` into preregistered groups `Delta_m`. For a legal
development observation `x_i`, measure the functional response

`d_im = a(x_i; theta_0 + Delta_m) - a(x_i; theta_0)`.

Stacking group responses gives `D_i`. FAMR solves bounded coefficients
`c in [0,1]^M` using target demonstration rows `T` and original-task retention
rows `R`:

`min_c mean_T Huber(a_0i + D_i c - a_i*) + lambda mean_R ||D_i c||_2^2`.

It then materializes one policy

`theta_famr = theta_0 + sum_m c_m Delta_m`

and validates the actual nonlinear policy before rollout. The first term keeps
new-task capability; the second directly limits original-task action drift.
The key ablation sets `lambda = 0`. The closest-prior arm uses scalar RETAIN
interpolation. Standard LoRA is a required fifth control because ordinary
new-task adaptation is a plausible explanation.

`SCIENTIFIC_METHOD`: constrained model merging in VLA action-function space.

`LOW_COMPUTE_PARAMETERIZATION`: a matched zero-effect rank-4 SmolVLA LoRA task
vector used only to make the endpoint and grouped merge feasible locally. The
method definition remains valid for full checkpoints.

This is not an IARC rescue. IARC modifies clean-versus-perturbed gradients
during robustness consolidation on the original 40 tasks. FAMR operates after
new-skill finetuning, compares checkpoint task-vector groups, and optimizes a
new-task-versus-generalist functional retention tradeoff. It uses new
discovery/validation identities and preserves IARC's result unchanged.
