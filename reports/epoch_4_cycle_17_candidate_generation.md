# Epoch 4 Cycle 17 Candidate Generation

Date: 2026-07-15 KST

Decision: `SELECT_FAMR_VLA`

Exactly three candidates were generated and scored under the active
performance-oriented, false-negative, and post-COVI method-design governance.
IARC remains closed as an implementation/action-validity failure and is not
retuned or reinterpreted here.

## Candidate 1: FAMR-VLA

Name: `FAMR-VLA`, Function-Aware Model Retention for VLA policies.

Contribution type: `PRIOR_EXTENSION` plus `CROSS_DOMAIN_MECHANISM_TRANSFER`.

Closest external prior: RETAIN,
https://arxiv.org/abs/2512.08333, with official code at
https://github.com/yajatyadav/RETAIN_code.

Positive external result: RETAIN shows that interpolating a generalist VLA
checkpoint with a new-task finetuned checkpoint improves OOD target-task
generalization while retaining prior skills in simulated and real robot
experiments.

Secondary mechanism prior: Fisher-weighted model merging,
https://arxiv.org/abs/2111.09832.

### Scientific Method

FAMR replaces coefficient selection from parameter distance or a scalar sweep
with one constrained action-function merge. It probes each preregistered
task-vector group on development observations, predicts the merged action
response, and fits bounded group coefficients to new-task demonstration
actions while penalizing drift on original 40-task retention observations.

For group responses `D_i`, target action `a_i*`, and frozen action `a_0i`:

`L_FAMR(c) = mean_T Huber(a_0i + D_i c - a_i*)`
`              + lambda mean_R ||D_i c||_2^2`,

subject to `0 <= c_m <= 1`.

The materialized checkpoint is evaluated directly; the linear response is not
treated as confirmatory evidence.

Key ablation: `famr_target_only`, the same response probes, groups, solver,
checkpoint endpoint, and compute with `lambda = 0`.

Closest-prior proxy: `retain_scalar_proxy`, the same endpoint with a single
validation-selected scalar coefficient, labeled transparent rather than
official.

Strongest additional control: `standard_lora_new_task`, the same finetuned
endpoint with coefficient `1` for every group. This control is required because
generic adaptation is a plausible explanation for any target-task gain.

### Mechanism Chain

- limited new-task finetuning -> uneven task-vector movement across VLA modules;
- uneven movement -> some groups add new-task action competence while others
  cause disproportionate original-task action drift;
- scalar merging -> one coefficient cannot separate useful from disruptive
  functional effects;
- excessive drift -> loss of generalist closed-loop skill or invalid actions;
- FAMR probes groupwise action response -> contribution and disruption become
  observable in deployment units;
- constrained coefficients -> retain useful target-task response while
  attenuating high-drift groups;
- expected action effect -> lower target error than Base/RETAIN with less clean
  drift than standard LoRA/target-only;
- expected closed-loop effect -> better new-task success than Base and RETAIN,
  with original-task success retained.

### Data And Integration

- new-task demonstrations: local official raw `libero_90` HDF5 files;
- original-task retention: discovery/validation rows from the 40-task
  `lerobot/libero` dataset;
- confirmatory task/reset rows remain sealed until configuration freeze;
- no privileged state or success signal is required at inference;
- task-vector groups and coefficients are checkpoint-only;
- coefficients are bounded in `[0,1]`, and zero-effect LoRA exactly reproduces
  Base before training.

Stage 0 must reject implementation/data failure before rollout if the raw
LIBERO-90 action mapping is invalid, the endpoint cannot fit a small subset,
group responses are nonacting or rank deficient, the actual materialized
checkpoint disagrees catastrophically with the response model, action validity
worsens beyond the preregistered Base-relative tolerance, or no new-task
headroom exists.

### Bounded Search

Maximum six total selection configurations:

- fine-group FAMR with `lambda in {0.1, 1.0, 10.0}`: three;
- coarse-group FAMR with `lambda = 1.0`: one;
- scalar RETAIN with `alpha in {0.5, 0.8}`: two;
- one seed for deterministic response fitting;
- one shared standard-LoRA endpoint;
- no confirmatory identity used.

The selected-group `lambda = 0` ablation, standard-LoRA endpoint, and Base are
fixed scientific controls rather than selectable configurations.

The validation score must combine target-task closed-loop success or the
closest feasible proxy, original-task clean retention, action validity,
response-model fidelity, and compute overhead. It may not select purely by
offline action L2.

### Score

- provisional novelty: `22 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `19 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `93 / 100`

## Candidate 2: PARF-VLA

Name: `PARF-VLA`, Paired-Advantage Reweighted Flow for VLA policies.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external priors:

- SimpleVLA-RL: https://arxiv.org/abs/2509.09674
- ForesightFlow: https://arxiv.org/abs/2606.04968

Positive external result: SimpleVLA-RL reports state-of-the-art LIBERO and
RoboTwin results from outcome-based VLA reinforcement learning. ForesightFlow
reports that advantage-weighted action-flow training improves over imitation
and separate-critic baselines.

Scientific method: collect multiple development rollouts under the same
`(task, reset_identity)`, center binary returns within each reset group, and
weight SmolVLA action-flow loss by the paired advantage. This controls reset
difficulty without a critic or privileged inference input.

Primary objective: paired-advantage-weighted flow matching.

Key ablation: globally centered reward-weighted flow matching with identical
rollouts and weights but no reset grouping.

Main risk: label health is unknown until enough successes and failures exist
within the same reset. Collecting that evidence and training a stable policy is
substantially more expensive than FAMR. Sparse all-zero/all-one groups would be
`DATA_OR_SUPERVISION_FAILURE`, not a scientific kill.

Score:

- provisional novelty: `21 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `5 / 10`
- decisive experiment feasibility: `5 / 10`
- total: `83 / 100`

## Candidate 3: TCFM-VLA

Name: `TCFM-VLA`, Teacher-Consistent Flow Matching for VLA policies.

Contribution type: `CROSS_DOMAIN_MECHANISM_TRANSFER`.

Closest external prior: FlowPolicy,
https://arxiv.org/abs/2412.04987, with official code at
https://github.com/zql-kk/FlowPolicy.

Positive external result: FlowPolicy reports a `7x` speed increase with
competitive manipulation success by enforcing flow consistency and generating
actions in one step.

Scientific method: train the SmolVLA action generator so a low-NFE student
reaches the frozen multi-step teacher endpoint from two flow times using one
self-consistency objective.

Primary objective: endpoint distillation from the frozen native SmolVLA
multi-step sampler.

Necessary auxiliary: cross-time endpoint consistency.

Key ablation: endpoint distillation without the cross-time consistency term.

Main risk: this changes the generative action decoder and therefore has high
identity-disruption risk. Its point-cloud prior is not same-backbone, and local
LIFT/action-head evidence makes a faithful decisive proxy less direct. It is
not identical to LIFT because it is training-time distillation, but it sits too
near a crowded and locally fragile action-generation axis for Cycle 17.

Score:

- provisional novelty: `18 / 25`
- importance of problem: `12 / 15`
- strength of positive prior anchor: `15 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `6 / 10`
- total: `75 / 100`

## Selection

`FAMR-VLA` is selected with `93 / 100`.

It has the strongest same-claim positive prior and official code, uses a local
held-out new-task source that was not in the 40-task SmolVLA checkpoint, makes
policy disruption observable in action units, and supports a bounded
development search before confirmatory evaluation. Its method is checkpoint
and function-space retention, not LoRA; rank-4 LoRA is only the local task
vector parameterization.

Unknown empirical performance is not a rejection reason. The next step is a
Researcher A proposal, Reviewer B attack, rebuttal, mathematical audit, and
preregistration. No training, validation search, or rollout is authorized by
candidate selection alone.

## First Serious Comparison

| Policy | Scientific question |
| --- | --- |
| `smolvla_base` | Does any adaptation improve held-out new-task success? |
| `retain_scalar_proxy` | Does action-function merging improve over the closest positive parameter-merging prior under the same endpoint? |
| `famr_full` | Does constrained functional merging add new-task capability while retaining original skills? |
| `famr_target_only` | Is the original-task functional retention term necessary? |
| `standard_lora_new_task` | Can ordinary matched new-task adaptation explain the gain? |

Exactly five policies are justified. No sixth internal control enters the first
serious experiment without a new concrete alternative explanation.
