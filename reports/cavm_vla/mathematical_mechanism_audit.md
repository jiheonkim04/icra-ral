# CAVM-VLA Mathematical Mechanism Audit

Date: 2026-07-13 KST

## Variables And Shapes

- `q_t in R^8`: official SmolVLA proprioceptive state.
- `a_t in R^7`: current frozen queued action after official postprocessing.
- `a_{t-1} in R^7`: previous executed action, zero at episode start.
- `rho_t in R`: chunk-index fraction.
- `e_task in {0,1}^2`: task one-hot.
- `z_t in R^25`: retrieval key `[q_t, a_t, a_{t-1}, rho_t, e_task]`.
- `M+`: successful trace records from acquisition identities.
- `M-`: failed trace records from acquisition identities.
- `a_i+ in R^7`: action attached to successful memory record `i`.
- `a_j- in R^7`: action attached to failed memory record `j`.

## Representation Learned

CAVM does not train a neural network in the first prototype. It learns a standardized non-parametric memory representation:

- feature mean and scale for `z_t`;
- retrieval bandwidth `sigma`;
- contrast margin `eta`;
- margin scale `gamma`;
- separated success and failure trace memories.

## Policy Component Affected

The frozen SmolVLA parameters, tokenizer, image processors, action queue, and official action semantics are unchanged.

CAVM changes only the executed 7D action after the frozen policy proposes it:

`a_t -> a'_t`.

## Objective And Supervision Source

There is no gradient objective.

Supervision source:

- terminal episode success/failure from official LIBERO rollout on acquisition identities.

This label is used only to partition trace memory into `M+` and `M-`.

## Inference Algorithm

For the current key `z_t`, retrieve same-task nearest records from `M+` and `M-` using standardized Euclidean distance.

Compute kernel weights:

`w_i+ = exp(-D(z_t, z_i+) / sigma) / sum_k exp(-D(z_t, z_k+) / sigma)`.

`w_j- = exp(-D(z_t, z_j-) / sigma) / sum_k exp(-D(z_t, z_k-) / sigma)`.

Compute action means:

`mu+(z_t) = sum_i w_i+ a_i+`.

`mu-(z_t) = sum_j w_j- a_j-`.

Contrast vector:

`v(z_t) = mu+(z_t) - mu-(z_t)`.

Margin:

`m(z_t) = ||v(z_t)||_2`.

Gate:

`c_t = density_gate(z_t) * clip((m(z_t) - eta) / gamma, 0, 1)`.

Full action:

`a'_t = clip_action((1 - alpha c_t) a_t + alpha c_t (mu+(z_t) + beta v(z_t)))`.

## Each Term Justification

### Retrieval Distance `D`

Quantities compared:

- current key `z_t`;
- memory key `z_i`.

Why appropriate:

- CAVM needs local trace neighborhoods in the same proprioceptive/action phase coordinate system used at inference.

Stage:

- calibration and inference.

Gradient:

- none.

Behavior induced:

- retrieves memory actions from similar frozen-policy states.

Simpler alternative:

- task-level mean action.

Ablation:

- nearest-success replay and success-only memory proxy.

### Success Mean `mu+`

Quantities compared:

- actions from nearby successful trace records.

Why appropriate:

- this is the direct success-memory prior and closest external-prior proxy.

Stage:

- inference.

Gradient:

- none.

Behavior induced:

- shifts action toward locally successful behavior.

Simpler alternative:

- single nearest successful action.

Ablation:

- nearest-success replay.

### Failure Mean `mu-`

Quantities compared:

- actions from nearby failed trace records.

Why appropriate:

- the proposed novelty is that failure traces can identify action directions to avoid when success and failure neighborhoods are both present.

Stage:

- Stage 0 diagnostic and inference.

Gradient:

- none.

Behavior induced:

- shifts away from locally failure-associated action directions.

Simpler alternative:

- remove failure memory and use success-only mean.

Ablation:

- `cavm_no_contrast_ablation` and `success_only_memory_proxy`.

### Contrast Margin `m`

Quantities compared:

- `mu+` and `mu-` action means.

Why appropriate:

- a contrastive action prior is meaningful only if nearby successful and failed action means differ.

Stage:

- Stage 0 hard gate, calibration, inference.

Gradient:

- none.

Behavior induced:

- blocks interventions when contrast is weak.

Simpler alternative:

- always blend toward success mean.

Ablation:

- success-only memory proxy.

### Gate `c_t`

Quantities compared:

- local memory density and contrast margin.

Why appropriate:

- prevents action changes when either memory support or success/failure separation is insufficient.

Stage:

- inference.

Gradient:

- none.

Behavior induced:

- sparse, high-confidence action prior intervention.

Simpler alternative:

- fixed blend coefficient.

Ablation:

- nearest-success replay uses the same maximum blend and density gate but no contrast direction.

## Expected Behavioral Effect

CAVM should make small, sparse action changes on held-out identities, moving actions toward local successful trace directions and away from local failed trace directions.

## Expected Closed-Loop Consequence

If the local trace contrast is causal rather than spurious, `cavm_full` should improve task-balanced closed-loop success over frozen SmolVLA, success-only memory, nearest-success replay, and no-contrast ablation.

## Closest Mathematical Alternative

Success-only kernel regression:

`a'_t = (1 - alpha c_t) a_t + alpha c_t mu+(z_t)`.

This is included as a baseline/proxy and must be weaker than full CAVM.

## Simplest Equivalent Baseline

Nearest-success action replay:

`a'_t = (1 - alpha) a_t + alpha a_nearest_success`.

If this matches CAVM, the method is dead.

## Known Failure Modes

1. Terminal labels are too coarse and mark useful failed-prefix actions as negative.
2. The memory key retrieves phase-mismatched states.
3. Success/failure separation is caused by reset identity rather than action value.
4. The action adjustment destroys gripper/contact timing.
5. The no-contrast ablation or success-only proxy explains all gains.

## Divergence Policy

CAVM uses no KL divergence, entropy term, mutual information term, contrastive loss, or probability-distribution distance. All distances are explicit vector distances over standardized features or 7D actions.
