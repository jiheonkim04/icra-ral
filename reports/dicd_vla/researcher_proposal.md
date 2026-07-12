# Researcher Proposal: DICD-VLA

Date: 2026-07-12 KST
Branch: `codex/auto-method-20260712-01-dicd-vla`
Role: Researcher A

Method name: `Delay-Indexed Chunk Distillation for VLA Policies`

## Exact Hypothesis

When a VLA policy emits an action chunk, controlled execution delay can be compensated better by a delay-indexed chunk adapter than by executing the stale first action or by directly taking the delayed chunk index. Under artificial delay `d > 0`, DICD-VLA should improve task-balanced closed-loop success over frozen SmolVLA and over direct chunk-index execution while retaining clean `d=0` behavior.

## Method

DICD-VLA treats the action chunk as an explicit prediction of near-future control. At each control step, frozen SmolVLA produces a chunk:

`A_t = [a_t^0, a_t^1, ..., a_t^{H-1}]`

The deployment has a known action delay `d`, so the command chosen at observation time `t` is executed after the robot has advanced. The full method learns a lightweight adapter:

`g_theta(A_t, h_t, d, z_t) -> a_exec`

where:

- `A_t` is the frozen VLA action chunk after official postprocessing;
- `h_t` is a compact history of recent executed actions;
- `d` is the declared delay in control steps;
- `z_t` is low-dimensional timing/proprioceptive metadata available at inference, such as step fraction and previous action delta.

The adapter is intentionally small and operates after the VLA action chunk, but it is not a scalar safety filter, hold rule, candidate ranker, or confidence head. Its only purpose is to learn which chunk/residual action should be executed for a known delayed actuator.

## Equations

For training examples from a trace with aligned action chunks and expert/control actions, define the target:

`y_t^d = a_{t+d}^{target}`

The model predicts:

`a_hat_t^d = g_theta(phi(A_t, h_t, d, z_t))`

The primary supervised objective is:

`L_delay(theta) = mean_t ||a_hat_t^d - y_t^d||_1`

Clean retention is regularized by:

`L_clean(theta) = mean_t ||g_theta(phi(A_t, h_t, 0, z_t)) - a_t^0||_2^2`

Total loss:

`L(theta) = L_delay(theta) + lambda_clean L_clean(theta) + lambda_smooth mean_t ||a_hat_t^d - a_exec_{t-1}||_2^2`

## Supervision

Training uses locally generated or existing LIBERO traces:

- frozen SmolVLA action chunks from the official stack;
- executed action history;
- target future actions from the same trace offset by delay `d`;
- no task-success reward labels;
- no human intervention or correction chunks;
- no privileged simulator value at inference.

If demonstration-action alignment is available, targets may be demonstration future actions. Otherwise, the first prototype may use self-distilled frozen-policy future actions and test whether the learned adapter can outperform direct chunk indexing under controlled delay.

## Inference

At each step:

1. preprocess the current observation and instruction with official SmolVLA processors;
2. call `predict_action_chunk`;
3. official-postprocess the whole chunk into environment action space;
4. build DICD features from the chunk, recent executed actions, delay index, and step timing;
5. execute `g_theta(features)`;
6. update executed-action history.

No simulator state, success oracle, future observation, or task/reset identity is used at inference.

## Closest Known Papers

- DEFLECT: delay-robust execution with flow-matching likelihood-estimated counterfactual tuning, https://arxiv.org/abs/2605.19294
- TIC-VLA: slow/fast latency-consistent control for VLA navigation, https://arxiv.org/html/2602.02459v2
- RobustVLA: robustness-aware reinforcement post-training, https://arxiv.org/abs/2511.01331
- PAPO-VLA: planning-aware VLA optimization, https://arxiv.org/html/2605.19580v1
- OpenVLA-OFT: continuous action chunk fine-tuning substrate, https://arxiv.org/abs/2502.19645

## Exact Distinction

DICD-VLA differs from DEFLECT by not using flow-matching likelihood ratios, stale/fresh preference optimization, or online/offline policy post-training of the VLA backbone. It learns an explicit delay-indexed adapter over already generated action chunks and recent executed actions.

DICD-VLA differs from TIC-VLA by not introducing a slow semantic planner and fast controller architecture. It is a deployment adapter for existing chunk-emitting manipulation VLAs.

DICD-VLA differs from prior local methods because it is not candidate ranking, not phase-conditioned feasibility projection, not temporal hold blending, and not intervention-chunk fine-tuning.

## Prototype

Backbone: official SmolVLA-LIBERO.

Tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

Delay condition:

- first prototype delay: `d=2` control steps;
- clean retention check: `d=0`.

Variants:

1. `frozen_smolvla_clean`
2. `frozen_smolvla_delay`
3. `direct_chunk_index_delay`
4. `dicd_no_history_ablation`
5. `dicd_full`

Primary metric:

task-balanced official closed-loop task success.

Mechanism smoke before rollout:

- action chunks are finite and have `H > d`;
- training examples contain nonzero feature and target variation;
- trainable parameters receive nonzero finite gradients;
- training loss decreases;
- checkpoint is persisted and disk-reloaded;
- loaded checkpoint changes delayed actions relative to frozen and differs from no-history ablation;
- no privileged inference fields are present.

## Expected Failure Mode

The strongest expected failure is that the frozen VLA chunk does not contain useful delayed commands, so direct chunk-index execution or frozen delayed execution matches the learned adapter. If the full method cannot beat direct chunk indexing, DICD-VLA should be killed as `SIMPLE_BASELINE_EXPLAINS_METHOD` or `KEY_COMPONENT_NOT_USEFUL`, depending on mechanism evidence.
