# Epoch 5 Task-1 Ours Candidate Design

Status: `EXACTLY_TWO_CANDIDATES_GENERATED_ONE_SELECTED_NO_TRAINING`.

Decision: `BR_XVLA_GRADIENT_SMOKE_PASS_TRAINING_LAUNCHER_PENDING`.

## Preconditions

- Official prior: X-VLA-Libero on LIBERO.
- Matched residual: `libero_10/task_1`, shared X-VLA/SmolVLA Base failure
  `20260727`, initial-state index `16`.
- Headroom: task-level HDF5 expert replay positive, but same-reset HDF5 expert
  unavailable.
- Data health: task1 basket data audit passed with 50 demos, 13,021 steps,
  12,671 chunks, 4,607 train and 1,079 validation one-target-remaining chunks,
  and zero residual init-state hash overlap.
- Out of scope: `20260725`, because SmolVLA Base succeeds there and X-VLA
  regresses.

## Candidates

| Candidate | Contribution type | Core mechanism | Score | Decision |
|---|---|---|---:|---|
| `BR-XVLA`: Basket-Remaining Reweighted X-VLA | `PRIOR_EXTENSION` | Keep the X-VLA-Libero observation/action interface, but LoRA/QLoRA-adapt with a phase-balanced imitation objective that upweights successful task1 HDF5 chunks where exactly one of cream cheese or butter is already in/near the basket and the remaining target still needs completion. | 86/100 | SELECTED |
| `OCB-XVLA`: Object-Contrast Basket X-VLA | `PRIOR_EXTENSION` | Balance cream-cheese-first and butter-first supervision with object-role/order contrast during adaptation, aiming to reduce object-order brittleness in two-target basket completion. | 73/100 | NOT SELECTED |

Selection rationale: `BR-XVLA` is the narrowest mechanism supported by the
measured residual and the task1 data audit. `OCB-XVLA` is plausible but adds a
broader object-role/order hypothesis before we have evidence that object order,
rather than the one-target-remaining completion phase, is the primary failure
axis.

## Selected Method Sketch: `BR-XVLA`

Let `x_t = (I_t, W_t, p_t, l)` be X-VLA's deployment input: agent-view RGB,
wrist RGB, proprioception, and instruction. Let `a*` be the expert action
chunk. Let `m_t = 1` only when training-only HDF5 state labels show exactly one
target object in the basket region and the other target object outside it.

The first bounded adaptation objective is:

`L(theta) = mean_t (1 + lambda * m_t) * ||pi_theta(x_t) - a*||_1`.

LoRA/QLoRA is infrastructure only. The scientific claim is not "more finetune";
it is that basket-remaining phase-balanced supervision fixes a stronger-prior
two-object completion residual while preserving non-residual phases.

## Frozen Boundaries Before Optimizer Steps

- Exactly two training arms are allowed in the first spec: primary `BR-XVLA`
  and uniform-weight ablation.
- Residual identities must not be used for model selection or retuning.
- Inference inputs may not include simulator object state.
- Closed-loop Ours evaluation is disallowed until an offline validation gate
  passes.
- If the frozen closed-loop residual evaluation fails, do not generate a new
  configuration from that failure.

Update: the no-training `BR-XVLA` spec is frozen at
`runs/xvla_prior/epoch5_br_xvla_training_spec_v1.json`, and the tiny
X-VLA-format data-adapter smoke passed at
`runs/xvla_prior/br_xvla_data_adapter_smoke_20260717T183355KST/result.json`.
The one-batch no-optimizer gradient smoke passed at
`runs/xvla_prior/br_xvla_gradient_smoke_20260717T190919KST/result.json`.
Next action is the bounded two-arm training launcher/offline-validation gate;
closed-loop Ours evaluation remains disallowed until offline validation passes.
