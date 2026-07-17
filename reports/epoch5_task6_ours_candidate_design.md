# Epoch 5 Task-6 Ours Candidate Design

Status: `MPR_XVLA_ARCHIVED_OFFLINE_NO_PASS_PRC_NOT_ELIGIBLE`.

Decision:
`TASK6_MPR_XVLA_OFFLINE_NO_PASS_RETURN_TO_PRIOR_RESIDUAL_MINING`.

## Preconditions

- Official prior: X-VLA-Libero on LIBERO.
- Matched residual: `libero_10/task_6`, shared X-VLA/SmolVLA Base failures
  `20260725` and `20260731`.
- Headroom: task-level HDF5 expert replay positive for both shared residuals,
  but same-reset HDF5 expert evidence unavailable.
- Data health: task6 spatial audit passed with 50 demos, 12,756 steps, 12,406
  chunks, 5,518 train and 1,372 validation mug-done/pudding-remaining chunks,
  all demos mug-first, and zero residual init-state hash overlap.
- Second-prior screen: Quantized OpenVLA-OFT INT4 completed both shared residual
  identities with zero infrastructure failures and zero successes.
- Out of scope: BR-XVLA retuning/rescue, generic local heads, residual gates,
  memory, verifiers, cached-feature probes, or proxy-only methods.

## Candidates

| Candidate | Contribution type | Core mechanism | Score | Decision |
|---|---|---|---:|---|
| `MPR-XVLA`: Mug-placed / Pudding-right Reweighted X-VLA | `PRIOR_EXTENSION` | Keep X-VLA-Libero's observation/action interface, but LoRA/QLoRA-adapt with a phase-balanced imitation objective that upweights successful task6 HDF5 chunks where the white mug is already on the plate and the chocolate pudding still needs the right-of-plate relation. | 88/100 | SELECTED |
| `PRC-XVLA`: Pudding-Right Contrast X-VLA | `PRIOR_EXTENSION` | Add relation/distractor contrast around pudding-right-of-plate versus red-mug/plate distractor geometry during adaptation. | 74/100 | NOT SELECTED |

Selection rationale: `MPR-XVLA` is the narrowest mechanism supported by the
measured residual and the data audit. The audit shows a large, non-collapsed
mug-done/pudding-remaining phase in every train and validation demo. `PRC-XVLA`
is plausible but adds a broader relation-contrast hypothesis before there is
evidence that red-mug distractor confusion, rather than second-subgoal completion
after mug placement, is the primary residual axis.

## Selected Method Sketch: `MPR-XVLA`

Let `x_t = (I_t, W_t, p_t, l)` be X-VLA's deployment input: agent-view RGB,
wrist RGB, proprioception, and instruction. Let `a*` be the expert action
chunk. Let `m_t = 1` only when training-only HDF5 state labels show the white
mug in the plate region and the chocolate pudding not yet in the right-of-plate
region.

The first bounded adaptation objective is:

`L(theta) = mean_t (1 + lambda * m_t) * ||pi_theta(x_t) - a*||_1`.

LoRA/QLoRA is infrastructure only. The scientific claim is not "more finetune";
it is that mug-placed/pudding-remaining phase-balanced supervision fixes a
stronger-prior two-object spatial residual while preserving non-residual phases.
Simulator object state is never an inference input.

## Frozen Boundaries Before Optimizer Steps

- Exactly two training arms are allowed in the first spec: primary `MPR-XVLA`
  and uniform-weight X-VLA LoRA/QLoRA ablation.
- The uniform ablation is mandatory because the task1 BR-XVLA screen showed
  that uniform adaptation can explain an apparent residual fix.
- Residual identities `20260725` and `20260731` must not be used for model
  selection or retuning.
- Inference inputs may not include simulator object state.
- Closed-loop Ours evaluation is disallowed until an offline validation gate
  passes.
- If a frozen closed-loop residual evaluation fails, do not generate a new
  configuration from that failure.

## Post-Training Outcome

`MPR-XVLA` completed the frozen two-arm training/offline-validation gate after
preoptimizer data-adapter and gradient smokes passed. The first full gate failed
only during offline validation from an infrastructure scope bug; after commit
`5faed4dabc7a97fa3d24cfbfafb9262dda6be8f1`, the saved adapters were revalidated.

Repaired offline validation:

- Artifact:
  `runs/xvla_prior/epoch5_mpr_xvla_offline_validation_step0064_repaired_20260717T2200KST.json`
- SHA-256:
  `ede498006a5832e3b2101de41fd344438b1c2dc4cdbd21f1355d8564e03fc59f`
- Decision: `MPR_XVLA_OFFLINE_SELECTION_NOT_PASSED`
- Prior phase-1 mean loss: `2.9992184042930603`
- `MPR-XVLA` phase-1 mean loss: `0.8785358915726343`
- Uniform-weight X-VLA LoRA phase-1 mean loss: `0.8785358369350433`

Interpretation: `MPR-XVLA` passed the absolute offline-health gate but did not
beat the required uniform-weight ablation. Closed-loop `MPR-XVLA` Ours
evaluation is disallowed, and retuning this configuration from the no-pass is
not authorized.

## Runner-Up Candidate Decision

`PRC-XVLA` remains `NOT_SELECTED` and is not elevated after the `MPR-XVLA`
no-pass. The reason is evidential rather than merely conservative: the original
selection rationale required evidence that red-mug/plate distractor confusion,
rather than generic task6 adaptation or second-subgoal completion, was the
dominant residual axis. The surviving data audit only shows that the red mug is
a stable off-plate distractor in demonstrations; it does not show that the
official prior failures confuse the red mug with the white mug or plate. Promoting
PRC now would convert an under-supported auxiliary contrast idea into another
local attachment after the uniform LoRA ablation already explained the offline
gain.

Next action: return to official-prior-first residual mining and look for a fresh
condition with matched Base/Prior residual, headroom, and a mechanism-specific
failure signal before proposing another Ours method.
