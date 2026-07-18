# Action-Consistent Missing-View Distillation: Frozen X-VLA Method Specification

Decision: `ACTION_CONSISTENT_MISSING_VIEW_DISTILLATION_XVLA_METHOD_SPEC_FROZEN`

Novelty status remains `INCREMENTAL_BUT_POTENTIALLY_PUBLISHABLE`. This is one
method, not a renamed RIFA/CVLR revision. No training, optimizer step,
closed-loop rollout, or confirmatory-outcome access occurred while freezing it.

The authoritative machine-readable contract is
[`configs/action_consistent_missing_view_distillation_xvla_frozen_spec.json`](../configs/action_consistent_missing_view_distillation_xvla_frozen_spec.json).

## Scientific mechanism

The clean teacher and dropout student use the same frozen official X-VLA
checkpoint, action preprocessing, 30-step horizon, diffusion state, and noise.
The teacher sees synchronized agent and valid wrist views. The student sees the
matched agent view, processor-equivalent black wrist pixels, the unchanged
official sample `image_mask=[true,true,false]`, and `missing_indicator=1`.
This is the exact already-frozen `mask_1_in_hand_dropout` implementation.

The real student attachment is the output of `model.transformer.norm` before
the official frozen `action_decoder`. A 0-initialized residual adapter maps each
dropout action hidden state through a 1024-to-128 shared trunk and back to 1024
dimensions. The residual is bounded per coordinate by `0.1`.

The same shared trunk is pooled by a training-only decoder that predicts the
clean 50x1024 wrist Florence2 block. Its MSE supplies auxiliary representation
supervision. The predicted wrist block is never an input to the action residual,
X-VLA token stream, action decoder, or deployment graph. This preserves the
frozen closure of CVLR v1 direct token insertion while retaining its verified
cross-view learning signal.

## Exact tensors and objectives

| Tensor | Shape | Role |
|---|---:|---|
| Clean wrist target | `[B,50,1024]` | Training-only auxiliary target |
| Clean teacher action hidden | `[B,30,1024]` | Full-chunk hidden alignment target |
| Clean teacher raw action | `[B,30,20]` | Translation, rotation, and raw-gripper targets |
| Dropout hidden before adapter | `[B,30,1024]` | Legal current-observation student feature |
| Student hidden after adapter | `[B,30,1024]` | Input to frozen action decoder |
| Student raw action | `[B,30,20]` | Pre-sigmoid continuous and gripper output |
| Predicted wrist latent | `[B,50,1024]` | Training-only reconstruction output |

X-VLA exposes a continuous flow action head rather than an action-token
distribution, so the representation target is the action hidden state. The
primary action-semantic composite is:

- `0.25 *` normalized full-chunk action-hidden MSE;
- `1.0 *` normalized translation MSE at indices `0,1,2,10,11,12`;
- `1.0 *` normalized 6D-rotation MSE at indices `3..8,13..18`; and
- `1.0 *` normalized raw gripper-margin MSE at indices `9,19` relative to raw
  threshold `0` (post-sigmoid threshold `0.5`).

The single auxiliary term is `0.25 *` normalized wrist-latent MSE. Objective
denominators come from fixed discovery Base-dropout-versus-clean-teacher rows
and are mechanically frozen before optimization. They are not tuned from
validation outcomes.

## Trainable and frozen scope

The module contains `shared_down`, `shared_core`, `action_residual_output`, a
50x128 wrist-token position table, `reconstruction_core`, and
`reconstruction_output`.

- exact Stage 0 trainable parameters: `434,816`;
- exact deployment parameters: `279,808`;
- official teacher trainable parameters: `0`;
- official student-backbone trainable parameters: `0`;
- full-model fine-tuning: prohibited;
- CPU/disk model offload and swap/pagefile training: prohibited.

Both output projections start at exact zero. On clean input the hook is not
activated, so execution takes the exact official X-VLA path. The module also
returns the original hidden tensor object immediately for an all-clean batch.

## Frozen arms and comparator questions

| Role | Frozen policy | Scientific question |
|---|---|---|
| Base | Frozen official X-VLA dropout | Is the verified condition improved? |
| External prior | `MECHANISM_FAITHFUL_RL4IL_LOCAL_PORT` | Is there performance or prespecified deployment Pareto advantage? |
| Ours | Full action-consistent distillation | Does the complete mechanism act? |
| Key ablation | No reconstruction | Does training-only cross-view supervision add practical value? |
| Mechanism ablation | No raw gripper margin | Does separate threshold-aware gripper supervision matter? |
| Generic control | Ordinary dropout adapter | Can generic demonstration-supervised adaptation explain the gain? |

The four trainable Stage 0 arms have identical `434,816`-parameter capacity,
data exposure, effective batch, optimizer-step budget, initialization, and
checkpoint rule. RL4IL retains its local-port label and is never described as
an official reproduction. AWF, CVLR v1 insertion, and zero-fill remain archived
diagnostics.

## Data, optimization, and checkpoint rule

Discovery uses the three frozen residual tasks, demonstrations `0..39`, and
official-reader positions `0,9,18,27` (`480` records). Validation uses demo
`40` at the same positions (`12` records). Demos `41..49` (`108` records) are
an inaccessible one-confirmation reserve. Confirmatory simulator outcomes are
untouched.

Each arm receives `128` AdamW optimizer steps, effective batch `8`, peak
learning rate `3e-4`, an eight-step warmup, and cosine decay to `3e-5`.
Microbatch is not guessed: the actual teacher/student/backward path tests
`1,2,4,8`, stopping after the first unsafe size and choosing the largest safe
prefix member. Checkpoints are written at steps `64` and `128`; only the final
step is selected, without validation checkpoint selection.

## Practical Stage 0 signal

The full method must improve over no-reconstruction on at least one of four
separate teacher-agreement measures—translation RMSE, rotation RMSE, raw
gripper-margin MAE, or action-hidden MSE—by both at least 5% relatively and at
least `max(frozen absolute floor, 10 x repeated-forward numerical noise)`.
Other measures may not materially regress, and full may not add a gripper
disagreement. Reconstruction MSE must be at most `0.95` times the
no-reconstruction result overall and lower on at least two tasks.

Full must similarly beat the generic adapter on at least one teacher-agreement
measure or the result is `STAGE0_GENERIC_ADAPTATION_EXPLAINS_GAIN`. A merely
nonzero difference is never sufficient. Conversely, Stage 0 is a validity and
directional mechanism gate, not the final closed-loop superiority estimate;
the fixed false-negative safeguard governs unresolved noncatastrophic effects.

## Legal inference and exact bypass

Dropout inference runs the official processor, official frozen VLM and action
transformer, the 279,808-parameter action residual, the frozen action decoder,
and official denoising/postprocessing. It uses no clean teacher, future frame,
expert/demo action, reward, success/done flag, simulator object/contact/pose
state, privileged reset identity, retrieval database, nearest-neighbor search,
or reconstructed wrist input. The reconstruction decoder is absent from the
inference export.

Latency uses 10 warmups and 100 measured fixed-input queries, reporting total
and adapter-only mean/median/p95. Peak allocated/reserved VRAM, system RAM,
pagefile growth, checkpoint bytes, inference-state bytes, and hashes are
mandatory.

## Simulation-only evidence path

The [simulation-only calibration](simulation_only_ral_evidence_calibration_result.md)
does not alter Stage 0. Stage A starts at three tasks by three identities and
expands once to five only at its frozen uncertainty boundary. Stage B uses at
least four tasks and three wrist-camera failure conditions, starts at 60
paired failure rows per policy, and expands once to 80 only when the frozen
claim interval overlaps its boundary.

A second backbone and camera-only validation are optional strengthening
evidence, not universal paper gates. A simulation-only candidate must instead
satisfy the stronger task/condition/ablation/uncertainty/resource gate and may
claim only `ROBUST VLA MANIPULATION UNDER SIMULATED WRIST-CAMERA FAILURES`.
Physical manipulation remains prohibited.

## Specification validation

The focused test suite verifies schema invariants, exact parameter counts,
zero-initialized identity, exact clean object/value bypass, the absence of a
reconstruction-to-action-output connection, and reconstruction-free inference
export: `5 passed`.

The subsequent actual-path preflight did not reach a valid Stage 0. Its frozen
decision is `STAGE0_IMPLEMENTATION_OR_RESOURCE_FAILURE`; see
[the Stage 0 result](action_consistent_missing_view_distillation_stage0_result.md).

Pre-execution erratum: the first pushed specification prose incorrectly called
the condition an image-mask dropout. Code inspection before any optimizer or
preflight step confirmed that the frozen RIFA/CVLR condition instead replaces
only wrist pixels with the processor-equivalent black tensor and leaves the
official image mask unchanged. Correcting that description changes no data,
mechanism, loss, threshold, or budget and does not consume the one bounded
implementation repair.

The one bounded implementation repair was consumed by the official-reader
import initialization boundary. The first worker showed that the pinned X-VLA
source root had not entered `sys.path`; after registering it, the unchanged
rerun showed that the repository's existing optional `mmengine.fileio` shim
also ran too late, after the official reader import. Both failed runs, results,
exit codes, and partial materializations remain under the two `noise_calibration`
run directories timestamped `024502KST` and `024907KST`. Completing the single
repair moves both already-existing initialization operations before the reader
import. It changes no row, loss, threshold, repetition, budget, dependency, or
model and authorizes no further repair.

The final unchanged rerun materialized all 12 fixed rows, confirming that the
reader repair cleared its declared boundary, but then failed at
`torch.cuda.reset_peak_memory_stats(device)` with `Invalid device argument`
before model load or any teacher/student forward. Because repair count remains
`1 / 1`, the distinct device-runtime defect cannot be repaired under the
frozen contract. No optimizer, microbatch, checkpoint, Stage A/B, or
confirmatory outcome was reached. This is an implementation/resource failure,
not a mechanism result.
