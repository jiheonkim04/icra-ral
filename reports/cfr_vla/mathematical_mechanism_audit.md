# CFR-VLA Mathematical Mechanism Audit

Date: 2026-07-16 KST

Decision: `CFR_MATHEMATICAL_AUDIT_PREREGISTERED`

Proposal: `reports/cfr_vla/researcher_proposal.md`

Proposal hash:
`9E2FC510B2D97C869F18BE6C5B339CE034DD98223802078358320AA8BEF3D0AE`

Reviewer attack: `reports/cfr_vla/reviewer_attack.md`

Researcher rebuttal: `reports/cfr_vla/researcher_rebuttal.md`

This audit freezes the mathematical objective and the minimum diagnostics
required before preregistration, implementation, training, validation search, or
rollout.

## Narrow Method Claim

CFR-VLA claims only continuous Base-start identity-preserving full-chunk
refinement for frozen SmolVLA action chunks.

It does not claim to invent iterative action refinement, full-sequence action
correction, or discrete flow matching. Those are closest-prior territory for
DFM-VLA.

## Variables And Shapes

All action tensors below use normalized SmolVLA action units unless explicitly
marked as postprocessed environment actions.

| Symbol | Shape | Source | Gradient path | Meaning |
| --- | --- | --- | --- | --- |
| `I_t^0` | `[3,H,W]` | legal current image stream 0 | frozen encoder only unless adapter targets it | current RGB observation |
| `I_t^1` | `[3,H,W]` | legal current image stream 1 | frozen encoder only unless adapter targets it | wrist/secondary RGB observation |
| `s_t` | `[8]` | legal current proprioception | no gradient | official SmolVLA proprio/state vector |
| `l_t` | string/tokens | task language | frozen text path unless LoRA targets it | legal instruction input |
| `h_t` | `[960]` | frozen SmolVLA visual-policy feature hook | no Base-weight update | deployment-observable feature |
| `B_t` | `[50,7]` | frozen Base SmolVLA decoded chunk | stopgrad | Base action chunk |
| `A_t` | `[50,7]` | demonstration action chunk | target only | expert action chunk |
| `M_t` | `[50,1]` | valid-step mask | no gradient | masks padded/unavailable future actions |
| `p_t` | `[1]` or scalar | normalized phase | no gradient | chunk phase/timestep feature |
| `k` | scalar | refinement step | no gradient | integer step index `0..K-1` |
| `e_k` | `[d_k]` | learned or sinusoidal step embedding | trainable only if learned | step-conditioning feature |
| `C_t^0` | `[50,7]` | `stopgrad(B_t)` | no gradient into Base | initial chunk |
| `C_t^k` | `[50,7]` | unrolled CFR state | gradient through CFR unroll | refined chunk at step `k` |
| `u_t^k` | `[50,7]` | CFR network raw velocity | trainable | unconstrained velocity |
| `V_t^k` | `[50,7]` | bounded velocity | trainable through `tanh` | residual velocity field |
| `q_t^k` | `[50,1]` | CFR network raw gate | trainable | timestep gate logit |
| `g_t^k` | `[50,1]` | bounded gate | trainable through sigmoid/scale | refinement magnitude gate |
| `D_t^k` | `[50,7]` | stopgrad target | no gradient | per-step residual target |
| `C_t^K` | `[50,7]` | final CFR state | gradient through unroll | final normalized chunk |
| `E(C_t^K)` | `[50,7]` | official postprocessor/env conversion | no training gradient | postprocessed action chunk for validity |

## Refinement Dynamics

Initialize:

`C_t^0 = stopgrad(B_t)`

For `k = 0, ..., K-1`:

`u_t^k, q_t^k = F_theta(h_t, s_t, l_t, p_t, C_t^k, e_k)`

`V_t^k = r_max * tanh(u_t^k)`

`g_t^k = g_max * sigmoid(q_t^k + b_g)`

`C_t^(k+1) = C_t^k + M_t * g_t^k * V_t^k`

The gate bias `b_g` must initialize so that `g_t^k` is numerically zero within
the identity tolerance used by the repository (`1e-6` max action error after
disk reload). If exact zero cannot be obtained with sigmoid bias alone, the
implementation must use an explicit zero-initialized residual scale parameter
or gate multiplier.

The bound `r_max` is fixed before validation search. It is not a post-hoc
clipping rescue. If the bounded refinement still violates official action
semantics, CFR stops before rollout.

## Targets

For each refinement step:

`D_t^k = stopgrad((A_t - C_t^k) / max(1, K-k))`

`D_t^k` is a training target only. It does not create a gradient path through
`A_t` or through the target copy of `C_t^k`.

The final target is:

`A_t`

and the clean retention target is:

`stopgrad(B_t)`.

## Objective

Use Huber loss with preregistered coordinate-mean reduction:

`rho_delta(x) = 0.5 * x^2 / delta` if `|x| <= delta`, else
`|x| - 0.5 * delta`.

The default Huber `delta` is `0.05` normalized action units unless the
prototype protocol freezes another value before execution.

Velocity consistency:

`L_v = mean_{t,k,i,j} M_t[i] * rho_delta(V_t^k[i,j] - D_t^k[i,j])`

Terminal action consistency:

`L_T = mean_{t,i,j} M_t[i] * rho_delta(C_t^K[i,j] - A_t[i,j])`

Refinement smoothness:

`L_s = mean_{t,k,i,j} M_t[i] * rho_delta(C_t^(k+1)[i,j] - C_t^k[i,j])`

Clean retention:

`L_clean = mean_{t,i,j} M_t[i] * rho_delta(C_t^K[i,j] - B_t[i,j])`

Optional ordinary flow retention, only if the implementation modifies the
SmolVLA flow path:

`L_flow = mean flow-matching loss under the existing official SmolVLA training
contract`

If CFR is implemented as a post-decode refinement head without modifying Base
flow, set `L_flow = 0` and document that no Base flow gradient exists.

Total objective:

`L = L_flow
   + lambda_v * L_v
   + lambda_T * L_T
   + lambda_s * L_s
   + lambda_clean * L_clean`

Default coefficient family for Stage 0 and bounded validation:

- `lambda_v in {0.3, 1.0}` according to the bounded search;
- `lambda_T = 1.0`;
- `lambda_s = 0.05`;
- `lambda_clean = 0.2` unless clean-retention diagnostics require a
  preregistered bounded-search coefficient;
- `g_max = 0.10`;
- `K in {2,4}`.

No coefficient may be tuned on confirmatory test outcomes.

## Gradient Paths

Allowed gradient paths:

- `L_v` updates `F_theta` through `V_t^k`, `g_t^k`, and the differentiable
  dependence of later `C_t^k` on earlier CFR steps.
- `L_T` updates all CFR parameters through the full unroll ending at `C_t^K`.
- `L_s` updates CFR parameters through adjacent unrolled chunk states.
- `L_clean` updates CFR parameters through `C_t^K` and is intended to preserve
  Base behavior.

Forbidden gradient paths:

- no gradient into `A_t`;
- no gradient into `B_t`;
- no update to frozen Base parameters unless a LoRA/adapter parameter is
  explicitly listed as trainable;
- no gradient through confirmatory outcomes, reward, success, done, object pose,
  future observation, or reset identity;
- no optimizer step may update parameters outside the declared CFR/LoRA
  trainable set.

Stage 0 must report:

- finite objective values for every term;
- finite nonzero gradient norm for expected CFR parameters;
- zero frozen-parameter gradient count;
- CFR-to-Base or CFR-to-flow gradient norm ratio no greater than `100`;
- gradient conflict/cosine diagnostics if `L_clean` and `L_T` strongly oppose
  each other.

## Scale Audit

Before training, Stage 0 must estimate on a small batch:

- mean and p95 of `|A_t - B_t|`;
- mean and p95 of `|D_t^k|`;
- `L_v`, `L_T`, `L_s`, `L_clean`, and optional `L_flow`;
- per-term gradient norm on the trainable CFR parameter set;
- maximum, mean, and p95 normalized action deltas from Base;
- postprocessed environment-action finite fraction and validity fraction.

If one weighted loss term has gradient norm more than `100x` another required
term before any optimizer step, classify as
`CFR_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` unless the prototype
protocol already froze a bounded coefficient adjustment.

## Action-Validity Semantics

CFR may not use an ad hoc `[-1,1]` rule as the sole validity gate.

Before any Stage 0 execution, the runner must persist the official action
semantics used for all five policies:

- model-native action shape;
- postprocessor / unnormalizer class and parameters;
- environment action shape;
- environment `action_space.low` and `action_space.high` if the official
  environment exposes them;
- finite checks after postprocessing;
- whether the official environment accepts the postprocessed action via its
  action space or equivalent validation routine;
- gripper convention and any known official saturation/normalization rule.

A postprocessed action is valid only if it is finite, has official shape, and
passes the frozen official environment/action-space semantics. The same
definition applies to Base, DFM proxy, CFR, no-iterative ablation, and standard
LoRA.

If Base fails before CFR acts, Stage 0 stops as
`CFR_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`. This is not a scientific
kill and cannot be repaired by clipping or changing the validity definition
after seeing results.

## Distance Choice

Huber and vector-field consistency are used because the targets are
deterministic continuous action chunks with occasional outliers and maskable
future steps.

Rejected alternatives:

- KL: invalid for deterministic 7D action vectors or SmolVLA flow vectors
  without a justified probability distribution;
- JS: same distributional-support problem as KL;
- Wasserstein: unnecessary and underidentified for small masked chunks without
  distribution samples;
- MMD: measures sample-distribution mismatch but does not directly supervise
  per-state residual velocity;
- Mahalanobis: requires stable covariance estimates and can hide per-dimension
  failure under poorly conditioned action covariance;
- plain L2: too sensitive to outlier gripper/rotation spikes;
- trajectory discrepancy alone: useful as a diagnostic but does not specify a
  local vector-field update at each refinement step.

## Required Ablations And Baselines

First serious comparison:

1. `smolvla_base`;
2. `dfm_vla_continuous_refinement_proxy` or official `dfm_vla` if installed;
3. `cfr_full`;
4. `cfr_no_iterative_refinement`;
5. `standard_lora`.

Required ablation:

`cfr_no_iterative_refinement` removes repeated full-chunk refinement and uses a
single terminal residual under the same input, optimizer, cap, and retention
policy.

Strongest simple killer:

`standard_lora` uses the same demonstrations, optimizer budget, rank, target
modules, and clean-retention policy but no CFR refinement objective or unrolled
decoder.

## Stage 0 Stop Classes

Stage 0 may end only as one of:

- `CFR_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `CFR_STAGE_0_NO_USABLE_HEADROOM`;
- `CFR_STAGE_0_DESIGN_FAILURE`;
- `CFR_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- `CFR_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

No Stage 0 stop is a closed-loop scientific kill. Bounded validation and any
rollout are allowed only after `CFR_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

## Audit Decision

CFR passes to preregistration only under this mathematical audit. Any later
objective, distance, action-validity semantics, proxy definition, ablation list,
or coefficient budget change must be recorded before execution and may not use
confirmatory outcomes.
