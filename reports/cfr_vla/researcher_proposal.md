# CFR-VLA Researcher A Proposal

Date: 2026-07-16 KST

Decision: `CFR_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `CFR-VLA`, Continuous Full-Chunk Refinement for VLA action-flow
decoding.

Contribution type: `PRIOR_EXTENSION`.

This proposal starts Epoch 4 Cycle 27 after the fixed AMP Stage 0
implementation/optimization failure. It does not repair AMP, change AMP
thresholds, clip AMP actions, reinterpret AMP action validity, reuse AMP
partial results as success evidence, or relabel AMP as a scientific kill.

## External Prior Anchor

Closest external prior: DFM-VLA, https://arxiv.org/html/2603.26320v1.

Project page: https://chris1220313648.github.io/DFM-VLA/.

Official code/assets: the DFM-VLA project page reports code as coming soon
during this pass. Until official code is locally installed and verified, policy
2 is a transparent local proxy named `dfm_vla_continuous_refinement_proxy`, not
an official DFM-VLA reproduction.

Positive prior result: DFM-VLA reports iterative action refinement with
discrete flow matching over full action-token sequences, `95.7%` LIBERO
average success, `4.44` CALVIN average success length, and `70.8%` real-world
average success on the project page.

Secondary priors:

- Adaptive Action Chunking, https://arxiv.org/html/2604.04161v2
- RotVLA, https://arxiv.org/abs/2605.13403
- GEAR-VLA, https://arxiv.org/abs/2606.08530

## Claim

If local SmolVLA failures include decoded action chunks whose later elements
remain inconsistent with the intended manipulation after one flow decode, then
a bounded iterative refinement field over the entire continuous action chunk
can improve closed-loop manipulation while preserving the normal SmolVLA
interface and clean behavior.

The claim is intentionally narrow. CFR is not generic LoRA fine-tuning, not
adaptive chunk-size selection, not action-manifold projection, not retrieval
memory, not nearest-neighbor action replay, and not an official DFM-VLA
reproduction unless official assets are installed and verified locally. LoRA is
only implementation infrastructure for an identity-preserving refinement field.

## Evidence Partitions

`DISCOVERY`:

- inspect Base decoded chunks, expert chunks, residual chunk structure, phase
  coverage, and action-validity ranges;
- fit transparent DFM-style proxy components from training demonstrations only;
- debug continuous refinement targets, unroll stability, action bounds,
  serializer, identity, reload, and gradient paths.

`VALIDATION`:

- select one CFR configuration from the bounded validation search;
- compare CFR against the DFM proxy, no-iterative ablation, and standard LoRA
  using only validation data or a frozen validation rollout/proxy;
- verify clean retention, action validity, bounded deltas, reload, and
  mechanism activation;
- select one final configuration using the frozen validation score.

`CONFIRMATORY_TEST`:

- one frozen paired official LIBERO manifest after method, configuration,
  policies, ablation, task/reset identities, metrics, and thresholds are
  frozen;
- confirmatory outcomes cannot retune CFR, iteration count, residual cap,
  loss weights, targets, projection, tasks, reset identities, or baselines.

## Scientific Method

For a legal demonstration timestep `t`, let:

- `o_t`: current legal RGB observations, proprioception, and language;
- `A_t in R^(50x7)`: normalized expert action chunk;
- `B_t in R^(50x7)`: frozen Base SmolVLA action chunk decoded from `o_t`;
- `x_t`: deployment-observable feature built from frozen SmolVLA visual tokens,
  proprioception, task/language identity, phase, and the Base chunk;
- `C_t^0 = B_t`: the initial continuous chunk before refinement;
- `K`: fixed refinement step count selected before confirmatory testing;
- `V_theta(o_t, C_t^k, k) in R^(50x7)`: bounded residual velocity field;
- `g_theta(o_t, C_t^k, k) in [0, g_max]`: zero-initialized refinement gate;
- `C_t^k`: refined chunk after step `k`.

CFR refines the complete action chunk before execution:

`C_t^(k+1) = C_t^k + g_theta(o_t, C_t^k, k) * V_theta(o_t, C_t^k, k)`

for `k = 0, ..., K-1`, then returns:

`A_hat_t = Bound(C_t^K)`

where `Bound` is the frozen postprocessed action-validity check and admissible
range guard. It is not a learned clipping rescue: if valid Base-compatible
action semantics are not preserved, CFR stops before rollout.

The residual velocity target at each refinement step is:

`D_t^k = stopgrad(A_t - C_t^k) / (K - k)`

with all terms in normalized action units. The training objective is:

`L = L_flow
   + lambda_v * mean_k Huber(V_theta(o_t, C_t^k, k), D_t^k)
   + lambda_T * Huber(C_t^K, A_t)
   + lambda_s * mean_k ||C_t^(k+1) - C_t^k||_Huber
   + lambda_clean * Huber(C_t^K, B_t)_clean`

All Huber terms are coordinate means in normalized action units after a
small-batch magnitude and gradient-norm audit. No KL divergence is used:
deterministic 7D actions and SmolVLA flow vectors are not probability
distributions.

At inference, CFR uses only current legal observations, current proprioception,
language/task input, frozen Base features/actions, and learned demonstration-
derived parameters. Rewards, success flags, done flags, object poses, future
observations, reset identities, and confirmatory outcomes are forbidden.

## Closest Prior And Controls

The transparent DFM proxy, `dfm_vla_continuous_refinement_proxy`, is a local
development-only proxy for the prior's iterative full-sequence refinement
principle. It quantizes normalized demonstration action chunks into fixed
discovery-derived bins, fits an iterative token-refinement or velocity proxy
from legal training demonstrations, dequantizes to `[50,7]`, and uses the same
current legal deployment inputs. It is clearly labeled as a proxy until
official DFM-VLA code/assets are installed.

The key ablation, `cfr_no_iterative_refinement`, uses the same inputs,
adapter/scaffold, residual cap, optimizer, and clean-retention policy, but
collapses the update into one terminal residual prediction with no repeated
chunk refinement. This tests whether iterative full-chunk refinement is
necessary.

Matched `standard_lora` receives the same demonstrations, optimizer budget,
rank, target modules, clean-retention policy, and ordinary flow objective, but
no iterative refinement objective or unrolled refinement decoder. This is the
single strongest simple reviewer-killer baseline because CFR updates policy
actions through low-compute adapter infrastructure.

## Low-Compute Parameterization

Use the repository's validated low-compute SmolVLA adapter path with
identity-preserving initialization. Rank-4 LoRA or an equivalent zero-effect
adapter may parameterize `V_theta` and `g_theta`, but the scientific
contribution is continuous full-chunk iterative refinement.

Every residual branch initializes to zero. Every gate initializes to Base
passthrough. The initialized and disk-reloaded policy must reproduce Base flow
and postprocessed actions within `1e-6`. Refinement magnitude is bounded by a
frozen residual cap, a frozen `g_max`, and postprocessed 7D action-validity
checks.

## Fixed Development Sources

Use the same fixed development task families as recent SmolVLA development
cycles for continuity:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

Within each source:

- discovery/training demonstrations: `0..7`;
- validation demonstrations: `8..9`;
- confirmatory task/reset identities: untouched until one configuration and all
  policies are frozen.

No reward, success, done flag, reset identity, simulator object pose, future
observation, or confirmatory outcome may enter target construction, validation
selection, or training.

## Pre-Experiment Gates

Before training or rollout:

1. proposal and source hashes match;
2. action, Base chunk, proprioception, image-feature, language/task, phase, and
   timestamp records are finite and aligned;
3. duplicate, missing, extra, and split-overlap keys are zero;
4. at least `512` discovery and `128` validation windows are available;
5. every task has validation rows and no task contributes more than `40%` of
   the audit subset;
6. Base-to-expert residual chunks are noncollapsed and have positive variance
   in every action dimension after valid-step masking;
7. a deployment-input residual/refinement probe beats a task/phase residual
   predictor by at least `5%` relative validation Huber or `0.005` absolute
   normalized Huber;
8. the DFM proxy leaves residual headroom for CFR of at least `5%` relative
   Huber or `0.005` absolute normalized Huber;
9. iterative refinement improves the frozen validation proxy over
   `cfr_no_iterative_refinement` before any rollout;
10. initialized and disk-reloaded adapter reproduces Base flow and
    postprocessed actions within `1e-6`;
11. CFR differs from Base, DFM proxy, and no-iterative ablation after a small
    training smoke, but the difference is bounded rather than global;
12. postprocessed action validity is preserved before rollout;
13. expected CFR parameters receive finite nonzero gradients and frozen Base
    parameters do not update;
14. exceptions are zero.

Failure classes:

- source, overlap, collapsed residuals, or collapsed proxy labels:
  `CFR_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- Base or DFM proxy leaves no usable residual headroom:
  `CFR_STAGE_0_NO_USABLE_HEADROOM`;
- residual/refinement targets are not predictable from deployment inputs or the
  iterative mechanism is equivalent to the no-iterative ablation:
  `CFR_STAGE_0_DESIGN_FAILURE`;
- hash, serialization, identity, persistence, gradient, bound, or
  action-validity defect:
  `CFR_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- all gates pass:
  `CFR_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

None of these Stage 0 stops is a closed-loop scientific kill.

## Bounded Validation Search

At most six trained development configurations:

1. CFR `K=2`, `lambda_v=0.3`, `g_max=0.10`;
2. CFR `K=4`, `lambda_v=0.3`, `g_max=0.10`;
3. CFR `K=4`, `lambda_v=1.0`, `g_max=0.10`;
4. `dfm_vla_continuous_refinement_proxy`;
5. `cfr_no_iterative_refinement`;
6. matched `standard_lora`.

The feature definition, task sources, split, residual target, adapter rank,
optimizer, step budget, residual cap, action-validity check, and checkpoint
selection rule are fixed before validation search. One seed per configuration
unless a fixed run is genuinely unresolved; no more than two seeds may then be
used before final selection.

Validation score for selecting the CFR configuration:

`S = 0.30 * validation_success_or_proxy
   + 0.25 * CFR_minus_DFM_proxy_margin
   + 0.20 * clean_action_retention
   + 0.15 * postprocessed_action_validity
   + 0.10 * refinement_overhead_score`.

If closed-loop validation is not feasible, `validation_success_or_proxy` must be
replaced before execution by one frozen deployment-observable proxy. Offline
action L2 alone may not select the configuration. Tie break: fewer refinement
steps, then lower residual cap.

## First Serious Comparison

Exactly five policy identities:

1. `smolvla_base`;
2. `dfm_vla_continuous_refinement_proxy` or official `dfm_vla` if installed;
3. `cfr_full`;
4. `cfr_no_iterative_refinement`;
5. `standard_lora`.

| Comparison | Scientific question |
| --- | --- |
| Base vs CFR | Does continuous full-chunk refinement improve SmolVLA? |
| DFM proxy vs CFR | Does continuous state-conditioned refinement beat the closest prior proxy? |
| No-iterative ablation vs CFR | Is repeated full-chunk refinement necessary? |
| Standard LoRA vs CFR | Is any gain explained by ordinary data-matched adaptation? |

## Paper-Candidate Gate

CFR becomes a serious paper candidate only if frozen SmolVLA comparisons show
that CFR beats Base, the DFM proxy or official DFM-VLA if installed, the
no-iterative ablation, and standard LoRA while retaining clean behavior,
preserving postprocessed action validity, and showing that refinement activates
in relevant states rather than everywhere.

Then verify the unchanged scientific method on Quantized OpenVLA-OFT INT4 and
add one claim-specific second condition or benchmark.

## Non-Claims

- CFR is not official DFM-VLA unless official code/assets are installed and
  verified.
- CFR is not adaptive chunk-size selection.
- CFR is not action-manifold projection, AMP, RAP, VDR, KITE, HEST, HASTE,
  IARC, FAMR, PCAV, SPARC, NICE, COVI, LIFT, or EAC rescue.
- CFR is not a generic LoRA, QLoRA, PEFT, or adaptation-efficiency method.
- CFR is not a KL or probabilistic action-distribution method.
