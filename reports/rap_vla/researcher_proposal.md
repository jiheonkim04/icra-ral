# RAP-VLA Researcher A Proposal

Date: 2026-07-16 KST

Decision: `RAP_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `RAP-VLA`, Retrieval-Anchored Prior residualization for VLA action
flows.

Contribution type: `PRIOR_EXTENSION`.

This proposal starts Epoch 4 Cycle 25 after the fixed VDR Stage 0A
implementation/optimization failure. It does not repair VDR, change VDR
thresholds, reinterpret VDR action validity, clip VDR actions, reuse VDR
partial results as success evidence, or relabel VDR as a scientific kill.

## External Prior Anchor

Closest external prior: OptimusVLA, https://arxiv.org/abs/2602.20200.

Official code/assets: https://github.com/iLearn-Lab/CVPR26-OptimusVLA. The
repository reports released inference code, LIBERO assets, memory features,
memory actions, FAISS index, GPM checkpoint, LCM checkpoint, and evaluation
scripts.

Positive prior result: OptimusVLA reports `98.6%` average LIBERO success,
CALVIN/RoboTwin/real-world gains, and `2.9x` inference speedup by replacing a
Gaussian action-generation prior with Global Prior Memory and adding Local
Consistency Memory over executed action history.

Secondary priors:

- Past-Token Prediction for long-context diffusion policies,
  https://arxiv.org/abs/2505.09561
- AutoHorizon, https://arxiv.org/abs/2602.21445

## Claim

If local SmolVLA failures include weak task-phase action priors or action chunks
that drift away from legal expert modes, then anchoring the action flow around
retrieved legal demonstration chunks and learning only a bounded residual can
improve manipulation success while preserving the normal SmolVLA inference
interface.

The claim is intentionally narrow. RAP is not a generic retrieval policy, not
a nearest-demonstration replay method, not a Local Consistency Memory clone,
not a future-visual-feature method, not an action-to-end-effector realization
method, and not a LoRA contribution. LoRA is only low-compute infrastructure
for an identity-preserving residual/gate path.

## Evidence Partitions

`DISCOVERY`:

- build retrieval memory from training demonstrations only;
- inspect current-observation feature quality, top-k diversity, task/phase
  coverage, and retrieved-anchor action validity;
- compare retrieved anchors against task/phase mean chunks and nearest-neighbor
  direct replay baselines;
- debug serializer, memory, and residual-target construction.

`VALIDATION`:

- choose one residual/gate coefficient from the bounded search;
- verify retrieval headroom, residual predictability, clean retention,
  postprocessed action validity, Base passthrough, reload, and gradient flow;
- select one final configuration using the frozen validation score.

`CONFIRMATORY_TEST`:

- one frozen paired official LIBERO manifest after method, configuration,
  policies, ablation, task/reset identities, metrics, and thresholds are
  frozen;
- confirmatory outcomes cannot retune RAP, memory construction, retrieval
  features, k, coefficients, thresholds, or baselines.

## Scientific Method

For a legal timestep `t`, let:

- `o_t`: current legal RGB observations, proprioception, and language;
- `A_t in R^(50x7)`: normalized expert action chunk used for ordinary flow
  supervision;
- `f_t in R^d`: frozen deployment-observable retrieval feature from SmolVLA
  visual-policy features, proprioception, language/task identity, and phase;
- `M = {(f_i, A_i, task_i, phase_i)}`: discovery-only memory entries;
- `N_k(t)`: top-k legal memory entries under the frozen retrieval metric;
- `w_i(t)`: frozen normalized retrieval weights;
- `a_bar_t = sum_{i in N_k(t)} w_i(t) A_i`: retrieved action anchor chunk;
- `r_t = A_t - a_bar_t`: expert residual around the retrieved anchor;
- `g_theta(o_t) in [0, g_max]`: learned residual gate initialized to zero;
- `R_theta(o_t, a_bar_t) in R^(50x7)`: learned residual action-flow path.

RAP trains a bounded residualized action-flow objective:

`A_hat_t = BaseFlow_theta(o_t) + g_theta(o_t) * R_theta(o_t, a_bar_t)`

with residual target:

`R_theta(o_t, a_bar_t) -> r_t`.

The residual target is constructed only from legal demonstration actions and
deployment-observable retrieval features. Future observations, rewards,
success labels, reset identities, simulator state, object poses, and
confirmatory outcomes are forbidden in memory construction and target
construction.

The training objective is:

`L = L_flow + lambda_r * Huber(R_theta(o_t, a_bar_t), r_t)
     + lambda_clean * ||A_hat_t - BaseFlow(o_t)||_Huber_on_clean`

where Huber terms are coordinate-mean losses in normalized action units after
scale checks. The residual gate starts at zero, so the initialized policy is
exactly Base. At inference, RAP uses only the frozen training memory, current
legal observation, current proprioception, current instruction, and legal
executed action history if the selected OptimusVLA proxy also uses it. No
privileged input is required.

## Closest Prior And Controls

The transparent OptimusVLA proxy uses the same memory entries and retrieval
features to replace the action-generation prior with a retrieved anchor
distribution, plus one lightweight local-consistency smoothing term over
executed history when available. It is a transparent proxy, not an official
OptimusVLA reproduction unless the official released assets are installed and
verified locally.

The key ablation, `rap_anchor_only_no_residual`, uses retrieved anchors and the
same gate scaffold but removes residual learning. This tests whether direct
memory priors alone explain any gain.

Matched `standard_lora` receives the same demonstrations, optimizer budget,
rank, target modules, clean-retention policy, and ordinary flow objective, but
no retrieval anchor or residual target. This is the single strongest simple
reviewer-killer baseline because RAP updates policy weights through a
low-compute adapter.

## Low-Compute Parameterization

Use the repository's validated low-compute SmolVLA adapter path with
identity-preserving initialization. Rank-4 LoRA or an equivalent zero-effect
adapter may implement `R_theta` and `g_theta`, but the scientific contribution
is retrieval-anchored residualized action-flow learning.

Every trainable residual branch initializes to zero. Every learned gate
initializes to Base passthrough. Anchor influence is bounded by a frozen
maximum residual norm and by postprocessed 7D action-validity checks.

## Fixed Development Sources

Use the same fixed development task families as the recent SmolVLA development
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

No reward, success, done flag, reset identity, simulator object pose, or
confirmatory outcome may enter retrieval memory, residual targets, validation
selection, or training.

## Pre-Experiment Gates

Before training:

1. proposal and source hashes match;
2. memory, feature, action, proprioception, language/task, phase, and timestamp
   records are finite and aligned;
3. duplicate, missing, extra, and split-overlap keys are zero;
4. at least `512` discovery and `128` validation windows are available;
5. every task has validation rows and no task contributes more than `40%` of
   the audit subset;
6. top-k retrieval neighborhoods are noncollapsed: at least `3` unique memory
   demonstrations in the median top-8 neighborhood and no single source row
   accounts for more than `25%` of all top-1 retrievals;
7. retrieved anchors beat task/phase mean chunks by at least `10%` validation
   action MSE or `0.01` normalized Huber;
8. residual targets have positive variance in every action dimension after
   masking invalid padded steps;
9. a deployment-input residual probe beats a zero-residual predictor by at least
   `5%` relative validation Huber or `0.01` absolute normalized Huber;
10. anchor-only ablation is distinct from RAP's learned residual path;
11. initialized and disk-reloaded adapter reproduces Base flow and
    postprocessed actions within `1e-6`;
12. postprocessed action validity is preserved and Base-relative deltas are
    bounded before rollout;
13. expected RAP parameters receive finite nonzero gradients and frozen Base
    parameters do not update;
14. exceptions are zero.

Failure classes:

- source, overlap, collapsed retrieval, or collapsed residual target:
  `RAP_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- retrieved anchors do not beat trivial task/phase means or Base has no
  relevant failure: `RAP_STAGE_0_NO_USABLE_HEADROOM`;
- residuals are not predictable from deployment inputs or anchor-only is
  equivalent to RAP: `RAP_STAGE_0_DESIGN_FAILURE`;
- hash, serialization, identity, persistence, gradient, or action-validity
  defect: `RAP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.

None of these Stage 0 stops is a closed-loop scientific kill.

## Bounded Validation Search

At most six trained development configurations:

1. RAP `lambda_r=0.1`, `g_max=0.25`;
2. RAP `lambda_r=0.3`, `g_max=0.25`;
3. RAP `lambda_r=1.0`, `g_max=0.25`;
4. transparent OptimusVLA memory-prior proxy;
5. `rap_anchor_only_no_residual`;
6. matched standard LoRA.

The retrieval feature definition, distance metric, top-k value, task sources,
memory split, residual target, adapter rank, optimizer, steps, and checkpoint
selection rule are fixed before validation search. One seed per configuration
unless a fixed run is genuinely unresolved; no more than two seeds may then be
used before final selection.

Validation score for selecting the RAP coefficient:

`S = 0.30 * validation_success_or_proxy
   + 0.25 * RAP_minus_anchor_only_margin
   + 0.20 * clean_action_retention
   + 0.15 * postprocessed_action_validity
   + 0.10 * memory_overhead_score`.

If closed-loop validation is not feasible, `validation_success_or_proxy` must be
replaced before execution by one frozen deployment-observable proxy. Offline
action L2 alone may not select the configuration. Tie break: smaller
`lambda_r`, then lower retrieval overhead.

## First Serious Comparison

Exactly five policy identities:

1. `smolvla_base`;
2. `optimusvla_memory_prior_proxy`;
3. `rap_full`;
4. `rap_anchor_only_no_residual`;
5. `standard_lora`.

| Comparison | Scientific question |
| --- | --- |
| Base vs RAP | Does retrieval-anchored residualized flow improve SmolVLA? |
| OptimusVLA proxy vs RAP | Does residualizing around anchors beat a transparent memory-prior sampler? |
| Anchor-only ablation vs RAP | Is learned current-state residualization necessary beyond direct retrieved action priors? |
| Standard LoRA vs RAP | Is any gain explained by ordinary data-matched adaptation? |

## Paper-Candidate Gate

RAP becomes a serious paper candidate only if frozen SmolVLA comparisons show
that RAP beats Base, the OptimusVLA proxy, anchor-only/no-residual ablation, and
standard LoRA while retaining clean behavior, preserving postprocessed action
validity, and showing that retrieved anchors plus learned residuals are both
active.

Then verify the unchanged scientific method on Quantized OpenVLA-OFT INT4 and
add one claim-specific second condition or benchmark.

## Non-Claims

- RAP is not official OptimusVLA unless official assets are installed and
  verified.
- RAP is not nearest-demonstration action replay.
- RAP is not Local Consistency Memory as a standalone contribution.
- RAP is not KITE, VDR, RAR, HEST, HASTE, COVI, LIFT, or EAC rescue.
- RAP is not a LoRA, QLoRA, PEFT, or adaptation-efficiency method.
