# VDR-VLA Researcher A Proposal

Date: 2026-07-16 KST

Decision: `VDR_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `VDR-VLA`, Visuomotor Dynamic Residual alignment for VLA policies.

Contribution type: `PRIOR_EXTENSION`.

This proposal starts Epoch 4 Cycle 24 after the fixed KITE Stage 0A
implementation failure. It does not repair KITE, change KITE thresholds, clip
KITE actions, reuse KITE partial results as success evidence, or relabel the
KITE outcome as a scientific kill.

## External Prior Anchor

Closest external prior: FutureVLA, https://arxiv.org/abs/2603.10712.

Positive prior result: FutureVLA reports that joint visuomotor predictive
modeling and latent embedding alignment improve downstream VLA frameworks by
decoupling visual state preservation from temporal action modeling.

Official code/checkpoint status: no official FutureVLA code or checkpoint was
verified locally. The local closest-prior comparison is therefore a faithful
transparent FutureVLA-style latent-alignment proxy, not an official
reproduction.

Secondary priors:

- ManiFlow, https://arxiv.org/abs/2509.01819
- FreqPolicy, https://arxiv.org/abs/2506.08822
- ALAM, https://arxiv.org/abs/2605.10819

## Claim

If local SmolVLA failures include weak alignment between generated action
chunks and the motor-induced component of near-future visual change, then a
training-only dynamic visual-feature residual objective can improve
closed-loop manipulation success while preserving the standard SmolVLA
inference path.

The claim is intentionally narrow. VDR is not a new full predictive
architecture, not an occlusion/complementary-view method, not an
end-effector-realization method, and not a latent-action or action-history
method. It tests whether subtracting an actionless static future-feature
predictor leaves a useful dynamic residual that generated actions can explain.

## Evidence Partitions

`DISCOVERY`:

- inspect local current/future visual-feature changes;
- fit frozen actionless static future-feature predictors;
- construct dynamic residual targets;
- audit target variance, task coverage, phase coverage, and trivial
  actionless baselines.

`VALIDATION`:

- select one coefficient from the preregistered VDR coefficient set;
- verify clean retention, action validity, residual predictability,
  full-versus-ablation distinction, finite gradients, and disk reload;
- select one final checkpoint using only the frozen validation score.

`CONFIRMATORY_TEST`:

- one frozen paired official LIBERO manifest after method, configuration,
  policies, ablation, task/reset identities, metrics, and thresholds are
  frozen;
- confirmatory outcomes cannot retune VDR.

## Scientific Method

For a demonstration timestep `t`, let:

- `o_t`: legal current observation with RGB streams, proprioception, and
  language/task instruction;
- `A_t in R^(50x7)`: normalized expert action chunk used for ordinary flow
  supervision;
- `E(o_t)`: frozen visual encoder feature pooled to `e_t in R^960`;
- `H in {4,12}`: short and medium future horizons;
- `e_(t+H)`: future frozen visual feature, used only for training targets;
- `p_t`: legal proprioceptive vector;
- `z_t`: legal task/language/phase feature used only as development input;
- `Ahat_t`: reconstructed clean action chunk from the SmolVLA flow path.

Using discovery rows only, fit a frozen actionless ridge predictor

`B_H(e_t,p_t,z_t) -> P_K(e_(t+H)-e_t)`

where `P_K` is a discovery-fitted PCA/whitening projection to `K=32`
coordinates. The predictor is forbidden to see generated actions or future
features at inference; it exists only to define the training target and
closest-prior ablation.

Define the dynamic residual target:

`r_(t,H) = P_K(e_(t+H)-e_t) - B_H(e_t,p_t,z_t)`.

VDR predicts the residual from the current policy representation and generated
clean action summary:

`rhat_(t,H) = D_theta(h_t, summarize(Ahat_t, H), e_t, p_t, z_t)`.

The primary VDR loss is coordinate-mean Huber with delta `1.0` in whitened
dynamic-residual units:

`L_vdr = mean_H Huber(rhat_(t,H), r_(t,H))`.

Training objective:

`L = L_flow + lambda_v L_vdr`.

No KL divergence, reward model, success label, event label, candidate reranker,
action clipping, future action latent, or inference-time correction is used.

## Low-Compute Parameterization

Use the repository's validated low-compute SmolVLA adapter path with
identity-preserving initialization. Rank-4 LoRA or an equivalent zero-effect
adapter is implementation infrastructure, not the scientific contribution.

Every trainable residual branch initializes to Base passthrough. At inference,
VDR removes all future-feature target construction, static predictors, residual
heads, and losses. The processor, solver, action horizon, and output action
path are exactly Base.

## Fixed Development Sources

Use the same fixed development task families as the immediately preceding
cycle for continuity, but with a different target and mechanism:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

Within each source:

- discovery/training demonstrations: `0..7`;
- validation demonstrations: `8..9`;
- confirmatory task/reset identities: untouched until one configuration and
  all policies are frozen.

No reward, success, done flag, reset identity, simulator object pose, or
confirmatory outcome may enter discovery, target construction, validation
selection, or policy training.

## Pre-Experiment Gates

Before training:

1. proposal and source hashes match;
2. visual features, actions, proprioception, and timestamps are finite and
   aligned;
3. duplicate, missing, extra, and split-overlap keys are zero;
4. at least `512` discovery and `128` validation windows are available for
   each horizon;
5. the PCA/whitened target has positive variance in all retained coordinates;
6. every task has validation rows and no task contributes more than `40%` of
   the sampled audit subset;
7. the actionless static predictor beats the discovery-mean future-feature
   predictor by at least `25%` validation MSE, proving static visual structure
   is measurable;
8. an action-conditioned residual probe beats the actionless residual probe by
   at least `5%` relative validation MSE or `0.02` normalized Huber, proving
   generated-action information is useful;
9. the FutureVLA-style full future-latent proxy leaves residual validation
   error large enough for VDR to improve by at least the same margin;
10. initialized and disk-reloaded adapter reproduces Base flow and decoded
    actions within `1e-6`;
11. action validity is `1.0` before rollout;
12. expected VDR parameters receive finite nonzero gradients and frozen Base
    parameters do not update;
13. exceptions are zero.

Failure classes:

- source, alignment, variance, count, or collapsed target:
  `DATA_OR_SUPERVISION_FAILURE`;
- no Base/prior residual headroom: `NO_USABLE_HEADROOM`;
- dynamic residual not predictable from generated-action information:
  `DESIGN_FAILURE`;
- hash, serialization, processor, identity, persistence, gradient, or action
  validity defect: `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.

None of these Stage 0 stops is a closed-loop scientific kill.

## Bounded Validation Search

At most six trained development configurations:

1. VDR `lambda_v=0.1`;
2. VDR `lambda_v=0.3`;
3. VDR `lambda_v=1.0`;
4. FutureVLA-style full future-latent proxy;
5. no-action-residual ablation;
6. matched standard LoRA.

Residual dimension `K=32`, horizons `{4,12}`, Huber delta `1.0`, optimizer,
steps, target modules, task sources, and checkpoint-selection rule are fixed.
One seed per configuration unless a fixed run is genuinely unresolved; no more
than two seeds may then be used before final selection.

Validation score for selecting the VDR coefficient:

`S = 0.35 * dynamic_residual_improvement + 0.25 * full_vs_ablation_margin +
0.20 * clean_action_retention + 0.15 * action_validity + 0.05 * efficiency`.

No configuration with nonfinite actions, disk-reload failure, Base-hash change,
clean retention failure, or action validity failure is eligible. Tie break:
smaller `lambda_v`.

## First Serious Comparison

Exactly five policy identities:

1. `smolvla_base`;
2. `futurevla_latent_alignment_proxy`;
3. `vdr_full`;
4. `vdr_no_action_residual`;
5. `standard_lora`.

The FutureVLA proxy aligns to the full projected future visual-feature change
without subtracting the actionless static predictor and without requiring the
generated action chunk to explain the residual. It is a transparent proxy, not
an official FutureVLA reproduction.

The no-action-residual ablation receives the same target but blocks generated
action information. Standard LoRA receives the same demonstrations, optimizer,
steps, rank, target modules, and flow objective but no dynamic-residual target.

## Paper-Candidate Gate

VDR becomes a serious paper candidate only if frozen SmolVLA comparisons show
that VDR beats Base, the FutureVLA proxy, the no-action-residual ablation, and
standard LoRA on the matched claim axis while retaining clean behavior,
preserving action validity, and showing that generated-action-conditioned
dynamic residuals are lower than the ablation. Then verify the unchanged
scientific method on Quantized OpenVLA-OFT INT4 and add one claim-specific
second condition.
