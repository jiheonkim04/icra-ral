# URF-VLA Reviewer B Attack

Date: 2026-07-16 KST

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Proposal under review: `reports/urf_vla/researcher_proposal.md`

Proposal SHA-256:
`E78829E736C3F22451E72574092221904ACBE4C4BE0BDA7FA046832DABED3532`

## Independent Closest Primary Sources

### 1. SUREFlow

Primary source: `https://arxiv.org/abs/2607.10504`

Official repository: `https://github.com/tanvirnwu/SUREFlow`

AUTHOR_STATED:

- SUREFlow integrates uncertainty-aware residual flow matching in a Mamba-based
  state-space VLA.
- It jointly predicts action velocities and input-dependent residual
  uncertainty.
- It reports `92.5%` average LIBERO success and LIBERO-PRO robustness with
  `179.1M` parameters.

INDEPENDENTLY_INFERRED:

- This is the closest prior. It already owns the phrase-level neighborhood
  "uncertainty-aware residual flow matching."
- URF cannot claim novelty from heteroscedastic residual flow alone.
- URF's only defensible difference is a conservative overlay around an already
  pretrained SmolVLA Base chunk, with Base passthrough and cell-level bounded
  routing.

CROSS_PAPER_SYNTHESIZED:

- If URF's uncertainty head is trained only as another heteroscedastic residual
  regressor and then globally changes actions, the method collapses into a
  small local SUREFlow proxy plus LoRA.

### 2. Guided Action Flow

Primary source: `https://arxiv.org/abs/2607.02092`

AUTHOR_STATED:

- Guided Action Flow keeps a pretrained SmolVLA policy frozen and uses a learned
  action-chunk critic to guide the reverse-time flow sampler.
- It clips and gates critic gradients during sampling, and includes
  uncertainty-aware critic-ensemble disagreement gating.
- It reports positive but modest held-out gains, making critic generalization a
  central bottleneck.

INDEPENDENTLY_INFERRED:

- This is the closest frozen-SmolVLA action-intervention prior. It changes the
  action generation trajectory without fine-tuning Base.
- URF must not describe itself as the first Base-preserving SmolVLA action
  correction method.
- URF differs only if it uses demonstration-only residual supervision and
  heteroscedastic residual routing, rather than rollout-success critics and
  test-time gradient guidance.

CROSS_PAPER_SYNTHESIZED:

- A reviewer may ask whether URF is just Guided Action Flow with an offline
  residual regressor replacing the critic. Stage 0 and the first comparison
  must separate uncertainty-routed residual transport from generic
  gradient/critic or residual guidance.

### 3. Flow-Based VLA Uncertainty / SAVE

Primary source: `https://arxiv.org/abs/2606.18043`

AUTHOR_STATED:

- This work estimates epistemic uncertainty in flow-based VLAs through
  velocity-field disagreement across a small ensemble.
- It uses this uncertainty for failure detection and active fine-tuning,
  reducing expert demonstration needs by at least `22%` in reported LIBERO
  experiments.

INDEPENDENTLY_INFERRED:

- It is a stronger uncertainty-estimation prior than URF's single-head
  heteroscedastic variance if the claim is "reliable uncertainty."
- URF must frame uncertainty as an internal routing variable for residual
  transport, not as a state-of-the-art epistemic uncertainty estimator.

CROSS_PAPER_SYNTHESIZED:

- The simplest reviewer-killer for URF's uncertainty component is not only
  no-uncertainty residual. It is also a residual route gated by ensemble or
  perturbation disagreement. If too expensive for the first five-policy
  prototype, it must appear as a Stage 0 diagnostic or later paper control if
  URF reaches GO.

### 4. Perturbation-Based Uncertainty For VLA Failure Detection

Primary source: `https://arxiv.org/abs/2606.20754`

AUTHOR_STATED:

- This paper injects Gaussian perturbations into hidden activations and uses
  action-disagreement to estimate epistemic uncertainty without labels.
- It reports stronger failure detection under LIBERO and LIBERO-PRO shifts than
  sampling-based uncertainty.

INDEPENDENTLY_INFERRED:

- It is not an action-improvement method, but it is a credible alternative
  uncertainty signal.
- If URF's uncertainty route does not outperform a simple perturbation or
  stochastic-disagreement route on validation calibration, URF's learned
  variance head is unjustified.

## Novelty Attack

URF is dangerously close to SUREFlow. The proposal uses the same core words:
uncertainty, residual, flow, and action generation. The novelty cannot be:

- heteroscedastic residual regression;
- predicting residual variance;
- using uncertainty to gate corrections;
- training with a lightweight adapter;
- or saying "SmolVLA" instead of "Mamba."

The only acceptable novelty boundary is:

`Base-preserving uncertainty-routed bounded residual transport around an
already trained SmolVLA action chunk, initialized to exact Base passthrough and
trained from existing demonstrations without rollout-success labels.`

If mathematical audit cannot preserve that boundary, URF should be rejected
before implementation.

## Prior Comparison Attack

The SUREFlow proxy must not be weak by construction. A fair proxy must:

- use the same discovery/validation/test partitions as URF;
- use only legal deployment-observable inputs;
- receive the same cached SmolVLA features, Base chunks, expert residuals, and
  optimizer budget where technically valid;
- train a heteroscedastic residual-flow objective, not a plain MSE residual;
- expose a comparable uncertainty or variance output;
- use the same postprocessor and action-validity definition;
- be reported as a transparent local proxy, not official SUREFlow, unless
  official SUREFlow assets are installed and verified.

If official SUREFlow is locally installed, URF must compare against it or
record why exact comparison is incompatible with the current SmolVLA-LIBERO
protocol.

## Simple Equivalence Attack

URF may be equivalent to one of these simpler methods:

1. ordinary residual LoRA with a smaller residual cap;
2. homoscedastic residual regression plus thresholding by residual magnitude;
3. task/phase mean residual plus route sparsity;
4. stochastic-sampling disagreement gate plus ordinary residual;
5. perturbation-disagreement gate plus ordinary residual;
6. clean-retention regularization suppressing residuals until the method is
   nearly Base.

The current first comparison includes standard LoRA and the no-uncertainty
route ablation. That is acceptable for the first prototype only if Stage 0 also
records cheap diagnostics for residual-magnitude, task/phase residual, and
sampling/perturbation disagreement explanations.

## Data And Supervision Attack

URF assumes residual headroom exists in `A_expert - A_base`. Recent local
cycles repeatedly found weak or unusable residual headroom:

- CFR found no usable full-chunk refinement headroom.
- TSC found masked action-cell structure insufficient.
- CCIF found coarse intent was not predictable enough and endpoint-only
  diagnostics explained signal.

Therefore Stage 0 must not proceed directly to training. It must prove:

- residual target variance is noncollapsed by task, phase, time, and action
  dimension;
- Base residuals are not mostly postprocessor noise;
- a heteroscedastic probe beats homoscedastic residual and task/phase residual
  baselines on validation;
- predicted uncertainty strata are monotonic with residual error on validation;
- URF full differs from both Base and no-uncertainty ablation after a small fit;
- route sparsity is neither all-zero nor all-one;
- gripper residuals do not dominate because of scale.

If these checks fail, classify correctly as
`URF_STAGE_0_DATA_OR_SUPERVISION_FAILURE`,
`URF_STAGE_0_NO_USABLE_HEADROOM`, or `URF_STAGE_0_DESIGN_FAILURE`, not as a
closed-loop scientific result.

## Mathematical Attack

The proposed heteroscedastic Huber objective is plausible, but the audit must
define:

- whether `ell_theta` parameterizes log variance, log scale, or log precision;
- how the Huber scale interacts with the variance term;
- whether `ell_theta` gradients can inflate uncertainty to avoid residual
  learning;
- whether a variance floor/ceiling is used;
- how uncertainty maps to the bounded route gate;
- tensor shapes for `[batch, horizon=50, action_dim=7]`;
- units and separate scaling for translation, rotation, and gripper;
- loss magnitudes and gradient norms before training;
- why no KL is used between deterministic action vectors.

A critical risk is decorative uncertainty: variance may improve NLL while the
route gate itself does not produce better or safer actions. The key ablation
must remove uncertainty routing while keeping residual capacity matched.

## Inference And Leakage Attack

URF may use:

- current images;
- current proprioception/state exposed to SmolVLA;
- task text or task identity already available to the policy;
- frozen Base decoded chunk;
- stored training statistics or checkpoints.

URF may not use:

- simulator reward/success/done;
- object pose;
- future observation;
- expert future action at inference;
- confirmatory reset identity;
- test-set residual statistics;
- failed rollout labels collected from confirmatory evaluation.

The proposal must explicitly freeze that SUREFlow proxy, standard LoRA, and
URF all use the same split and do not tune thresholds on confirmatory outcomes.

## Reviewer Conditions For Rebuttal

Researcher A must accept or answer these conditions before mathematical audit:

1. URF novelty is narrowed to Base-preserving uncertainty-routed bounded
   residual transport around a frozen SmolVLA chunk.
2. SUREFlow remains the closest prior and policy 2 unless official SUREFlow
   assets are locally installed and verified.
3. The SUREFlow proxy must be heteroscedastic and not a strawman.
4. Guided Action Flow must be listed as the closest frozen-SmolVLA
   action-intervention prior, even if not a first-stage policy.
5. Flow-Based VLA UQ / SAVE and perturbation-based uncertainty must be treated
   as uncertainty-signal alternatives.
6. `urf_no_uncertainty_route_ablation` remains the key ablation.
7. `standard_lora` remains the first simple reviewer-killer because URF trains
   on the same demonstrations.
8. Stage 0 must include task/phase residual, residual-magnitude, stochastic
   disagreement, or perturbation disagreement diagnostics where cheap.
9. Stage 0 must prove uncertainty strata are noncollapsed and monotonic with
   residual error on validation.
10. The route gate must not be globally active; all-zero and all-one routing
    are design failures.
11. Mathematical audit must define log-variance semantics, scale, gradient
    paths, floors/ceilings, gate mapping, and group-wise action units.
12. No deterministic-action KL is allowed.
13. No privileged inference inputs or confirmatory-test tuning are allowed.
14. CCIF, TSC, CFR, AMP, RAP, and VDR may not be rescued or reinterpreted.

## Reviewer Decision

Conditional pass to Researcher A rebuttal.

URF is not killed before implementation because a narrow, defensible mechanism
remains: demonstration-supervised heteroscedastic residual transport that keeps
Base as the default and uses uncertainty to decide which Base action cells may
move. However, it is close enough to SUREFlow and Guided Action Flow that the
claim must be narrowed and Stage 0 must aggressively test whether uncertainty
routing does anything beyond ordinary residual adaptation, task/phase residuals,
and cheap disagreement gates.

Next stage: Researcher A rebuttal must accept or answer all Reviewer B
conditions before mathematical audit, preregistration, implementation, or
training.
