# TSC-VLA Researcher A Rebuttal

Date: 2026-07-16 KST

Method: `TSC-VLA`

Proposal hash:
`0DF143D2D8773D7ABF4FC76AB7CC083FE7EE65DF84EA06631E67C2445F6DC941`

Reviewer decision answered:
`REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Rebuttal decision: `TSC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

## Accepted Novelty Boundary

Researcher A accepts Reviewer B's narrowed novelty boundary.

TSC-VLA does not claim novelty for temporal-spatial masking broadly, action
masking broadly, or masked action generation broadly. TS-Mask VLA owns the
direct prior territory for native discrete action-token masking with a
diffusion action expert.

TSC's only claimed mechanism is:

> Base-clamped continuous temporal-spatial action-cell completion around a
> frozen SmolVLA `[50,7]` chunk, targeted by a deployment-observable
> Base-error mask.

If implementation collapses into ordinary TS-Mask proxy, confidence-only
failure detection, or sparse residual gating without a meaningful structured
mask effect, the method must stop before rollout.

## Accepted Closest-Prior Comparison

Policy 2 remains:

`ts_mask_continuous_proxy` or official `ts_mask_vla` if official code and
checkpoints are locally integrated and verified.

The local proxy must be faithful, not a strawman. It should share the same data
partition, observation inputs, action validity checks, and comparable compute.
The differentiating TSC element is the Base-error-targeted sparse mask and
Base-clamped completion rule.

## Accepted Ablation And Baseline

The key ablation remains:

`tsc_no_targeted_mask_ablation`

It must preserve comparable completion capacity while removing the targeted
Base-error mask.

Matched `standard_lora` remains required because TSC uses trainable lightweight
policy infrastructure. If a later implementation becomes strictly inference
only, Reviewer B may reevaluate this condition before confirmatory testing, not
after results.

Stage 0 must also include cheap diagnostics against simple residual
explanations:

- global residual gate;
- per-timestep or per-dimension gate when feasible;
- magnitude-only mask;
- non-trainable smoothing or simple residual baseline when cheaper than rollout.

## Accepted Label And Threshold Discipline

Error-label construction must be frozen before decisive Stage 0 evaluation.
Allowed construction can use discovery-only robust statistics or a
predeclared validation-only selection rule, but confirmatory outcomes may not
tune:

- positive-label threshold;
- top-k mask fraction;
- per-dimension normalization;
- mask activation threshold;
- completion scale.

If label health is collapsed or only rescued by post-hoc threshold movement,
classify the result as `TSC_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

## Accepted Offline/Closed-Loop Boundary

Offline action metrics are development diagnostics only. TSC will not claim
paper viability from action L2, masked-cell Huber, or completion loss alone.

Before any closed-loop rollout, Stage 0 must show:

- the mask is noncollapsed;
- the completion field acts in selected cells;
- unselected cells remain Base-clamped;
- action changes are bounded and sparse;
- clean validation behavior is retained;
- official action validity is preserved;
- the full method beats the closest-prior proxy and key ablation on the
  predeclared development score.

Closed-loop task success remains the primary evidence once rollout begins.

## Accepted Reporting Requirements

For every mechanism smoke and policy preflight, TSC must report:

- Base action chunk shape and finite status;
- TSC action chunk shape and finite status;
- mask positive rate;
- changed cell count;
- changed dimensions;
- per-dimension delta norms;
- max and p95 action delta;
- action validity under official semantics;
- clean retention proxy;
- checkpoint save/reload;
- finite gradients and nonzero expected-parameter gradients;
- no frozen-parameter gradients.

## Accepted No-Privileged-Inference Rule

TSC inference may use only:

- current deployment RGB streams;
- current proprioception;
- language instruction;
- frozen Base SmolVLA chunk;
- learned TSC parameters.

It may not use demonstration targets, future frames, reward, success, reset
identity, simulator object pose, object labels unavailable from deployment
inputs, task-progress oracle, or confirmatory-test labels.

## Researcher A Response To Reviewer B

The criticism is accepted. The method will proceed only as a narrow,
falsifiable extension of TS-Mask to continuous Base-clamped SmolVLA chunk
completion. The mathematical audit must define the exact mask labels, tensor
shapes, objective scales, gradient paths, proxy, ablation, and Stage 0 gates
before implementation.

If the 2D time-dimension mask does not add measurable mechanism value over a
faithful TS-Mask proxy, no-targeted-mask ablation, standard LoRA, and simple
residual gates, TSC should be killed or stopped before closed-loop rollout.
