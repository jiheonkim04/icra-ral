# TSC-VLA Reviewer B Attack

Date: 2026-07-16 KST

Method under review: `TSC-VLA`

Proposal hash:
`0DF143D2D8773D7ABF4FC76AB7CC083FE7EE65DF84EA06631E67C2445F6DC941`

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

## Independent Prior Check

Reviewer B identifies the closest current primary sources as:

1. TS-Mask VLA, `https://arxiv.org/abs/2607.09818`
2. Frequency-Aware Flow Matching, `https://arxiv.org/abs/2606.20135`
3. Guided Action Flow, `https://arxiv.org/abs/2607.02092`

TS-Mask VLA is the direct novelty threat. It already claims 2D
temporal-spatial masking over action tokens, a discrete diffusion action
expert, Bridge Attention conditioning, `95.7%` average LIBERO success, and
CALVIN average sequence length `4.19`. TSC cannot claim novelty for "masking
action tokens", "temporal-spatial action structure", or "masked action
generation" broadly.

The only defensible TSC novelty is narrower:

> Base-clamped continuous `[50,7]` action-cell completion, targeted by a
> deployment-observable Base-error mask, around an existing frozen SmolVLA
> flow-matching policy.

## Main Novelty Risks

### 1. TSC May Collapse Into TS-Mask Proxy

If the method is just continuous TS-Mask with a different action representation,
the closest prior owns the mechanism. The first comparison must include a
faithful `ts_mask_continuous_proxy`, not a weak random-mask toy.

Condition: the proxy must use the same training split, observation inputs,
Base chunks if necessary for continuous compatibility, comparable compute, and
the same action validity checks. The only missing element should be TSC's
Base-error-targeted sparse mask.

### 2. TSC May Collapse Into A Confidence Head

A predicted error mask alone is not a method contribution. Prior campaign rules
reject confidence-only, failure-detection-only, and candidate-ranking-only
methods.

Condition: TSC must show that the completion field changes action behavior in
the selected cells and that the full method beats:

- the same mask predictor without completion;
- completion with non-targeted masks;
- magnitude-only or residual-size mask baselines.

### 3. TSC May Be Sparse Residual Gating Under A New Name

The formula `A_TSC = A_B + M * Delta` resembles a sparse residual adapter.
This is only defensible if the temporal-spatial `[H,D]` mask structure matters.

Condition: Stage 0 must include evidence that a structured 2D mask over
time-dimension cells beats at least one simpler residual formulation, such as:

- global residual gate;
- per-timestep gate;
- per-dimension gate;
- residual magnitude threshold baseline.

### 4. Offline Action Error May Not Predict Closed-Loop Success

The campaign already learned that offline action L2 is not a reliable proxy for
closed-loop task success. A Base-vs-demo error mask might identify cells that
reduce teacher-forced loss but harm closed-loop behavior.

Condition: Stage 0 may use offline headroom only as development evidence. It
must not become a paper claim, and validation selection must include clean
retention, action validity, bounded intervention, and mechanism activation. If
Stage A happens, closed-loop task success remains primary.

### 5. Error-Label Thresholds Are A Tuning Trap

Per-cell positive labels can be made noncollapsed by choosing thresholds after
looking at validation. That would be post-hoc protocol manipulation.

Condition: mask-label thresholds, top-k rules, or robust-MAD construction must
be frozen before the decisive Stage 0 audit or selected only by predeclared
discovery/validation rules. Confirmatory outcomes may never tune them.

### 6. Official Action Semantics Must Be Preserved

TSC must not introduce arbitrary action bounds or clipping just because a
completion head produces invalid values. The current official action semantics
for SmolVLA are postprocessor-shape/finite/action-validity checks unless the
official stack exposes bounds.

Condition: if action validity fails or requires ad hoc bounds, classify as
`TSC_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, not as a scientific
closed-loop result.

### 7. Privileged Inputs Must Stay Out

The proposal is valid only if inference uses current deployment observation,
language, proprioception, Base chunk, and learned parameters. It must not use:

- demonstration action targets;
- future frames;
- success/reward;
- reset identity;
- simulator object pose;
- task progress oracle;
- precomputed confirmatory-test labels.

## Required Rebuttal Commitments

Researcher A must explicitly accept all of the following before mathematical
audit:

1. TSC novelty is narrowed to Base-clamped continuous action-cell completion,
   not temporal-spatial masking broadly.
2. `ts_mask_continuous_proxy` or official TS-Mask VLA remains policy 2 in the
   first serious comparison.
3. The proxy must be faithful enough that Reviewer B cannot call it a strawman.
4. `tsc_no_targeted_mask_ablation` remains the key ablation.
5. Matched `standard_lora` remains required unless Reviewer B later proves it
   irrelevant under a frozen inference-only implementation.
6. Stage 0 must test whether the 2D time-dimension mask beats simpler residual
   gates or magnitude-only mask baselines.
7. Error-label threshold construction must be frozen before decisive Stage 0
   evaluation.
8. Offline action L2 cannot be the sole validation score or paper claim.
9. TSC must report Base action, TSC action, changed cells, mask positive rate,
   per-dimension deltas, action validity, and clean retention.
10. No privileged inference input and no confirmatory-test tuning.

## Conditional Pass

Reviewer B conditionally allows TSC to continue because the narrowed mechanism
is distinct enough to test:

- TS-Mask uses native discrete token masking inside a diffusion action expert;
- TSC would perform targeted continuous masked completion around a frozen Base
  SmolVLA chunk;
- existing LIBERO demonstrations can supply the required development labels;
- the first comparison can include the closest prior proxy early.

This is not yet a novelty pass for paper claims. It is permission to rebut,
mathematically audit, and preregister a bounded Stage 0 that can kill the method
before rollout if the mask or completion mechanism is trivial.
