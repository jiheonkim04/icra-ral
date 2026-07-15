# COVI-VLA Researcher A Rebuttal

Date: 2026-07-15 KST

Proposal: `reports/covi_vla/researcher_proposal.md`

Proposal hash: `338430D2C6CF1D82410C036D79102ED3F38B2367BB35B9AE2811161698A3E621`

Reviewer attack: `reports/covi_vla/reviewer_attack.md`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Rebuttal decision: `COVI_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

## Accepted Narrowed Novelty

Researcher A accepts the narrowed novelty.

`COVI-VLA` is not official VIM, not a full viewpoint-imagination method claim,
not a calibration-free view-robust action-generation method, and not a generic
visual robustness defense. The defensible claim is:

> a frozen-SmolVLA, identity-preserving complementary-feature adapter for
> scene-induced occlusion, with a bounded gate initialized to Base passthrough
> and a development-only source gate that prevents clean-view, simulator-state,
> and confirmatory-identity leakage.

The method may proceed only if the mathematical audit and Stage 0 preserve this
narrow claim.

## Accepted Source Fidelity

Researcher A accepts that `vim_view_imagination_proxy` is a transparent local
proxy until official LIBERO-Occ/VIM equivalence is established.

The proposal may cite LIBERO-Occ/VIM as the closest positive prior, but it may
not:

- call the local proxy official VIM;
- compare local proxy numbers directly to incompatible published numbers;
- omit VIM's full generative image-token/action-token training mechanism;
- claim that a frozen visual adapter reproduces VIM.

If official LIBERO-Occ code or assets are used later, their exact local
coverage, omissions, checkpoints, and protocol differences must be documented.

## Accepted Direct-Fusion Challenge

Researcher A accepts the direct two-camera fusion challenge.

Stage 0 must separate four quantities:

1. Base policy under the predeclared occlusion condition;
2. a clean complementary-view oracle used only as a diagnostic upper bound;
3. direct use or reweighting of currently available camera streams;
4. predicted complementary representation from legal occluded deployment
   inputs.

COVI may proceed past Stage 0 only if the predicted complementary
representation has noncollapsed signal and adds value beyond direct
pass-through or trivial multi-view reweighting under the preregistered margin.

## Accepted Random-Cutout Simple Killer

Researcher A accepts that `random_cutout_clean_retention_baseline` remains live
and cannot be dropped, renamed, weakened, or moved out of the first serious
five-policy comparison.

The first serious comparison remains exactly:

1. `frozen_smolvla_occluded`
2. `vim_view_imagination_proxy`
3. `covi_full`
4. `covi_no_imagined_view_ablation`
5. `random_cutout_clean_retention_baseline`

Additional Stage 0 diagnostics may be recorded, but they do not replace the one
mandatory simple killer in the first paper-oriented comparison.

## Accepted Physical Occlusion Requirement

Researcher A accepts that COVI must test scene-induced occlusion or a faithful
local proxy for scene-induced occlusion.

If official LIBERO-Occ assets can be used locally within the campaign resource
budget, they are preferred. If they cannot be used immediately, any local proxy
must document:

- how the occlusion is physically or semantically tied to manipulated objects,
  receptacles, or dual task-relevant evidence;
- why the task remains executable;
- why failure can be attributed to partial observability rather than invalid
  task construction;
- how the proxy differs from ordinary rectangular cutout.

If Stage 0 uses only synthetic image masks without validating the connection to
scene-induced occlusion, the claim must be downgraded and the current paper
route is not allowed to proceed as COVI.

## Accepted Identity-Preserving Integration

Researcher A accepts that the adapter must initialize to Base passthrough and
that Stage 0 must report:

- initial action delta p95;
- translation, rotation, and gripper deltas;
- action-bound validity;
- clean validation behavior;
- feature residual norm;
- gate values;
- activation frequency and localization;
- checkpoint save/reload behavior;
- finite nonzero gradients for intended adapter and gate parameters.

If no stable SmolVLA visual-feature intervention point is available, the result
must be classified as `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, not rescued by
quietly changing the method into an action residual or generic wrapper.

## Accepted No-Privileged-Inference Rule

Researcher A accepts that all of the following are forbidden at inference:

- clean unoccluded evaluation images;
- future frames or future actions;
- segmentation masks;
- simulator object pose;
- object visibility labels;
- reset identities or manifest keys;
- task success/failure outcomes;
- confirmatory-test outcomes or identity-specific tuning.

Privileged labels may be used only as discovery/validation training labels or
diagnostic oracle upper bounds when absent from inference and absent from
confirmatory-test tuning.

## Proceed Condition

Proceed to mathematical mechanism audit only under these accepted constraints.
The audit must define exact variables, tensor shapes, feature intervention
point, objective terms, scale/gradient checks, direct-fusion diagnostics,
random-cutout baseline, VIM proxy status, identity-preserving initialization,
and Stage 0 stop classes.

No implementation, validation search, rollout, or confirmatory evaluation may
occur before that audit and preregistration are frozen.
