# AMP-VLA Reviewer B Attack

Date: 2026-07-16 KST

Role: Reviewer B

Frozen proposal hash:
`67ACC693C706B76BC9FB84F9E59BA3DF9C0463A0BAFABE539312D0E232DFE9A4`

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

AMP is not rejected before rebuttal because it is anchored to a current positive
external prior and because support-preserving action-manifold adaptation is a
distinct mechanism from ordinary LoRA, RAP retrieval anchors, VDR future-feature
residuals, or direct action clipping. However, the proposal is high-risk unless
Researcher A accepts the constraints below.

## 1. ABot-M0 Proximity Is The Central Novelty Risk

ABot-M0 already claims Action Manifold Learning for continuous robot actions.
AMP cannot claim broad novelty for "using an action manifold" or for simply
projecting actions onto a learned low-dimensional support.

Reviewer requirement:

- the closest-prior comparison must remain policy 2:
  `abot_m0_action_manifold_proxy`;
- if official ABot assets are not installed and verified, policy 2 must be
  labeled as a transparent local proxy, not an official reproduction;
- every deviation from official ABot-M0 action manifold learning must be listed
  before Stage 0;
- AMP's novelty must be narrowed to frozen-SmolVLA identity-preserving
  manifold-constrained residual adaptation;
- no generic claim about action manifolds, LoRA, or efficient robot learning is
  allowed.

## 2. Projection Must Not Collapse Into Clipping

Recent campaign failures repeatedly involved action-validity gates. AMP's
projection may become a disguised clipping or range-safety filter if it only
forces outputs into bounds.

Reviewer requirement:

- define the action-validity unit system before Stage 0: raw normalized chunks,
  postprocessed 7D LIBERO actions, or both;
- separately report bound validity and manifold consistency;
- show that projected actions are closer to demonstration support than simple
  coordinate clipping under the preregistered metric;
- include a cheap clipping or bound-only diagnostic inside Stage 0 if clipping
  could explain the projection benefit;
- no clipping rescue, bound widening, post-hoc validity reinterpretation, or
  threshold change after Stage 0 begins.

## 3. Manifold Health Must Be Measured Before Training

A collapsed action manifold would make AMP meaningless. A manifold that simply
encodes task identity or phase means would also be too weak.

Reviewer requirement:

- report retained dimension, explained variance, reconstruction Huber, per-task
  coverage, phase coverage, and coordinate variance;
- require a task/phase mean action baseline and a task/phase mean coordinate
  baseline;
- manifold reconstruction must beat the task/phase mean action predictor by
  the frozen margin;
- deployment-input coordinate prediction must beat the task/phase coordinate
  predictor by the frozen margin;
- if any retained coordinate is all-zero, all-one, duplicate, or nonfinite, stop
  as `AMP_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

## 4. ABot Proxy Headroom Must Remain Live

AMP can only be a paper candidate if it improves on the closest prior, not only
on Base. A weak ABot proxy would inflate AMP.

Reviewer requirement:

- first attempt to identify whether official ABot-M0 assets can be integrated
  within local budget;
- if a proxy is used, match action normalization, action chunking, manifold
  fitting data, inference budget, and postprocessing as closely as possible;
- freeze the proxy before AMP validation performance is known;
- Stage 0 must estimate whether the proxy leaves usable residual headroom;
- if ABot proxy matches or beats AMP in the first serious comparison, AMP is
  not a paper candidate.

## 5. Standard LoRA Remains Mandatory

AMP changes policy behavior through trainable adapter infrastructure. Ordinary
data-matched adaptation is therefore a plausible explanation.

Reviewer requirement:

- matched standard LoRA must use the same demonstrations, optimizer steps, rank,
  target modules, clean-retention coefficient where applicable, and checkpoint
  selection budget;
- if standard LoRA matches or beats AMP under the frozen manifest, AMP does not
  become a paper candidate;
- AMP may not substitute a weaker simple baseline after seeing validation
  results.

## 6. Identity Preservation Is A Hard Integration Requirement

AMP is motivated by support-preserving behavior. A globally acting projection or
residual that changes most Base actions is a failure, even if offline loss
improves.

Reviewer requirement:

- initialized and disk-reloaded AMP must reproduce Base within `1e-6`;
- report Base action, AMP action, projection delta, residual norm, gate value,
  changed dimensions, and activation context for representative rows;
- activation must be relevant-state selective, not everywhere;
- action delta p95, translation delta, rotation delta, gripper delta,
  intervention frequency, and action validity must be saved before rollout;
- a catastrophic global action change stops as
  `AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.

## 7. Objective Engineering Must Avoid Decorative Math

The proposal correctly avoids KL between deterministic actions, but the
projection objective can still hide scale bugs.

Reviewer requirement:

- mathematical audit must define `Phi`, `DecodeManifold`, `P`, `P_mix`,
  `lambda_m`, `lambda_p`, `lambda_clean`, tensor shapes, units, and gradient
  paths;
- inspect small-batch loss magnitudes and gradient norms before training;
- normalize or justify objective scales;
- report gradient norm ratios and frozen-parameter gradient count;
- if projection is nondifferentiable, specify exactly where gradients flow and
  what term supervises the trainable adapter.

## 8. Validation Score Cannot Be Offline Action L2 In Disguise

AMP targets closed-loop support, not just lower action regression error.

Reviewer requirement:

- validation score must include clean retention, action validity, mechanism
  activation, and AMP-minus-prior/ablation margins;
- offline action L2 or Huber alone cannot select the final configuration;
- all six tried configurations, including negative results, must be saved;
- confirmatory outcomes may not retune latent dimension, projection strength,
  thresholds, or coefficients.

## 9. Conditional Pass

AMP may proceed to Researcher A rebuttal only if Researcher A accepts:

1. ABot-M0 proxy as the closest-prior policy in the first serious comparison;
2. honest official-vs-transparent ABot status;
3. no-projection ablation and matched standard LoRA;
4. clipping/bound-only diagnostics so projection is not confused with clipping;
5. frozen discovery/validation/test separation for manifold fitting;
6. manifold health, headroom, and predictability Stage 0 gates;
7. exact identity/reload and bounded-action-integration checks;
8. mathematical objective and gradient-scale audit;
9. no RAP repair, rescue, threshold change, or reinterpretation through AMP.

If any of these are rejected, AMP should stop before implementation as
`AMP_REVIEWER_REJECTED_OR_NEEDS_REDESIGN`.
