# G3P-VLA Reviewer B Attack

Date: 2026-07-15 KST

Reviewed frozen proposal: `reports/g3p_vla/researcher_proposal.md`

Proposal hash: `BEE3822D8F54EFBD09C1CA47A9BF126EBE694B7B6219002FF770C5794ED7AA71`

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Primary sources reviewed:

- Direct Action-Head Injection of A Grounded 3D Point Unlocks Spatial and Task Generalization, https://arxiv.org/abs/2606.27663
- RoboPoint, https://arxiv.org/abs/2406.10721
- RoboGround, https://arxiv.org/abs/2504.21530
- AffordanceVLA, https://arxiv.org/abs/2606.06155
- ActionMap paper and pre-release repository, https://arxiv.org/abs/2606.06904 and https://github.com/showlab/ActionMap

## Summary Ruling

Do not kill G3P-VLA before implementation. The closest prior is a strong positive anchor: it directly supports the idea that a grounded 3D point represented relative to the gripper and injected at the action interface can improve VLA spatial and task generalization.

However, the proposal's novelty is narrow and conditional. The closest prior already contains the core 3D displacement plus action-head injection mechanism. G3P is viable only as a local frozen-SmolVLA, source-gated, identity-preserving adaptation of that idea, and only if Stage 0 proves that the required point source is legal, observable from deployment inputs, noncollapsed, and not explained by a trivial 2D or task-phase heuristic.

## Attack 1: Closest-Prior Duplication Risk

The nearest prior already claims the main mechanism:

- convert task grounding into a 3D point;
- compute gripper-relative displacement;
- inject that spatial signal into the action head;
- show large LIBERO-PRO gains on multiple VLA backbones.

G3P therefore cannot claim broad novelty for grounded 3D point injection, gripper-relative displacement, or action-head conditioning. The only defensible novelty is local:

- frozen SmolVLA integration;
- strict source gating to forbid privileged inference;
- exact or near-exact Base passthrough initialization;
- bounded action deltas on a chunked 7D SmolVLA action surface;
- a fair local comparison to a closest-prior proxy, a no-3D/no-injection ablation, and a simple heuristic killer.

Required rebuttal:

- Explicitly disclaim broad 3D point injection novelty.
- State that `g3p_3d_point_proxy` dominance kills or archives the local contribution.
- State that official-prior reproduction is not claimed unless official code/checkpoints and protocol equivalence are later verified.

## Attack 2: The Point Source May Be Privileged

The proposal depends on target points. In LIBERO-like simulation, target object pose, placement pose, reset identity, task metadata, or success state can easily leak through implementation shortcuts. Any hidden oracle point at inference would invalidate the method.

Hard requirements:

- Stage 0 must list every source of point labels, point predictions, gripper position, camera calibration, and frame transform.
- Inference may use only deployment RGB, language, proprioception, and Base features/actions available through the official runner.
- Simulator object pose, placement coordinates, reward, success flags, reset identity, future frames, or confirmatory manifest metadata may be used only for discovery/validation labels or diagnostics, never for confirmatory inference.
- The final runner must contain an explicit source gate proving no privileged fields are accessed.

Reject before rollout as `DATA_OR_SUPERVISION_FAILURE` or `DESIGN_FAILURE` if a legal deployment-observable point source cannot be built.

## Attack 3: 2D Affordance Or Task Phase May Explain The Method

RoboPoint supports language-conditioned 2D affordance points, RoboGround supports grounding masks, and AffordanceVLA supports structured affordance cues. The local gain may come from coarse 2D localization or task-phase timing rather than 3D displacement and action-head injection.

Required constraints:

- Keep `simple_2d_phase_or_nearest_object_heuristic` as the single simple reviewer-killer.
- The simple heuristic must be selected before confirmatory testing using validation-only evidence.
- Stage 0 must measure whether point labels or point predictions are task, phase, or object-name shortcuts.
- If a 2D point or nearest-object heuristic matches G3P, the 3D injection claim is not supported.

## Attack 4: Local Closest-Prior Proxy Could Be Unfair

No official closest-prior code or checkpoint is verified in this repository. A weak proxy would make G3P look better unfairly; an overpowered proxy could also hide a useful local integration. Either way, the comparison must be transparent.

Hard requirements:

- Label `g3p_3d_point_proxy` as a faithful transparent local proxy, not an official reproduction.
- Use the same legal point source, same data partitions, same inference input restrictions, and comparable inference budget for proxy and full method.
- Document the exact technical difference between proxy and full method.
- If official code/checkpoints become available later, official-equivalence claims require a separate documented equivalence audit before confirmatory testing.

## Attack 5: Frame, Units, And Calibration Are Failure-Prone

The proposal uses gripper-relative 3D displacement. In practice, camera frame, robot base frame, end-effector frame, normalization, action units, and sign conventions can silently flip or scale the signal. A method can appear active while injecting the wrong vector.

Stage 0 and mathematical audit must define:

- point frame and gripper frame;
- transform from pixel/depth or simulator label to deployment vector;
- units for translation, rotation, and gripper dimensions;
- normalization and clipping constants;
- sign convention for `d = p_t - p_g`;
- valid range for every 7D action component.

Reject before rollout as `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` if source vectors cannot be sanity-checked against known gripper/object geometry on development identities.

## Attack 6: SmolVLA Action Interface May Not Match The Prior

The closest prior injects through an action head on specific backbones. SmolVLA's local action interface, chunking behavior, hidden states, and flow-style action generation may differ. A naive adapter can damage a strong Base policy even if the prior is correct elsewhere.

Hard requirements:

- Initial G3P behavior must be exact or near-exact Base passthrough.
- Translation, rotation, and gripper deltas must be bounded and reported separately.
- The adapter must be zero-initialized, gate-initialized to Base passthrough, or otherwise identity-preserving.
- Stage 0 must verify finite nonzero gradients only in intended point/adaptation parameters.
- Before any rollout, disk-reload must preserve the same policy identity and action-delta statistics.

If the adapter globally changes actions or changes gripper behavior everywhere, stop as `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, not as a closed-loop result.

## Attack 7: Label Health And Observability Are Not Optional

G3P can fail scientifically only after proving the underlying supervision exists. Collapsed point labels, single-object shortcuts, missing depth, or unpredictable points from RGB are pre-rollout design/data failures.

Reject before rollout if:

- point labels are all-zero, all-one, single-task, or single-phase;
- target-object and placement examples are collapsed into one mode;
- positive/negative confidence examples are unavailable;
- train/validation/test frame, task, episode, or reset identities overlap;
- point prediction from deployment inputs fails to beat majority, phase, nearest-object, or language-only baselines;
- confidence is always high, always low, or identical across tasks;
- oracle headroom shows no plausible spatial gain.

## Attack 8: The Key Ablation Must Be Matched

The no-3D/no-injection ablation must not be a deliberately weak strawman. It should keep the same training records, Base policy, data partitions, and comparable parameter budget where possible, while removing the specific 3D displacement and action-injection pathway.

Required reporting:

- full-versus-ablation action L2 by translation, rotation, and gripper;
- mechanism activation contexts;
- whether the ablation receives 2D, language, phase, or confidence cues;
- whether full and ablation differ on validation before rollout.

If full and ablation are action-equivalent on validation, stop as exact trivial equivalence before rollout.

## Attack 9: Mathematical Objective Risks

The mathematical audit must not decorate a simple residual with unjustified divergences. In particular, do not compute KL directly between deterministic 7D actions.

Required definitions:

- image tensor, proprioception tensor, language embedding, point tensor, confidence scalar, gripper position, displacement, adapter embedding, Base action chunk, and adapted action chunk shapes;
- action formula, gate formula, clipping/bounding formula, and gradient paths;
- small-batch objective magnitudes and gradient norms for point prediction, action imitation, Base retention, confidence calibration, and delta regularization;
- simpler alternative objective and required ablation;
- whether the point predictor is trained jointly, frozen, or generated offline.

## Required First Comparison

The first serious comparison must remain exactly:

1. `frozen_smolvla`
2. `g3p_3d_point_proxy`
3. `g3p_full`
4. `g3p_no_3d_no_injection_ablation`
5. `simple_2d_phase_or_nearest_object_heuristic`

Additional baselines may not precede this comparison unless Stage 0 exposes a concrete implementation ambiguity that would otherwise invalidate the five-policy test.

## Required Stage 0 Stop Rules

Stage 0 must stop before training search or rollout for any of:

- legal deployment-observable point source cannot be constructed;
- point labels or confidence targets are collapsed;
- point prediction is not observable above trivial baselines;
- oracle diagnostics show no spatial headroom;
- train/validation/test or reset identity separation fails;
- source gate detects privileged inference access;
- adapter cannot preserve Base behavior at initialization;
- intended parameters receive no finite nonzero gradients;
- action deltas are global or unbounded;
- full, proxy, ablation, or simple heuristic are action-equivalent in a way that invalidates the frozen comparison.

## Reviewer B Decision

`REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

G3P may proceed to Researcher A rebuttal because it is strongly anchored, meaningfully different from the just-killed EAC queue-scheduling route, and plausibly targets a real spatial-generalization failure mode. It cannot proceed to mathematical audit, implementation, Stage 0, validation search, training, or rollout until Researcher A accepts the narrowed novelty, source-legality requirements, trivial-baseline constraints, and identity-preserving integration gates above.
