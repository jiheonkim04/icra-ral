# DAGR-VLA Reviewer B Attack

Date: 2026-07-14 KST

Reviewed frozen proposal: `reports/dagr_vla/researcher_proposal.md`

Proposal hash: `BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89`

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

## Closest Prior Attack

Closest prior: DAM-VLA, https://arxiv.org/abs/2603.00926.

DAM-VLA already claims the central idea that arm movement and gripper manipulation should be handled by specialized action models and dynamically coordinated by an action-routing mechanism. DAGR-VLA cannot claim the broad novelty of dynamic arm/gripper routing. Its possible novelty is narrower:

- frozen SmolVLA action preservation;
- small route-gated residual module instead of a full dynamic action decoder;
- group-specific residual learning from expert-minus-base targets;
- a matched local comparison to a static DAM-style component proxy and shared-residual ablation.

The proposal must state clearly that `dam_static_component_proxy` is a faithful transparent local proxy, not an official DAM-VLA reproduction.

## Trivial Equivalence Risks

1. Generic residual adapter

If DAGR's route gates activate on most frames or the same residual is applied to all groups, the method collapses to a generic residual adapter. Stage 0 must require route sparsity, group-specific activation differences, and full-versus-shared-residual separation.

2. Static arm/gripper loss weighting

If the dynamic route head does not change behavior beyond static component weights, the DAM-style proxy explains the method. The key comparison must include a static component-weighted proxy with the same training records and comparable parameter budget.

3. Gripper-transition heuristic

If improvements come only from gripper open/close timing, a simple gripper-transition heuristic explains the result. This must be the mandatory simple killer baseline.

4. MTF-style frame selection

If DAGR reuses milestone or gripper-transition scoring primarily to select frames, it becomes an MTF-adjacent rescue. Route labels may use gripper-transition events, but the method's central mechanism must be action-group routing, not training-frame selection.

## Leakage And Partition Risks

- Expert-minus-base residual targets may be generated only on discovery/training/validation identities before confirmatory testing.
- Route thresholds and residual alpha must be selected on validation only.
- Confirmatory task/reset identities may not influence route-label definitions, group thresholds, alpha, policy list, or decision thresholds.
- If base actions are generated for training labels, the exact frozen base checkpoint and preprocessing route must be recorded.
- If official SmolVLA stable prediction artifacts are reused, split overlap must be proven at frame, task, episode, and reset identity levels.

## Mathematical Risks

- Do not compute KL between deterministic 7D action vectors.
- Group-normalized Huber/L2 losses must report scale by translation, rotation, and gripper group.
- Route-label BCE/focal loss must report positive/negative counts and gradient norm relative to residual loss.
- The action formula must define clipping, group masks, tensor shapes, and gradient flow.
- If residuals are clipped at inference but not during training, the mismatch must be documented.

## Data Health Risks

The proposal depends on route labels being noncollapsed and predictable. Reject before rollout if:

- any route label is all-zero/all-one or has fewer than a preregistered minimum positive examples;
- route prediction cannot beat a trivial majority baseline on validation;
- one task family dominates the positive route examples;
- gripper labels are merely timestep/phase labels;
- full and ablation receive effectively identical targets;
- route gates are active everywhere on validation.

## Identity-Preservation Risks

DAGR can disrupt a strong base policy if residuals act globally. Before rollout, require:

- exact zero-initialized residual or measured initial equality to Base;
- action validity for the full 7D action;
- translation, rotation, and gripper delta summaries;
- route activation frequency by action group;
- clean validation action delta p95;
- full, static proxy, shared ablation, and simple heuristic all loaded from frozen identities.

## Required First Comparison

The first serious comparison must use exactly five policies:

1. `frozen_smolvla`
2. `dam_static_component_proxy`
3. `dagr_full`
4. `dagr_no_dynamic_route_ablation`
5. `gripper_transition_heuristic`

Additional internal controls may not precede this comparison unless a Stage 0 audit shows a concrete implementation ambiguity that would otherwise invalidate the five-policy test.

## Reviewer Verdict

Do not reject before implementation. DAGR-VLA is not an exact duplicate of DAM-VLA because it is a frozen-policy route-gated residual adapter with identity-preserving constraints and a fair local proxy comparison. However, the novelty is conditional and narrow. The method is viable only if Stage 0 proves noncollapsed, observable route labels and if the full model demonstrably differs from static component weighting and shared residual ablations before rollout.
