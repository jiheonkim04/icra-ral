# Epoch 4 Cycle 29 Candidate Generation

Date: 2026-07-16 KST

Decision: `CCIF_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

Exactly three candidates were generated after the Cycle 29 prior mechanism map.

Previous method `TSC-VLA` remains closed as
`TSC_STAGE_0_NO_USABLE_HEADROOM`; no repair or rescue is allowed.

## Candidate 1: CCIF-VLA

Full name: Continuous Coarse Intent Field for base-preserving VLA chunks.

Closest prior:

- Coarse-to-Control, https://arxiv.org/abs/2606.07107.

Prior positive result:

- Coarse-to-Control reports `97.9%` average LIBERO success and uses
  action-token planning before executable action generation.

Proposed mechanism:

- decompose each demonstration action window into a coarse motor-intent field:
  net translation, net rotation, gripper endpoint, and low-frequency waypoint
  summaries over the future chunk;
- train a deployment-observable intent predictor from current visual features,
  proprioception, instruction/task identity, and Base decoded chunk;
- condition a bounded residual action field on the predicted coarse intent;
- preserve Base exactly at initialization and constrain the residual so the
  method changes only directions supported by the coarse intent.

Technical difference from prior:

- Coarse-to-Control uses discrete plan/execution action tokens in a shared
  residual-VQ vocabulary;
- CCIF uses a continuous coarse intent field as an action-space constraint for
  an already continuous SmolVLA chunk, without adding a discrete decoder.

Problem chain:

condition -> multi-stage or long-horizon windows require an intermediate motor
intent not fully represented by the current chunk.

failure mechanism -> Base predicts locally plausible motion but drifts in net
direction, gripper endpoint, or low-frequency waypoint structure.

policy behavior -> direct residual learning overfits high-frequency action
details and fails to preserve long-horizon motor intent.

proposed method -> predict and enforce a continuous coarse intent field before
low-level residuals.

expected behavior -> residuals remain bounded and directionally consistent with
coarse future motor structure, improving targeted long-horizon windows while
retaining clean Base behavior.

Data/supervision viability:

- labels exist in existing LIBERO demonstrations;
- no privileged inference input is required;
- future actions are used only to create training labels;
- Stage 0 can audit coarse-intent predictability, Base residual headroom, proxy
  comparison, identity preservation, and action validity.

Identity-preserving integration:

- residual branch initialized to zero;
- Base passthrough default;
- intent gate initialized to zero influence;
- clean-retention loss required if training proceeds.

First serious comparison:

1. `smolvla_base`
2. `coarse_to_control_continuous_proxy`
3. `ccif_full`
4. `ccif_no_coarse_intent_ablation`
5. `standard_lora`

Score:

- provisional novelty: `23 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `19 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `92 / 100`

## Candidate 2: URF-VLA

Full name: Uncertainty-gated Residual Flow for base-preserving SmolVLA chunks.

Closest prior:

- SUREFlow, https://arxiv.org/abs/2607.10504, with official code at
  https://github.com/tanvirnwu/SUREFlow.

Prior positive result:

- SUREFlow reports `92.5%` average LIBERO success and competitive LIBERO-PRO
  robustness using `179M` parameters.

Proposed mechanism:

- train a heteroscedastic residual-flow head over SmolVLA action chunks;
- use predicted residual uncertainty to gate where a bounded residual flow is
  allowed to refine Base;
- keep low-uncertainty cells exactly or nearly Base.

Technical difference from prior:

- SUREFlow trains an end-to-end state-space VLA with uncertainty-aware residual
  flow;
- URF-VLA is a frozen-backbone, Base-preserving overlay whose novelty is
  uncertainty-gated residual intervention on a pretrained SmolVLA chunk.

Risk:

- must not collapse into a mere confidence head;
- must show that uncertainty changes the residual action field and not only
  report calibration.

First serious comparison:

1. `smolvla_base`
2. `sureflow_uncertainty_residual_proxy`
3. `urf_full`
4. `urf_no_uncertainty_gate_ablation`
5. `standard_lora`

Score:

- provisional novelty: `20 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `88 / 100`

## Candidate 3: ASR-VLA

Full name: Action-Side Routing for task-progress dependent SmolVLA residual
experts.

Closest prior:

- CoRE-VLA, https://arxiv.org/abs/2607.03693.

Prior positive result:

- CoRE-VLA reports `98.7%` average LIBERO success and shows noncollapsed
  task/phase expert routing patterns.

Proposed mechanism:

- route action-side temporal/dimensional residual experts using current visual
  features, proprioception, instruction identity, and phase proxies;
- apply only one or two selected residual experts per action chunk;
- initialize all residual experts to Base passthrough.

Technical difference from prior:

- CoRE routes internal action-side representations in a full expert action
  generator;
- ASR-VLA would route bounded residual experts around frozen SmolVLA outputs.

Risk:

- close to previously killed task/instruction adapter routing unless Stage 0
  proves the routing is action-side, phase-sensitive, noncollapsed, and
  behaviorally meaningful.

First serious comparison:

1. `smolvla_base`
2. `core_vla_action_routing_proxy`
3. `asr_full`
4. `asr_no_action_side_routing_ablation`
5. `standard_lora`

Score:

- provisional novelty: `17 / 25`
- importance of problem: `12 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `80 / 100`

## Selection

Selected method: `CCIF-VLA`

Selection reason:

CCIF-VLA has the strongest balance of novelty, prior anchoring, local
supervision viability, and decisive Stage 0 feasibility. It extends a strong
action-space planning prior into a continuous, identity-preserving SmolVLA
mechanism instead of replaying killed routing, generic correction, or
uncertainty-only routes.

Next action:

Freeze the CCIF-VLA Researcher A proposal before Reviewer B attack,
mathematical audit, preregistration, or implementation.
