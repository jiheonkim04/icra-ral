# EvoState-VLA Reviewer B Attack

Date: 2026-07-14 KST

Proposal hash under review: `A44ED68CC8E1F296DB8B0B3E16FF84D7D5BBE684EAF63EAE29E7CC91DCFD93C9`

Reviewer decision before rebuttal: `CONDITIONALLY_ALLOW_ONLY_WITH_STRICT_AUDIT`

## Closest Three Papers

1. EvoScene-VLA, https://arxiv.org/abs/2605.21862
2. DREAM-Chunk, https://arxiv.org/abs/2606.18589
3. Adaptive Action Chunking, https://arxiv.org/abs/2604.04161

Additional pressure sources:

- A2C2, https://arxiv.org/abs/2509.23224
- Health-conditioned VLA, https://arxiv.org/abs/2605.16056
- PDF, https://arxiv.org/abs/2604.18107

## Major Novelty Risk

The proposal may be a weak local proxy for EvoScene rather than a new method. EvoScene's contribution is a geometry-aware recurrent scene prefix inside the action decoder, trained with scene-prediction and geometric-anchor supervision. EvoState discards the scene representation and uses only 8D robot state.

Reviewer concern:

- If only robot proprioception is used, the method may not solve the scene-state problem EvoScene identified.
- If object/contact state is essential, the local method may have no observability.
- If the method works only under artificial proprioceptive faults, the EvoScene prior may be the wrong anchor.

Required response:

- Frame the contribution as a deployment-feasible action-evolved state controller, not as full scene-belief modeling.
- The audit must test whether the necessary latent is observable from 8D state and actions. If not, classify `DESIGN_FAILURE`.

## DREAM-Chunk Equivalence Risk

DREAM-Chunk uses a latent world model for robust chunk execution. EvoState also uses a latent dynamics model during chunk execution. The difference must be more than "smaller DREAM".

Reviewer concern:

- If EvoState only predicts state and selects/corrects chunks, it could be a simplified DREAM-Chunk.
- If the DREAM-lite proxy beats EvoState, the proposed controllability correction adds no value.

Required response:

- DREAM-lite must be an early baseline.
- The selected method must show a mechanism distinction: correction in a controllable mismatch direction, not only candidate selection.

## RCV And FEDO Revival Risk

The campaign already killed:

- RCV, where no-context/stateless replanning explained the result.
- FEDO, where action-realization correction lost to simple static inverse gain and clean behavior collapsed.

Reviewer concern:

- EvoState may revive RCV by using a fancier state mismatch gate to decide when to replan.
- EvoState may revive FEDO by adding an inverse-dynamics correction under action faults.

Required response:

- `evostate_no_state_prior_ablation` must remove the persistent predicted state while keeping correction capacity.
- `static_inverse_dynamics` must be the simple killer baseline.
- Stage A/B cannot claim success unless EvoState beats both.

## Data And Supervision Risk

The proposed transition tuples come from frozen SmolVLA rollouts under the clean environment. They may not contain enough controlled mismatch examples.

Reviewer concern:

- A transition model trained only on clean traces may not extrapolate to mismatch states.
- Consecutive step records may include repeated chunk actions that produce low state diversity.
- The model may learn trivial time/phase dynamics rather than controllability.

Hard audit requirements:

- count valid consecutive transition pairs per task and identity;
- report state variance by dimension;
- report next-state prediction against constant, actionless, per-task linear, and previous-state baselines;
- report controllability rank and condition number;
- report validation mismatch bins, not only global MSE;
- reject if action input does not improve next-state prediction over an actionless model.

## Mechanism Observability Risk

EvoState uses only 8D proprioception, not images or object pose. Many LIBERO failures are object/contact state failures.

Reviewer concern:

- If the robot state cannot reveal object slippage, wrong grasp, or release position, the controller may correct the robot toward a bad expected trajectory.
- The method could make successful recovery less likely by tracking a stale predicted path.

Required response:

- The method must gate corrections off when mismatch is not controllable by the learned action-state map.
- Clean validation action deltas must remain small.
- Closed-loop metrics must report when the gate activates and on which state dimensions.

## Mathematical Risk

The inverse correction formula can become unstable when `B B^T` is singular.

Required response:

- Use damped least squares with a preregistered damping coefficient.
- Clip the correction norm.
- Reject configurations with nonfinite corrections or invalid action bounds.
- Do not use KL, entropy, or probability language unless actual distributions are defined.

## Experimental Risk

The primary condition must not be cherry-picked after seeing outcomes.

Required response:

- The controlled mismatch condition must be declared before rollout.
- Base, DREAM-lite proxy, EvoState, no-state-prior ablation, and static inverse dynamics must share the identical task/reset manifest.
- If Stage B is clearly negative or tied with simple baselines, no rescue is allowed.

## Required Changes Before Preregistration

1. Explicitly define the controlled mismatch condition.
2. Define the transition-pair construction and excluded identities.
3. Define exact audit hard stops.
4. Define the no-state-prior ablation and static inverse baseline.
5. Define the DREAM-lite proxy sufficiently to prevent a straw baseline.
6. Define the validation score and six-config search budget.
7. Define action-delta and clean-retention gates.

If these are added, Reviewer B allows one bounded development audit. Reviewer B does not approve any closed-loop rollout until audit and validation pass.
