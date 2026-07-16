# MHS-VLA Reviewer B Attack

Date: 2026-07-16 KST

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Reviewed proposal: `reports/mhs_vla/researcher_proposal.md`

Proposal hash:
`BBDF67AE3EC4BD9D025707A8BB3A5008BAB5EB5C691D02D44516157802A87BF3`

## Independent Prior Search

Primary sources checked:

- MTIL: `https://arxiv.org/abs/2505.12410`,
  `https://arxiv.org/html/2505.12410v3`,
  `https://github.com/yulinzhouZYL/MTIL`
- MaIL: `https://arxiv.org/abs/2406.08234`,
  `https://github.com/ALRhub/MaIL`
- Mamba Policy: `https://arxiv.org/abs/2409.07163`
- AEM: `https://arxiv.org/abs/2606.12499`,
  `https://arxiv.org/html/2606.12499v1`
- MEM: `https://arxiv.org/abs/2603.03596`
- MemoryVAM: `https://arxiv.org/abs/2606.20679`,
  `https://arxiv.org/html/2606.20679v1`
- Diff-Control: `https://arxiv.org/abs/2404.12539`,
  `https://github.com/ir-lab/Diff-Control`

## Novelty Attack

MHS cannot claim novelty as:

- using Mamba for imitation learning;
- using full history for manipulation;
- using memory in a VLA;
- improving long-horizon manipulation with temporal context;
- using compact action-effect memory;
- using statefulness in an action policy;
- adding an adapter or LoRA to SmolVLA.

MTIL already claims full-trajectory history encoded by Mamba state and reports
results on LIBERO. MaIL already tests Mamba imitation learning on LIBERO.
AEM, MEM, MemoryVAM, and MemoryVLA-style work make the broad memory-for-VLA
claim crowded. Diff-Control also weakens any generic "stateful policy" claim,
although its closest axis is diffusion statefulness rather than Base-preserving
history residuals.

The only defensible novelty is narrow:

`frozen SmolVLA Base + deployment-observable recurrent history state +
identity-initialized bounded residual gate + prior-first MTIL proxy comparison`.

This is enough to continue only if Researcher A accepts that the paper claim is
not "memory helps VLAs" or "Mamba improves imitation learning." The claim must
be "history-state residual integration can improve a strong frozen VLA without
replacing its default actions."

## Closest Prior Boundary

MTIL remains the closest prior and must be policy 2 in the first serious
comparison.

If official MTIL code cannot be run under the local SmolVLA/LIBERO action
semantics, the proxy must be labeled:

`mtil_history_state_proxy`

not official MTIL reproduction.

The proxy must preserve MTIL's essential mechanism:

- recurrent state-space history encoding;
- action prediction conditioned on history state and current observation;
- demonstration-only imitation supervision;
- no Base-passthrough residual gate as the claimed mechanism.

Do not compare MHS only to Base, no-history ablation, or standard LoRA before
the MTIL proxy enters.

## Mechanism Plausibility Attack

The proposal's mechanism is plausible but not yet demonstrated. It depends on
all of the following being true:

1. There are current-frame ambiguous states in the selected LIBERO development
   tasks.
2. The ambiguity is recoverable from deployment-observable history rather than
   from privileged simulator state.
3. The frozen Base actually makes history-conditioned mistakes on those states.
4. Expert-minus-Base residuals are noncollapsed and safe to learn.
5. A recurrent history state changes the residual/gate beyond what a
   current-frame residual head can do.
6. The residual gate acts selectively rather than globally.

Any failure here is a data/design failure, not a closed-loop scientific kill.

## Data And Supervision Attack

The largest preimplementation risk is label construction. Researcher A must not
invent history ambiguity labels that are just another name for action L2.

Before training, require:

- exact discovery/validation/test split identity table;
- history-window count by task and phase;
- valid previous-frame/action availability;
- positive and negative ambiguity/usefulness counts;
- label entropy and majority baseline;
- current-frame-only baseline for the same target;
- duplicate-key and split-overlap checks;
- proof that labels do not use rewards, success, done, object poses, simulator
  state, future observations, or confirmatory identities;
- proof that inference needs only the robot's observed history and executed
  actions.

If labels collapse, if the current-frame baseline matches the history model, or
if history targets require privileged/future information, stop as
`DATA_OR_SUPERVISION_FAILURE`.

## Objective And Math Attack

The proposal uses Huber/BCE objectives and avoids deterministic-action KL,
which is acceptable. However:

- `L_hist` is under-specified until `z_target_t` is frozen;
- `m_t` is under-specified until the ambiguity/usefulness rule is frozen;
- `lambda_hist` and `lambda_gate` can dominate if label imbalance is high;
- a gate BCE can create global activation if positives are too broad;
- residual Huber can reduce to current-frame action regression unless the
  history path is necessary.

Required before nontrivial training:

- exact tensor shapes for observations, action chunks, history state, residuals,
  gate, and labels;
- exact formula for `m_t` and `z_target_t`;
- loss magnitude and gradient norm audit on a small batch;
- frozen coefficient search range with at most six configurations;
- no KL on deterministic 7D actions.

## Identity-Preservation Attack

MHS is high disruption risk if the gate activates broadly. Before any rollout:

- disk reload must reproduce Base passthrough within tolerance;
- Base action, MHS action, residual norm, gate value, changed dimensions, and
  activation context must be reported;
- translation, rotation, and gripper deltas must be capped separately;
- clean validation rows must preserve Base behavior;
- action validity after official postprocessing must be checked;
- frozen SmolVLA parameters must receive zero gradients.

If MHS changes nearly every action, classify as `DESIGN_FAILURE` or
`IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, not a scientific kill.

## Required First Serious Comparison

The first serious comparison remains:

1. `smolvla_base`
2. `mtil_history_state_proxy`
3. `mhs_full`
4. `mhs_no_history_state_ablation`
5. `standard_lora`

Policy 5 is justified because MHS trains on demonstrations and standard LoRA is
the simple reviewer-killer for generic adaptation. Do not add a broad internal
control suite before this comparison.

The no-history ablation must keep the residual/gate capacity, data, optimizer,
steps, action caps, and selection metric matched. It may remove only recurrent
history state, replacing it with current-frame inputs and Base chunk.

## Conditional Pass Requirements

Researcher A must accept these conditions:

1. Narrow the novelty claim to Base-preserving history-state residual
   integration.
2. Keep MTIL or a transparent MTIL proxy as policy 2.
3. Freeze the exact history-window construction before Stage 0.
4. Freeze `m_t` and `z_target_t` before training.
5. Prove labels and residual targets are noncollapsed across tasks.
6. Prove history predicts the target above trivial and current-frame-only
   baselines.
7. Require no-history ablation as the key ablation.
8. Require standard LoRA as the single simple reviewer-killer baseline.
9. Require exact Base passthrough at initialization and disk reload.
10. Require bounded action deltas and clean retention before rollout.
11. Forbid rewards, success flags, done flags, object poses, future
    observations, demonstration actions at inference, and confirmatory-test
    identities.
12. Treat failed label/headroom/history observability gates as
    `DATA_OR_SUPERVISION_FAILURE` or `NO_USABLE_HEADROOM`, not as a closed-loop
    scientific result.

## Decision

`REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

MHS may continue only under the narrow claim and conditions above. No
implementation, training, validation search, rollout, or confirmatory-test
access is authorized before Researcher A rebuttal, mathematical audit,
preregistration, and prototype protocol are frozen in order.
