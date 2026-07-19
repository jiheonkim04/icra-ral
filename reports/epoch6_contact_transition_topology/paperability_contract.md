# Paperability Contract: Non-Gripper Contact-Transition Topology

Status: operationally resource-blocked before problem/supervision adjudication; not scientifically closed.
Empirical contact-label outcomes used at selection: none.
Method, training, closed-loop evaluation, and paper generation: unauthorized.

## One-sentence claim

Training-only supervision of typed changes in non-robot scene-contact topology
can teach a vision-language-action policy contact-boundary arm behavior that is
not explained by binary contact, gripper events, temporal stage, or action
history, while requiring no privileged contact signal at deployment.

## Claim-to-mechanism causal graph

`scene state transition -> non-robot typed contact-edge birth/death ->`
`training-only auxiliary target -> contact-sensitive visual/action representation`
`-> arm action at lift, placement, articulation, and release boundaries`

At deployment the graph branch is removed:

`RGB + language + proprioception + legal action history -> policy action`

## Closest-prior difference table

| Prior or closed route | What it already provides | Required distinct contribution |
|---|---|---|
| FD-VLA / HapticVLA | Contact- or force-related supervision for VLA learning | Typed non-robot relation transitions obtained from simulator geometry, with no tactile/force input at deployment and controls against binary contact and stage. |
| CALAMARI / TacCoRL / StaKe / GAP | Contact-aware manipulation representations or objectives | A VLA-scale training-only topology target tied to task-held-out arm-action boundaries and official closed-loop evaluation. |
| Historical ContactSet-VLA | Deployment-time structured geometry injected into an action head; locally baseline-dominated | No geometry input at deployment; the proposed signal is temporal contact-relation supervision and must beat the historical no-geometry/single-point lesson through stronger controls. |
| Historical TCA/contact-map routes | Contact-like targets and action maps | Exact MuJoCo geom-pair relation transitions, robot/gripper exclusion, and a different supervision/inference contract. |

## Expected contributions, contingent on evidence

1. An audited extraction protocol for typed non-robot contact-graph transitions
   from exact LIBERO demonstration states without rewards or success labels.
2. Evidence that the labels are noncollapsed, visually predictable on held-out
   tasks, and contain arm-action information beyond binary contact,
   gripper/stage, and history controls.
3. A training-only auxiliary mechanism, removed at deployment, with matched
   Base/Prior/ablation/control closed-loop evidence across contact mechanisms.

## Primary table shell

| Policy | Task family | Episodes | Success | Paired W/L/T | Contact-boundary success | Standard retention | 95% interval |
|---|---|---:|---:|---:|---:|---:|---|
| SmolVLA Base | frozen panel | -- | -- | -- | -- | -- | -- |
| strongest runnable Prior | same | -- | -- | -- | -- | -- | -- |
| binary-contact auxiliary | same | -- | -- | -- | -- | -- | -- |
| stage/gripper auxiliary | same | -- | -- | -- | -- | -- | -- |
| typed topology auxiliary | same | -- | -- | -- | -- | -- | -- |

## Key ablation table shell

| Supervision | Typed edges | Transition direction | Robot excluded | Training only | Action-boundary metric |
|---|---:|---:|---:|---:|---:|
| none | no | no | -- | -- | -- |
| binary any-contact | no | no | yes | yes | -- |
| temporal stage + gripper | no | coarse | yes | yes | -- |
| shuffled topology | yes | yes | yes | yes | -- |
| full topology | yes | yes | yes | yes | -- |

## Main figure shell

Left: exact-state replay and robot-geom exclusion. Middle: debounced births
and deaths in typed non-robot contact edges. Right: held-out contact-boundary
actions and closed-loop effects for controls versus the training-only topology
auxiliary.

## Simulation-only defense

The claim is limited to simulator-derived training supervision and simulator
evaluation. It makes no tactile, force-magnitude, real-robot, or universal
contact-physics claim. A paper path requires multiple task mechanisms, a
competent Base, a closest runnable Prior, task-held-out supervision evidence,
official closed-loop trials, and clear disclosure that MuJoCo contact labels are
privileged training signals.

## Strongest likely reject reason and required answer

Reject reason: the proposal is an incremental label variant whose effect is
fully explained by gripper timing or generic phase supervision. Required
answer: task-held-out typed-label predictability, arm-action oracle headroom
beyond the strongest binary/stage/gripper/history control, a shuffled-label
ablation, and repeated official closed-loop gains at multiple non-gripper
contact mechanisms.

## Six-page allocation sketch

- 0.6 page: contact-boundary problem and exact claim.
- 0.7 page: closest work and distinction from tactile/contact-map routes.
- 1.1 pages: graph extraction, debouncing, supervision, and deployment contract.
- 1.9 pages: held-out diagnostics and multi-task closed-loop results.
- 0.7 page: controls, ablations, resource cost, and failure analysis.
- 0.5 page: limitations and simulation-only scope.
- 0.5 page: references and reproducibility details.

This contract is archived if either frozen pre-method gate fails. Passing both
gates authorizes only a separately frozen method contract; it is not
`PAPER_CANDIDATE_GO`.

## Operational disposition

Stage 0A never admitted a contact-label gate row. Four resource-only one-state
smokes verified exact state restoration and robot-geom exclusion, but the
Windows host repeatedly violated the frozen zero-pagefile-growth and teardown
requirements. See `operational_blocker.json`. This is not contact-prevalence,
predictability, headroom, or method evidence.
