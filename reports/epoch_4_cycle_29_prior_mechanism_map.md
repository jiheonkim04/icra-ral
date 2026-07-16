# Epoch 4 Cycle 29 Prior Mechanism Map

Date: 2026-07-16 KST

Decision: `CYCLE_29_PRIOR_MECHANISM_MAP_COMPLETED`

Previous method: `TSC-VLA`

Previous decision: `TSC_STAGE_0_NO_USABLE_HEADROOM`

TSC remains closed. Its Stage 0 result is preserved unchanged in
`reports/tsc_vla/stage_0_result.json`; no threshold repair, proxy change,
task change, or TSC rescue is allowed.

## Primary-Source Anchors

### Coarse-to-Control

Primary source: https://arxiv.org/abs/2606.07107

Mechanism:

- action-token chain-of-thought;
- coarse planning tokens before executable action tokens;
- joint plan/execute residual-VQ action vocabulary;
- inference decodes only executable tokens.

Positive result:

- reports `97.9%` average LIBERO success;
- reports especially strong long-horizon performance, with `95.0` on LIBERO
  Long in the paper table;
- claims action-space planning gives more control-aligned guidance than text or
  visual CoT.

Local reproducibility:

- official code was not locally identified in this pass;
- a faithful local proxy is possible from existing LIBERO demonstrations by
  constructing coarse action-plan targets from future action chunks, then
  comparing against a direct-action policy under the same SmolVLA backbone.

Useful extension axis:

- adapt action-space planning to a continuous flow/chunk VLA without adding a
  discrete autoregressive action-token decoder;
- use the coarse plan as a bounded continuous constraint on a base-preserving
  action residual path.

### SUREFlow

Primary source: https://arxiv.org/abs/2607.10504

Official code: https://github.com/tanvirnwu/SUREFlow

Mechanism:

- state-space VLA backbone;
- memory-guided action decoder;
- joint velocity and uncertainty prediction;
- uncertainty-aware residual flow that selectively refines unreliable action
  dimensions.

Positive result:

- reports `92.5%` average LIBERO success;
- reports competitive LIBERO-PRO normalized success with only `179M`
  parameters;
- source code is publicly linked.

Local reproducibility:

- uncertainty and residual targets can be generated from existing legal LIBERO
  demonstrations;
- privileged future observations are not required at inference if future data is
  used only as a training label.

Useful extension axis:

- make uncertainty affect a base-preserving SmolVLA residual/refinement field
  rather than replacing the whole policy;
- compare early against a transparent SUREFlow-style uncertainty residual proxy.

### CoRE-VLA

Primary source: https://arxiv.org/abs/2607.03693

Mechanism:

- conditional routing of experts inside action generation;
- task-intent routing;
- action-side representation selection;
- expert utilization changes over rollout phases and tasks.

Positive result:

- reports `98.7%` average LIBERO success;
- reports strongest gains on long-horizon LIBERO;
- provides routing visualizations showing noncollapsed task/phase-dependent
  expert usage.

Local reproducibility:

- exact CoRE architecture may be heavy locally;
- a transparent proxy can route lightweight action-side residual experts using
  current observation, proprioception, instruction identity, and phase proxies
  derived from demonstrations.

Risk:

- prior campaign already killed broad task/instruction adapter routing;
- a valid candidate must route action-side temporal/dimensional computation,
  not merely choose a task LoRA.

## Design Constraint For Cycle 29

Cycle 29 must use:

- one genuinely new mechanism;
- LoRA only as implementation infrastructure;
- the closest external prior in the first serious comparison;
- discovery/validation/test separation;
- no confirmatory-test tuning;
- no rescue of the closed TSC result.
