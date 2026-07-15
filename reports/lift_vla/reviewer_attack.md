# LIFT-VLA Reviewer B Attack

Date: 2026-07-15 KST

Reviewed proposal: `reports/lift_vla/researcher_proposal.md`

Proposal hash:
`3D263AA6FF73B342523D85AD4854145AF4D79DE2B90C6119F417D37A8B08F55F`

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

LIFT is not rejected before implementation because the closest prior is
positive, the mechanism is falsifiable, the backbone can expose the required
vector fields, and the first comparison is unusually well matched. The current
proposal nevertheless overstates experimental readiness. A rebuttal must accept
the essential constraints below before mathematical audit or preregistration.

## Essential Paper Evidence

Only the following issues may block prototype GO.

### 1. Novelty Must Be Narrower Than The Proposal Title Suggests

Classifier-free guidance already combines conditional and unconditional model
predictions during a generative trajectory. Applying that equation to each
SmolVLA flow step is therefore not a new guidance algorithm. CAG also explicitly
derives its action rule from classifier-free guidance and evaluates flow-based
VLA families.

The only defensible novelty claim is:

> an empirical and mechanistic study of pathwise language guidance for a
> continuous VLA action flow, showing that it differs from and improves over
> CAG's final-action mixing under matched branches and flow evaluations.

Required constraint:

- do not claim a new CFG equation, new flow-matching method, new sampler, or
  general VLA guidance framework;
- treat exact or practical equivalence to CAG as a valid kill;
- search related work for any released per-denoising-step language guidance in
  VLA action flows before a paper novelty claim is allowed.

### 2. The Current Development Proxy Is Not Yet A Valid Counterfactual Benchmark

LIBERO-Goal contains a fixed scene with different goals, but swapping an
instruction string into another task's reset does not by itself make the new
goal feasible or change the simulator success predicate. Existing local
offline counterfactual reports explicitly mark their pairs `offline_proxy_only`
and include cross-scene pairs that are unsuitable here.

Required pre-rollout source gate:

- create a new LIFT-specific manifest from same-scene task pairs only;
- verify every target object, receptacle, fixture, and goal predicate exists in
  the selected initial state;
- instantiate or otherwise independently validate the counterfactual BDDL goal,
  not only the language string;
- persist a target-grounding and task-success scorer for the counterfactual goal;
- separate discovery, validation, and reserved confirmatory task/reset
  identities with zero overlap;
- reject cross-scene, absent-object, impossible, or original-goal-only pairs;
- label all local evidence as a custom development proxy unless official
  LIBERO-CF assets are obtained and checksum-verified.

If no scoreable feasible manifest exists, Stage 0 must return
`LIFT_DATA_OR_BENCHMARK_FAILURE`. Offline action separation cannot substitute
for closed-loop counterfactual headroom.

### 3. The CAG Proxy Must Be Defined In The Same Native Action Space

The proposal does not yet specify exactly where final-action mixing occurs.
Mixing independently postprocessed 7D actions would confound CAG with clipping,
normalization, unpadding, and the LIBERO bridge.

Required constraint:

- conditioned Base, empty-language branch, CAG, LIFT, and ablation must begin
  from the same sampled `x_0`;
- CAG must mix the two completed native SmolVLA flow outputs in
  `R^(B x H x D)` before action unpadding, normalization/postprocessing, and the
  7D LIBERO bridge;
- every arm then uses the same single postprocessing path;
- preserve the label `transparent_training_free_cag_proxy`; do not claim
  official equivalence;
- report whether the CAG paper specifies noise coupling; same-noise coupling is
  a local fairness choice if the paper is silent.

### 4. The Key Ablation Needs Matched Branch Evaluations

As written, `lift_last_step_only_ablation` uses one field evaluation for the
first nine steps and two only on the final step, whereas full LIFT uses two at
every step. This leaves repeated inference as an avoidable alternative
explanation and creates a latency mismatch.

Required constraint:

- the ablation must compute both conditioned and unconditioned fields at all ten
  steps;
- it must discard the unconditional field during steps `0,...,K-2` and apply
  guidance only at `K-1`;
- report both algorithmic field-evaluation count and measured latency;
- do not add another policy. This matched-compute implementation remains the
  same key ablation.

### 5. Mechanism Activation Must Be Practically, Not Merely Numerically, Distinct

Requiring any nonzero difference is too weak. Floating-point roundoff can make
LIFT differ from CAG without a meaningful path effect.

Before implementation, the mathematical audit must freeze:

- native-chunk and executed-first-action separation metrics;
- a practical separation threshold justified from discovery-only Base action
  scale and repeated same-noise numerical error;
- per-flow-step `||v_c-v_u||`, LIFT-versus-CAG chunk delta, and
  LIFT-versus-ablation chunk delta;
- target-aware action or grounding consequences, not action L2 alone;
- the exact outcome when separation is below threshold:
  `LIFT_DESIGN_FAILURE_PRACTICAL_EQUIVALENCE`.

### 6. Headroom Must Exist Beyond Both Base And CAG

Language sensitivity is not headroom. Large action differences can be wrong,
and high ordinary LIBERO-Goal success can leave little room for improvement.

Stage 0 must show on discovery/validation only:

- Base has a meaningful counterfactual grounding or success failure rate;
- final-action CAG leaves residual failure;
- at least one target-aware diagnostic indicates that stronger pathwise
  conditioning could address that residual;
- the intervention does not merely increase action magnitude or gripper
  switching.

Otherwise return `LIFT_NO_HEADROOM` without rollout.

### 7. Compute Feasibility Is An Empirical Gate

Two cached prefixes and two field evaluations per step may exceed the local
16GB budget or make rollout impractically slow.

Required constraint:

- perform a load-only then one-chunk peak-memory and latency smoke before broad
  offline decoding;
- preserve the scientific method if implementation is changed from concurrent
  to sequential branch evaluation;
- require numerical identity between feasible branch schedules within the
  frozen tolerance;
- classify an unresolved resource failure as `LIFT_COMPUTE_INFEASIBLE`, not as
  a scientific method kill.

## Strongest Alternative Explanation

The strongest relevant alternative explanation is `final_action_cag`, not
standard LoRA. Both Prior and Ours are inference-only, frozen, use the same
observations, and require two policy branches. Generic adaptation, extra data,
and extra trainable parameters cannot explain an LIFT gain.

Reviewer decision on controls:

- standard LoRA: `IRRELEVANT_EXPERIMENT`, omit;
- trained VA CAG branch: `USEFUL_DIAGNOSTIC_OR_FUTURE_SCALEUP`, not essential for
  the first prototype because it changes training and compute;
- dynamic guidance schedule: `IRRELEVANT_OR_NEW_VARIANT`, forbid in this cycle;
- fifth policy: not required;
- matched-compute last-step ablation: `ESSENTIAL_PAPER_EVIDENCE`.

## Useful Diagnostics

These are useful but may not independently block prototype GO:

- conditional and empty-language attention or token influence summaries;
- per-action-dimension guidance direction;
- queue-level effects across the full action chunk;
- task-category breakdown within the development proxy;
- concurrent versus sequential branch timing after numerical equivalence is
  established.

## Optional Supplementary Evidence

- a trained vision-action CAG branch;
- an additional guidance scale outside the frozen validation list;
- alternate null prompts;
- extra samplers or integration schedules;
- broad language paraphrase robustness.

These may not be added to rescue a failed confirmatory result.

## Reviewer Verdict

`REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

LIFT may proceed only if Researcher A accepts:

1. the narrow empirical VLA-flow novelty claim;
2. a scoreable feasible counterfactual benchmark gate;
3. native-flow-space same-noise CAG mixing;
4. matched-compute last-step ablation;
5. practical-equivalence thresholds and target-aware diagnostics;
6. headroom beyond Base and CAG;
7. one-chunk memory/latency feasibility before broad decoding;
8. no standard-LoRA or fifth-policy expansion.

