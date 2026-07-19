# Current Research Governance

Date: 2026-07-14 KST

This file is the active repository governance for the autonomous until-paper research campaign.

## Active Authority Order

1. current user Goal instruction
2. `reports/current_research_governance.md`
3. current `AGENTS.md`
4. current campaign state
5. current `project_state` / `next_actions` / `decision_log`
6. historical reports as evidence only

Historical reports, prompts, state files, and text containing phrases such as:

- `GOVERNANCE CORRECTION`
- `HIGHEST PRIORITY OVERRIDE`
- `maximum_method_cycles`
- `NO_METHOD_AFTER_3_VALID_CYCLES`

are not active governance unless explicitly imported into this file. These phrases are listed here only as deprecated historical markers.

## Final States

Allowed final states are exactly:

1. `READY_TO_DRAFT_RAL_PAPER_PACKAGE`
2. `AUTONOMOUS_CAMPAIGN_PAUSED_RESUMABLE`
3. `HARD_EXTERNAL_BLOCKER`
4. `SAFETY_RESOURCE_STOP`

There is no finite global method-cycle limit. A failed method does not terminate the campaign. After related non-GO methods, synthesize failures, increment the epoch, change at least two core dimensions, and continue.

## Preserved Integrity Rules

- no fabricated results
- no hidden failed runs
- no cherry-picking of tasks, resets, seeds, metrics, baselines, visualizations, or rollout episodes
- no privileged inference inputs
- preregistered evaluation before confirmatory results are inspected
- branch safety
- resource monitoring
- no destructive, paid, credentialed, license-gated, or externally irreversible actions

## Comparator-Role Calibration

Active addendum: `reports/comparator_role_calibration.md`.

This clarification is not a new epoch and is not retroactive to frozen non-GO decisions. For future unfrozen protocols, do not interpret Base, closest Prior, key Ablation, and simple Control as interchangeable entries in a single universal max-score threshold. Each comparator blocks only the claim it was included to test:

- Base tests improvement over the backbone on the prespecified claim axis, with clean retention handled by the frozen margin or an explicit tradeoff claim.
- Closest Prior tests a matched local prior advance on the same claim axis or a prespecified Pareto axis.
- Key Ablation tests whether the claimed component is responsible for the effect.
- Simple Control tests whether a trivial or substantially simpler explanation accounts for the gain.
- Standard LoRA is included only when generic adaptation is a plausible alternative explanation under a matched setup; it remains implementation infrastructure unless the paper explicitly claims an adaptation algorithm.

If a future experiment is already running under a frozen universal beat-all scalar rule, finish it unchanged and report both `FROZEN_PROTOCOL_DECISION` and `CALIBRATED_SCIENTIFIC_INTERPRETATION`. Do not convert a frozen non-GO into GO post hoc.

## Epoch And Method Governance

Researcher A must freeze and hash a proposal before Reviewer B begins. Reviewer B independently searches closest primary sources, attacks novelty, identifies direct and simple killer baselines, and checks leakage or trivial equivalence. Governor C enforces this file, prevents premature termination, blocks underpowered permanent kills, and forces epoch pivots after related failures.

Reviewer B may reject before implementation only for:

- near-exact prior-art duplication across problem, representation, supervision, objective, policy component, inference, data, and claim;
- mathematical equivalence to a trivial baseline;
- essential unavailable resource.

Broad similarity is not sufficient.

Epoch 2 and later methods must avoid cosmetic variants of prior routes. A new epoch after related failures must change at least two core dimensions among core problem, representation, supervision, objective, policy generation, inference-time intervention, data source, and claim.

## Prototype Statistical Governance

Do not use a 5-percentage-point GO or kill gate with only 10 episodes per policy.

### Stage A

Purpose:

- validate mechanism;
- detect catastrophic harm;
- estimate direction.

Use approximately 10 paired episodes per policy.

Stage A may permanently kill only when one of the following holds:

1. implementation or data mechanism is invalid;
2. full method is at least 30 absolute percentage points below the strongest baseline or key ablation;
3. full method has `0 / 10` success while a paired baseline has at least `4 / 10`;
4. oracle or upper bound proves no usable headroom;
5. exact trivial equivalence is demonstrated.

Otherwise:

- positive result -> Stage B;
- small negative result -> Stage B;
- tie -> Stage B;
- one- or two-episode difference -> Stage B;
- noisy cross-task result -> Stage B.

### Stage B

Use at least:

- 40 paired episodes per key policy;
- identical task/reset identities;
- task-balanced allocation.

Required key policies:

1. unmodified backbone;
2. strongest direct baseline;
3. simple killer baseline;
4. key ablation;
5. full method.

Report successes/counts, per-task result, paired wins/losses/ties, paired bootstrap confidence interval, McNemar-style paired comparison when applicable, effect size, failure-rate reduction, mechanism activation, latency, and VRAM.

A method reaches `PROTOTYPE_GO` when:

- full beats the strongest baseline and ablation;
- absolute gain is at least 10 points at prototype scale, or paired evidence is consistently positive with meaningful failure-rate reduction;
- mechanism is active;
- no privileged inference signal is used;
- clean behavior is retained.

A method receives a permanent scientific kill only when:

- implementation is valid;
- mechanism acts;
- Stage B is complete;
- full is clearly worse, or the upper confidence bound excludes a useful improvement, or a baseline/ablation explains the method.

If Stage B remains unresolved, allow one preregistered expansion to at most 80 paired episodes per key policy. No third expansion is allowed.

## After Prototype GO

Do not stop at GO. Continue with larger primary-backbone confirmation, recent direct baselines, ablations, quantized OpenVLA-OFT INT4 integration when risk-assessed and locally permitted, same-backbone comparison, one claim-specific second condition or benchmark, clean retention, statistics, efficiency, figure/table-ready artifacts, reproducibility package, limitations, failure inventory, and a section-level paper outline.

Stop normally only when `READY_TO_DRAFT_RAL_PAPER_PACKAGE` is satisfied. Do not write the full manuscript.

## Post-CAVM Performance-Oriented Research Design Governance

This section applies to all method candidates after the valid CAVM-VLA adjudication. It is not retroactive to CAVM, PSE, or any earlier fixed-protocol result.

Future method cycles must be problem-first, novelty-aware, mechanism-explicit, mathematically justified, performance-oriented during development, and external-prior-anchored when possible. Rigorous research does not mean refusing to improve a method before confirmatory testing. It means keeping development evidence separate from the held-out confirmatory test and reporting failed development configurations honestly.

Maintain three evidence partitions:

- `DISCOVERY`: used to discover the problem, inspect failures, design supervision, test representations, and identify plausible mechanisms.
- `VALIDATION`: used to select architecture, loss weights, bounded hyperparameters, development variants, clean-retention behavior, and one final configuration.
- `CONFIRMATORY_TEST`: used once after method, configuration, baseline list, ablation, tasks, reset identities, metrics, and thresholds are frozen.

Confirmatory outcomes may not be used to retune the same method. A major redesign after confirmatory test becomes a new method cycle.

### Candidate Selection

Generate exactly three candidates. For each candidate identify:

1. the closest external prior;
2. the positive result that prior already demonstrates;
3. official code, checkpoint, or a reproducible mechanism;
4. the assumption or limitation being extended;
5. the minimal technical difference proposed by this campaign;
6. why that difference could improve the same claim axis.

Prefer a strong prior plus one technically meaningful extension plus a fair matched comparison over an unanchored local module. A less anchored candidate may proceed only with stronger problem evidence and a clearer falsifiable mechanism.

Every candidate must declare exactly one contribution type:

- `PRIOR_EXTENSION`
- `IMPLICIT_GAP_SOLUTION`
- `CROSS_PAPER_SYNTHESIS`
- `CROSS_DOMAIN_MECHANISM_TRANSFER`
- `NEW_DEPLOYMENT_PROBLEM`

Score exactly three candidates on:

- provisional novelty: `25%`
- importance of problem: `15%`
- strength of positive prior anchor: `20%`
- technical mechanism quality: `20%`
- data/supervision feasibility: `10%`
- decisive experiment feasibility: `10%`

The score selects what to test. It is not a prediction of empirical success.

For close literature, do not treat abstracts, contribution lists, limitations, future-work text, or discussion framing as authoritative. Reconstruct papers from equations, algorithms, architecture, training supervision, inference procedure, data generation, code when available, appendices, ablations, failure cases, benchmark conditions, omitted comparisons, and fixed experimental variables.

Each close-paper record must separate:

- `AUTHOR_STATED`: what the authors explicitly claim as contribution, novelty, limitation, or future work;
- `INDEPENDENTLY_INFERRED`: what follows from the actual method and evidence;
- `CROSS_PAPER_SYNTHESIZED`: what becomes visible only after comparing multiple papers.

For the closest literature, build a mechanism map covering observation/input, learned representation, supervision, objective, policy component changed, action-generation mechanism, inference-time intervention, assumed feedback, benchmark condition, primary metric, actual demonstrated causal link, and untested causal link.

### Pre-Experiment Audit

Before expensive training or rollout, perform a bounded development-only audit using discovery and validation data, not confirmatory test identities.

Check problem headroom: whether Base fails meaningfully on the claimed condition, the closest prior leaves meaningful residual failure, a plausible maximum gain exists, and a diagnostic oracle or privileged upper bound shows that the intervention target is useful. Oracles are diagnostics only, never inference methods.

Check label and contrast health: class balance, positive/negative counts, variance, task and phase coverage, censor/mask frequency, no all-zero or all-one targets, no accidental duplication, and no train/test overlap.

Check mechanism observability: whether the required latent or state can be inferred from deployment inputs, whether the target is predictable above a trivial baseline, and whether the signal survives across tasks.

Check policy disruption risk before rollout: action delta from Base, translation/rotation/gripper deltas, intervention frequency, action-bound validity, and clean validation behavior.

Do not proceed to large rollout when labels are collapsed, no headroom exists, the module is nonacting, the module catastrophically changes all actions, or the intended mechanism cannot be inferred from deployment inputs. Classify these as `DATA_FAILURE`, `NO_HEADROOM`, `IMPLEMENTATION_FAILURE`, or `DESIGN_FAILURE`, not as closed-loop scientific results.

### Bounded Validation Search

A method may receive bounded validation search before confirmatory testing. Predeclare the search budget. The default maximum is:

- no more than `6` total configurations;
- no more than `2` random seeds per configuration for lightweight training;
- no more than `2` architecture choices;
- no more than `3` values for one critical coefficient;
- no combinatorial grid over many variables.

Use discovery/validation only. Candidate factors may include residual or gate magnitude, loss coefficient, context horizon, latent dimension, intervention threshold, clean-retention coefficient, number of samples or views, and learning rate.

Select one configuration using a preregistered validation score that normally combines validation closed-loop success or the closest feasible proxy, clean retention, mechanism activation, action validity, and compute overhead. Do not select purely by offline action L2.

After selection, freeze the configuration, save its checkpoint, save all tried configurations and negative results, and do not tune it on the confirmatory test.

### Mathematical Mechanism Audit

Every selected method must include `reports/<method>/mathematical_mechanism_audit.md` before confirmatory testing. The audit must define variables and tensor shapes, mathematical formulation, representation learned, exact policy component affected, training objective, inference algorithm, data and supervision source, gradient path, expected behavioral effect, expected closed-loop consequence, closest mathematical alternative, simplest equivalent baseline, key ablation, known failure mode, and an identity-preserving integration audit.

For every proposed module and objective term, document exact variables, tensor shapes, mathematical formula, scale, units, gradient path, intended representation or action effect, simpler alternative, and required ablation. Before training, estimate term magnitudes and gradient norms on a small batch. When multiple objectives are used, normalize or justify their scale, inspect gradient norm ratios, inspect gradient conflict when relevant, choose coefficients on validation data only, and freeze coefficients before confirmatory test.

Do not add decorative mathematics. KL divergence may be used only when both arguments are valid probability distributions or justified density approximations, support and normalization are defined, the KL direction is justified, and the estimator is reliable. SmolVLA flow outputs are not automatically normalized action probability distributions. Do not compute KL directly between deterministic 7D action vectors.

Prefer identity-preserving integration when feasible: residual branches initialized to zero, gates initialized to base-policy passthrough, adapters initialized near identity, clean-retention regularizers, bounded interventions, or calibrated mixtures with default base behavior. A method that arbitrarily replaces strong pretrained actions receives a high disruption-risk penalty.

### Mechanism Smoke

Before closed-loop confirmatory evaluation require:

- checkpoint persists and disk reloads;
- expected parameters receive finite nonzero gradients;
- training and validation objectives behave sensibly;
- Ours differs from Base and ablation;
- the difference is bounded rather than globally destructive;
- action validity is preserved;
- clean validation behavior is retained;
- intended mechanism activates in relevant states rather than everywhere;
- no privileged inference input;
- no hidden use of test identities.

For a residual or adapter method, report Base action, Ours action, residual norm, gate value, dimensions changed, and activation context. For representation methods, report representation metric, action-distribution consequence, and clean-versus-shift behavior.

### Prior-First Prototype

The first serious comparison for each future method should normally use exactly five policies:

1. Base;
2. closest external prior or faithful transparent proxy;
3. Ours;
4. key ablation;
5. one strongest simple reviewer-killer baseline.

No more than one mandatory simple killer baseline is required at the initial prototype stage. Additional internal controls are allowed only when they correspond to a concrete reviewer objection, test a genuinely different trivial explanation, could change the scientific decision, and are cheaper than proceeding directly to the prior comparison.

Use a matched paired manifest.

Stage A uses approximately `10` paired episodes per policy to detect catastrophic harm, obvious prior dominance, mechanism invalidity, no headroom, or exact trivial equivalence. Small differences advance to Stage B.

Stage B uses at least `40` paired episodes per key policy and reports paired wins/losses/ties, bootstrap confidence interval, effect size, failure-rate reduction, per-task breakdown, mechanism activation, clean retention, and efficiency. Allow one expansion to `80` only when Stage B is genuinely unresolved.

### Paper-Candidate Decision

A method becomes a serious paper candidate when Base plus Ours beats Base, Ours beats the closest external prior on the matched claim axis, Ours beats the key ablation, one strongest simple explanation does not account for the gain, novelty remains defensible, clean behavior is retained, and mechanism evidence supports the intended explanation.

After paper-candidate status, immediately verify Quantized OpenVLA-OFT INT4, add one claim-specific second condition or benchmark, add directly relevant recent baselines when feasible, measure compute and latency, and prepare figure/table-ready evidence.

The final comparison must include SmolVLA versus SmolVLA plus Ours and Quantized OpenVLA-OFT INT4 versus Quantized OpenVLA-OFT INT4 plus Ours.

Before any future terminal decision, `scripts/check_current_research_governance.py` must pass.

## Post-RAC Honest Positive-Result Governance

This section applies only after the closed RAC-VLA Stage B result. It is not retroactive to RAC or any earlier frozen protocol.

The campaign objective for subsequent methods is:

`MAXIMIZE_THE_PROBABILITY_OF_AN_HONEST_PAPER_WORTHY_POSITIVE_RESULT`

This means choosing stronger ideas, anchoring them to real positive prior evidence, verifying usable data and headroom, designing mathematically functional mechanisms, allowing bounded development and validation tuning, preserving the pretrained policy, and then running a frozen fair confirmatory experiment. It does not permit held-out test tuning, cherry-picking, threshold changes after results, or rescue of a valid kill.

### Evidence Partitions

Every new method must create and persist strictly separated identities for:

1. `DISCOVERY PARTITION`: failure discovery, trajectory/video inspection, hypothesis formation, label construction, method design, signal existence checks, and diagnostic headroom.
2. `DEVELOPMENT / VALIDATION PARTITION`: architecture choice, one or two critical coefficients, context length, residual/gate scale, training duration, final configuration selection, mechanism activation, and clean retention.
3. `CONFIRMATORY TEST PARTITION`: one-shot held-out evaluation after architecture, hyperparameters, checkpoint-selection rule, task/reset manifest, baselines, ablations, metrics, and decision thresholds are frozen.

Confirmatory-test outcomes may not be used to retune the same method. A major redesign after test is a new method cycle. Every split must prove zero identity overlap.

### Candidate Quality Gate

Generate exactly three candidates. Do not select a method merely because it is a small MLP, a lightweight residual, a simple frozen-policy attachment, a convenient image transform, an easy threshold, or immediately runnable on one task.

For each candidate reconstruct:

- exact robotics or VLA problem;
- closest external prior;
- positive result already demonstrated by that prior;
- actual prior mechanism, not only contribution wording;
- remaining assumption or limitation;
- proposed representation, objective, supervision, or policy mechanism;
- falsifiable mechanism-to-success hypothesis;
- required data and whether it exists;
- diagnostic headroom;
- policy-disruption risk;
- fair external-prior comparison path;
- decisive local experiment;
- second-backbone integration path;
- claim-specific second condition.

Score exactly three candidates on:

- provisional novelty: `25%`
- importance of the problem: `15%`
- strength of positive external-prior anchor: `20%`
- technical mechanism quality: `20%`
- data and supervision feasibility: `10%`
- decisive experiment feasibility: `10%`

Unknown future performance is not a rejection reason. Pre-implementation rejection is allowed only for near-exact prior-art duplication, mathematically invalid formulation, obvious equivalence to a trivial method, missing essential data or supervision, unavailable essential hardware, or no falsifiable experimental path.

### Positive-Prior Anchor

Future methods should preferably use:

strong external prior + clearly identified unresolved assumption + one technically meaningful extension + matched Base/Prior/Ours comparison.

Before implementation, identify closest external prior, strongest recent method on the same claim axis, official code or checkpoint status, exact positive result, backbone and benchmark compatibility, observation and action semantics, inference budget, privileged inputs, and what must be faithfully reproduced locally.

Do not treat published numbers from incompatible protocols as a direct baseline. When official code cannot be run, implement a faithful transparent proxy only if the essential mechanism is preserved; list omitted components; never call a proxy an official reproduction.

A less anchored method may proceed only when problem evidence is strong, diagnostic headroom is clear, supervision is available, and the mechanism is deeper than another local add-on.

### Problem And Headroom Audit

Before expensive implementation, training, or rollout, test usable headroom on discovery and validation identities only.

Check whether Base fails meaningfully, the closest prior leaves residual failure, the condition is neither saturated nor at an unusable floor, and the failure appears across more than one task or controlled condition.

Use a diagnostic oracle, privileged training-only upper bound, teacher policy, known inverse dynamics, successful trajectory evidence, intervention upper bound, or representation probe only when it measures the proposed intervention target.

If Base, prior, and oracle all fail, classify the condition as `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE`. If a stronger working backbone fully solves the claimed problem, do not frame it as a general VLA failure without a new deployment condition or stronger claim.

### Data And Supervision Health Gate

Before training, report records, positive and negative counts, task coverage, phase coverage, label variance, mask or intervention frequency, duplicate count, train/validation/test overlap, trivial-majority accuracy, and predictability from deployment-time inputs.

Reject or redesign before rollout when labels collapse, full and ablation receive effectively identical targets, supervision depends on inaccessible inference-time information, the signal is not observable from RGB/proprioception/instruction, coverage collapses, required successful counterexamples do not exist, or data construction does not represent the claimed mechanism. Classify this as `DATA_OR_SUPERVISION_FAILURE`, not as a scientific method kill.

### Identity-Preserving Integration

A strong pretrained VLA should be preserved by default. Prefer zero-initialized residuals, identity adapters, base-passthrough gates, bounded residual magnitude, conservative latent updates, clean-retention objectives, and context-dependent interventions initialized to zero so that Ours is initially close to Base.

Before rollout quantify action delta from Base, translation/rotation/gripper deltas, intervention frequency, output validity, action-bound violations, clean validation success, and full-versus-ablation difference.

Do not proceed to confirmatory rollout if the component strongly modifies nearly every action, clean validation collapses, gripper behavior changes unintentionally, the action distribution leaves the valid pretrained region, or the method acts globally when it should act selectively.

### Bounded Development Search

Do not send the first arbitrary implementation directly to confirmatory rollout. Every selected method may receive one bounded development search using discovery and validation data only.

Default maximum:

- at most `6` total configurations;
- at most `2` architecture variants;
- at most `3` values of one critical coefficient;
- at most `2` lightweight training seeds per selected configuration;
- no broad combinatorial sweep;
- no confirmatory-test use.

Before search, write configurations, selection metric, compute budget, and tie-breaking rule. Prefer a validation score that combines closed-loop success when affordable, clean retention, mechanism activation, action validity, and compute overhead. Do not select by offline action L2 alone. Save every attempted result, final checkpoint, and negative result.

### Mathematical Objective Engineering

For every objective term provide exact variables and tensor shapes, mathematical definition, units and scale, training stage, gradient path, expected representation change, expected action-distribution change, expected closed-loop consequence, simplest substitute, and required ablation.

Before full training, measure loss magnitudes and gradient norms, detect one loss dominating others, detect gradient conflict when relevant, and validate coefficient scale.

Do not add ornamental KL, contrastive loss, entropy loss, mutual information, causal terminology, or arbitrary weighted objectives. KL is permitted only when both arguments are valid probability distributions or justified density approximations with defined support, normalization, estimator, gradient destination, and direction. Do not compute KL directly between deterministic 7D action vectors or assume SmolVLA flow vectors are normalized distributions.

### Mechanism Smoke

A method may enter confirmatory rollout only when applicable checks pass:

- intended parameters receive finite nonzero gradients;
- loss and validation behavior are sensible;
- checkpoint persists and disk reloads;
- evaluation loads the intended checkpoint;
- mechanism changes the intended representation or action;
- full differs from key ablation;
- change is bounded;
- clean validation behavior is retained;
- action validity is preserved;
- no privileged inference signal;
- no identity leakage;
- no collapsed gate, mask, label, memory, or intervention.

For learned modules report parameter count, training loss, validation loss, gradient norms, activation frequency, output variance, action delta, and clean retention.

### First Serious Experiment

The default first paper-oriented comparison uses exactly five policies:

1. Base
2. closest external prior or faithful transparent proxy
3. Ours
4. key ablation
5. one strongest simple reviewer-killer baseline

Do not run a large suite of internal controls before the external prior. Additional controls require a concrete alternative explanation, decision relevance, and lower cost than proceeding to final comparison.

Use identical paired task/reset manifest, matched inference budget where possible, same observation and action semantics, comparable training data, and transparent compute differences.

### Staged Closed-Loop Evaluation

Stage A is a catastrophic and directional screen using approximately `10` paired episodes per policy. It may permanently kill only when full has `0 / 10` while a paired baseline has at least `4 / 10`, full is at least `30` absolute points below baseline or ablation, mechanism is valid and clearly harmful, no diagnostic headroom exists, implementation or supervision is invalid, or trivial equivalence is proven. Small differences, ties, and one- or two-episode gaps advance to Stage B.

Stage B is the confirmatory paired prototype using at least `40` paired episodes per key policy. Report successes/counts, task-balanced success, paired wins/losses/ties, paired bootstrap confidence interval, effect size, relative failure-rate reduction, per-task result, clean retention, mechanism activation, latency, and VRAM.

Allow one expansion to `80` only when full is not clearly inferior, the confidence interval remains unresolved, and useful improvement has not been excluded. No second expansion.

### Result Classification

Classify results precisely:

- `PROTOTYPE_GO`: Ours beats Base, closest prior, key ablation, and strongest simple explanation; clean behavior is retained; mechanism is active; effect is meaningful.
- `UNDERPOWERED_ONE_EXPANSION_ALLOWED`: evidence is directionally positive or unresolved.
- `GENUINE_METHOD_KILL`: implementation and data are valid, mechanism acts, Stage B or valid catastrophic Stage A is complete, and full clearly fails against prior, baseline, or ablation.
- `DATA_OR_SUPERVISION_FAILURE`
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`
- `CONDITION_TOO_SEVERE_OR_NO_HEADROOM`
- `SIMPLE_BASELINE_EXPLAINS_METHOD`
- `KEY_COMPONENT_NOT_USEFUL`

Only `PROTOTYPE_GO`, `GENUINE_METHOD_KILL`, `SIMPLE_BASELINE_EXPLAINS_METHOD`, and `KEY_COMPONENT_NOT_USEFUL` are scientific formulation decisions. Data, implementation, optimization, and no-headroom failures inform the next design rather than proving a research family impossible.

### After Prototype GO

Do not stop at SmolVLA GO. Immediately proceed to larger primary-backbone confirmation, strongest recent directly relevant baseline, Quantized OpenVLA-OFT INT4 integration, same-backbone OpenVLA-OFT INT4 versus OpenVLA-OFT INT4 plus Ours comparison, one claim-specific second condition or benchmark, clean retention, statistical confirmation, latency, VRAM, parameters, training cost, figure/table-ready artifacts, and novelty re-adjudication against latest papers.

Final paper comparison must include SmolVLA versus SmolVLA plus Ours and Quantized OpenVLA-OFT INT4 versus Quantized OpenVLA-OFT INT4 plus Ours.

### Long-Running Execution

For future long-running WSL experiments, use detached durable execution, save PID, heartbeat, logs, partial result, exact resume command, and resume only missing evaluation keys after interruption.

## False-Negative Safeguard For Pre-Rollout Decisions

Before any Stage 0 or pre-rollout permanent kill, Reviewer B must classify the
evidence as exactly one of:

- `FATAL_PREIMPLEMENTATION`: near-exact prior duplication, mathematically
  invalid objective, exact trivial equivalence, essential unavailable
  resource, or non-falsifiable mechanism.
- `ROBUST_EMPIRICAL_DESIGN_FAILURE`: valid data and implementation, adequate
  independent records, decisive headroom diagnostics, and uncertainty that
  excludes a preregistered practically useful candidate advantage.
- `UNDERPOWERED_OR_UNRESOLVED`: small point estimate, unavailable or wide
  interval, normalization sensitivity, large subgroup variance, weak record
  independence, or practical tie.
- `IMPLEMENTATION_OR_DATA_FAILURE`: collapsed labels, missing contrast,
  nonacting gradients, invalid construction, wrong checkpoint, or integration
  defect.

Only `FATAL_PREIMPLEMENTATION` and `ROBUST_EMPIRICAL_DESIGN_FAILURE` may produce
a permanent pre-rollout kill. `UNDERPOWERED_OR_UNRESOLVED` receives exactly one
cheap preregistered decisive check. `IMPLEMENTATION_OR_DATA_FAILURE` is not a
scientific method kill.

Before a kill, Reviewer B must record the strongest fair interpretation of the
frozen proposal, the narrowest honest publishable claim, false-positive risk,
false-negative risk, confidence, record count, independence analysis,
variance or bootstrap interval, practical-effect threshold, normalization
sensitivity, and exact evidence required for a permanent kill. Risk alone is
not proof of failure, and unknown performance remains an empirical question.

This safeguard applies to new empirical evidence. It does not reopen a method
review that has already passed or retroactively rescue a valid frozen result.

## Post-COVI LoRA And Minimum-Sufficient Design Governance

Effective after the frozen COVI Stage 0 closed on `2026-07-15`, this section
overrides earlier generic requirements that every future first experiment use
exactly five policies or automatically include standard LoRA. It does not
change COVI's frozen method, comparator list, result, or adjudication.

### Scientific Method Versus Low-Compute Parameterization

Every future proposal must document two separate layers:

1. `SCIENTIFIC_METHOD`: the new representation, objective, supervision,
   decomposition, action generation, temporal reasoning, control mechanism,
   or closed-loop intervention.
2. `LOW_COMPUTE_PARAMETERIZATION`: the locally feasible realization, such as
   LoRA, QLoRA, a lightweight adapter, a small head, frozen-feature training,
   or cached-feature training.

LoRA and QLoRA are compute-enabling implementation mechanisms, not default
scientific contributions. Removing the words LoRA and QLoRA from a method
description must not remove its novelty. Do not frame a method as a PEFT paper
unless adaptation efficiency is itself the preregistered research problem.

### Conditional Standard-LoRA Control

Standard LoRA or QLoRA is required only when generic adaptation is a plausible
alternative explanation, normally because Ours updates policy weights, uses
the same PEFT scaffold, receives extra training data, or uses a new supervision
signal or objective. In that case it is a supporting diagnostic with matched
checkpoint, data, split, steps, optimizer, batch/accumulation, rank, target
modules, augmentation, and checkpoint-selection rule where technically valid.

Standard LoRA may be omitted when Ours is inference-only, the backbone remains
frozen, generic adaptation does not test the claim, or the prior and ablation
already isolate the mechanism. Every omission must include one sentence
stating why standard LoRA does not test the claimed mechanism.

When both Prior and Ours require training, hold the locally feasible
LoRA/QLoRA or adapter scaffold constant where technically valid so the
scientific comparison remains Prior mechanism versus Ours mechanism. Never
label a PEFT proxy as an official reproduction when the prior fundamentally
requires incompatible full fine-tuning; disclose the mismatch or select a
different fair prior.

### Minimum-Sufficient Method And First Experiment

Prefer one core mechanism, one primary objective, at most one necessary
auxiliary term, and one key ablation. More components require a central
mechanism justification, observable training or inference effect, isolatable
contribution, and compute value. Do not add modules, gates, memories, losses,
consistency terms, or divergences merely to increase apparent depth.

The default first paper-oriented comparison for future unfrozen methods is:

1. Base
2. closest external Prior or faithful transparent proxy
3. Ours
4. key ablation
5. one additional control only when it tests the strongest plausible
   alternative explanation

The fifth policy is conditional, not mandatory. It may be standard LoRA,
a simple inference baseline, a data-matched ordinary objective, or absent.
Every baseline must have a distinct scientific question in a baseline
rationale table; remove a baseline that has none. Earlier template inclusion
alone is not a scientific reason.

### Capacity And Identity Classification

Before confirmatory rollout, verify gradients, small-subset fit, distinction
from Base and ablation, bounded action change, clean retention, disk reload,
legal inference sources, and adapter targets aligned with the mechanism. When
the locally feasible adapter cannot express the unchanged scientific method,
classify `LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT`, not a scientific method
kill. Allow one bounded capacity adjustment only when the bottleneck is
demonstrated, the scientific method does not change, and confirmatory-test
identities remain untouched. Do not run broad LoRA rank or target-module
sweeps.

### Primary Paper Gate

The primary effects remain same-backbone comparisons:

- SmolVLA versus SmolVLA plus Ours;
- after prototype GO, Quantized OpenVLA-OFT INT4 versus Quantized
  OpenVLA-OFT INT4 plus Ours.

A serious paper candidate must also beat the closest prior and key ablation,
survive the strongest relevant alternative explanation, retain clean
behavior, preserve novelty, and support the intended mechanism. Standard LoRA
superiority is required only when standard LoRA is that relevant explanation.

## Active Wrist-Dropout Simulation-Only RA-L Calibration

Effective `2026-07-19`, this section governs only the active final direction
`ACTION-CONSISTENT MISSING-VIEW DISTILLATION` and overrides any earlier rule
that makes a second backbone, camera-only validation, or a physical experiment
a universal prerequisite for `PAPER_CANDIDATE_GO`. It does not modify the
method, Stage 0 mechanism, losses, discovery/validation split, frozen Stage 0
thresholds, archived decisions, or the prohibition on confirmatory-test
tuning.

The allowed narrow paper claim is:

`ROBUST VLA MANIPULATION UNDER SIMULATED WRIST-CAMERA FAILURES`

Physical robot manipulation remains prohibited. Do not claim real-world robot
robustness, physical manipulation success, real sensor reliability, hardware
safety, sim-to-real transfer, or real-world deployment readiness. Evidence
must use official policy-controlled closed-loop simulator execution and
official task success.

### Stage 0 Role

Stage 0 is a validity and mechanism-activation gate. It must establish real
teacher/student execution, optimization health, persistence, exact clean
bypass, legal outputs, directional teacher-agreement improvement, and a
practical full-versus-ablation effect above numerical noise. It need not prove
final closed-loop superiority. A small but consistent noncatastrophic offline
effect with unresolved uncertainty is not by itself a permanent method kill;
final mechanism support comes from Stage A/B closed-loop evidence under the
false-negative safeguard.

### Stage A Sampling

Stage A begins with at least three tasks and exactly three held-out identities
per task, matched across frozen X-VLA, the `MECHANISM_FAITHFUL_RL4IL_LOCAL_PORT`,
Ours, the no-reconstruction ablation, and the generic wrist-dropout adaptation
control. Expand once to five identities per task only when the initial result
is positive but uncertain, mixed across tasks, or near the frozen decision
boundary. Do not spend the larger budget automatically.

### Stage B Breadth And Adaptive Size

A simulation-only paper-candidate Stage B uses at least four tasks, preferably
across at least three compatible LIBERO categories, with held-out matched
identities and at least three failures confined to the wrist-camera claim:
persistent blackout, intermittent frame loss, and frozen/repeated frame or a
preregistered partial-occlusion equivalent.

Begin with at least 60 paired failure-condition episode rows per key policy.
Report task-balanced results and paired uncertainty. Stop when the frozen
performance or noninferiority claim is resolved. If its interval overlaps the
claim boundary, expand exactly once to the preregistered maximum of 80 or 100
rows and stop regardless of significance. Freeze the exact row manifest and
maximum before Stage B outcomes; repeated significance-seeking expansion is
prohibited.

### RL4IL Performance And Pareto Paths

Against RL4IL, accept one of three preregistered paths:

1. clearly higher dropout success, clean retention, and mechanism evidence;
2. statistically comparable or noninferior success plus one large structural
   deployment advantage, such as eliminating the inference retrieval library
   or a large latency/memory/artifact reduction;
3. statistically comparable or noninferior success plus two moderate useful
   deployment advantages.

Before Stage A/B outcomes, freeze the success noninferiority and clean-retention
margins, latency and memory protocols, retrieval-storage accounting, and the
major/moderate practical thresholds. Do not select favorable dimensions after
outcomes, and do not treat a trivial cost difference as a contribution.

### Optional Strengthening Evidence

A second backbone is optional strong generalization evidence attempted only
after positive X-VLA Stage B when a matched multi-view interface, action
normalization, unchanged mechanism, compute, and local execution are valid.
Incompatibility or resource limits are documented infrastructure limitations,
not method failures and not automatic paper blockers.

`CAMERA-ONLY REAL-IMAGE ACTION-STABILITY VALIDATION` is optional supplementary
evidence. Its absence does not block Stage B, `PAPER_CANDIDATE_GO`, paper-package
generation, or an RA-L submission recommendation. It is never called a robot
experiment or physical manipulation validation.

### Simulation-Only Paper-Candidate Gate

`PAPER_CANDIDATE_GO` is permitted without a second backbone or camera-only
experiment when all of the following are established:

1. the simulated wrist-camera failure problem is repeatedly verified;
2. the method survives the strengthened mechanism-level overlap audit;
3. official closed-loop LIBERO success improves over frozen X-VLA;
4. Ours establishes a preregistered performance or Pareto advantage over
   RL4IL;
5. the key ablation supports the claimed mechanism;
6. generic dropout adaptation does not explain the full gain;
7. clean performance remains within its frozen retention margin;
8. results are consistent across multiple tasks and wrist-failure conditions;
9. paired uncertainty supports the primary claim;
10. inference uses no privileged information;
11. latency, VRAM, RAM, retrieval storage, checkpoint size, and trainable
    parameters are reported; and
12. manifests, checkpoints, telemetry, and commands are reproducible.

A single-backbone paper candidate therefore carries a stronger task,
condition, ablation, and statistical burden. If it passes, position it as a
method and benchmark study of VLA manipulation under simulated wrist-camera
failures, and state the absence of physical validation, sim-to-real evidence,
and real sensor-failure recovery as explicit limitations.

## Active Final-Direction Execution Outcome

Effective `2026-07-19`, the frozen final direction has the exact Stage 0
decision `STAGE0_IMPLEMENTATION_OR_RESOURCE_FAILURE` and paper-level decision
`IMPLEMENTATION_DATA_OR_RESOURCE_FAILURE`.

The sole permitted implementation repair was consumed by the official-reader
import initialization boundary. The unchanged final noise-calibration rerun
then materialized all 12 fixed discovery rows, confirming that repair at its
declared boundary, but failed at `torch.cuda.reset_peak_memory_stats(device)`
with `RuntimeError: Invalid device argument` before model load, teacher/student
forward, optimizer execution, checkpointing, or confirmatory access. The
distinct device-runtime defect may not be repaired under the current `1 / 1`
budget.

Therefore:

- no Stage 0 GO or scientific mechanism decision exists;
- no numerical-noise floor or practical threshold was frozen;
- Stage A/B, closed-loop Ours evaluation, and paper-candidate progression are
  not authorized;
- no mechanism support, mechanism rejection, generic-adaptation explanation,
  clean-retention result, or robust empirical design failure may be claimed;
- no current-method rerun, renamed v2, threshold relaxation, or replacement
  local wrist-dropout candidate is authorized; and
- the campaign is `AUTONOMOUS_CAMPAIGN_PAUSED_RESUMABLE`, with resumption
  requiring explicit user authority for a second narrow repair or a strategic
  pivot outside the current candidate scope.

Authoritative outcome records are
`reports/action_consistent_missing_view_distillation_stage0_result.json` and
`reports/action_consistent_missing_view_distillation_exact_scientific_status.json`.

## Exceptional CUDA Telemetry Resumption

Effective `2026-07-19`, the user explicitly authorizes exactly one
infrastructure-only exception for the previously observed
`torch.cuda.reset_peak_memory_stats(device)` / `Invalid device argument`
failure. Its classification is `EXCEPTIONAL_TELEMETRY_DEVICE_REPAIR`, never
`METHOD_REPAIR` or `SCIENTIFIC_REDESIGN`. This authority does not reset the
general `1 / 1` repair budget and changes no scientific contract.

Diagnosis in the actual WSL environment showed that the pinned PyTorch build
left CUDA uninitialized after availability queries and `empty_cache`; the
original `torch.device("cuda:0")` reset then reproduced the failure. Obtaining
the integer index through `torch.cuda.current_device()` initialized CUDA, and
all requested reset forms succeeded. The minimal patch now uses that same
validated index consistently for CUDA telemetry. A telemetry-only RTX 5080
smoke passed without X-VLA, discovery/validation/confirmatory access, CPU
fallback, or optimizer execution.

The historical Stage 0 and paper-level implementation-failure labels remain
preserved as pre-resumption evidence. The same frozen numerical-noise stage
subsequently completed validly, and its normalization denominators, practical
floors, and smoothness envelopes were frozen before any optimizer step. The
actual-path microbatch preflight then found all candidates safe and selected
microbatch `8` with accumulation `1`; its four throwaway steps consumed zero
Stage 0 optimizer budget. Frozen Stage 0 implementation and training are now
authorized. If another unrelated implementation defect prevents model execution, stop without
claiming mechanism failure. All existing
method, data, split, identity, comparator, threshold, optimizer, budget,
microbatch, evaluation, and no-confirmatory-access contracts remain immutable.

Authoritative exception records are
`reports/action_consistent_missing_view_distillation_cuda_device_diagnosis_result.json`
and
`reports/action_consistent_missing_view_distillation_telemetry_device_repair_result.json`.

## Final Action-Consistent Stage 0 Outcome

Effective `2026-07-19`, the valid resumed Stage 0 decision is
`STAGE0_MECHANISM_NOT_SUPPORTED`; the paper-level decision is
`KEY_COMPONENT_NOT_SUPPORTED`. This final scientific result supersedes the
pre-resumption implementation-failure label as the current status while
preserving that earlier label as historical audit evidence.

Execution was valid: all four 434,816-parameter arms completed 128 optimizer
steps and 1,024 exposures, with real frozen X-VLA teacher/student forwards,
finite nonzero gradients, changed weights, step-64/128 checkpoints, exact disk
reload, unchanged X-VLA parameters, exact clean bypass, legal and smooth
actions, no swap growth, and zero exceptions. No confirmatory outcome or
physical manipulation was accessed.

The cross-view auxiliary was learned: Full reconstruction MSE was
`0.9079925567` versus `0.9941876630` for no reconstruction, ratio
`0.9133009697`, and Full won all three tasks. Full also passed the directional
Base gate. However, no Full-versus-no-reconstruction action-semantic metric
passed the frozen practical effect rule. Relative improvements were `0.0125%`
translation, `-0.0747%` rotation, `0.2282%` raw gripper margin, and `0.0844%`
action hidden; Full also had `194` discrete gripper disagreements versus `193`
for no reconstruction. The key action-level contribution of reconstruction is
therefore not supported.

This result is not positive-uncertain or boundary-near under the frozen rules:
some statistically positive effects are far below the 5% practical threshold,
rotation is slightly negative, and the exact no-added-gripper-disagreement
rule fails. Do not access the fixed confirmation reserve or Stage A/B.

The method is archived as not Stage-A-ready. Close the wrist-dropout
method-development axis. Do not create a renamed v2, relax thresholds, rerun
the formulation, generate another local wrist-dropout candidate, reopen broad
prior search, or resurrect RIFA/CVLR. Only an explicit user-authorized
strategic pivot outside this axis can resume empirical method development.

Authoritative current records are
`reports/action_consistent_missing_view_distillation_resumed_stage0_result.json`,
`reports/action_consistent_missing_view_distillation_archive_decision.json`,
and
`reports/action_consistent_missing_view_distillation_exact_scientific_status.json`.

## Strategic Pivot Epoch 1 Authority

Effective `2026-07-19`, the user explicitly authorized a strategic pivot outside
the closed wrist-dropout method-development axis through the paper-completion
autonomy steer whose SHA-256 is
`FCFDE6371541CDB635F1B2D660A80379D227F2B0D32C14B38BBDD9BE7FFD68CC`.
This authority does not reopen or reinterpret any wrist-dropout result.

`PIVOT_EPOCH_1` evaluated exactly three research theses and returned
`PIVOT_SELECTED`. The selected thesis is action-chunk reactivity under
asynchronous inference delay, anchored to the official A2C2 SmolVLA/LIBERO
implementation. The previous EAC scheduler, TL-ChunkRepair, and phase-retiming
formulations remain closed; the selected thesis tests an injected inference-delay
condition and a trained current-observation action-correction prior that those
routes did not test.

The next authorized stage is official-prior-first problem verification only.
Ours may not be designed or executed unless the result is
`VERIFIED_PRIOR_RESIDUAL`. The authoritative selection records are
`reports/strategic_pivot_epoch1_selection_result.json` and
`reports/strategic_pivot_epoch1_selection_result.md`.

## A2C2 Official-Prior-First Protocol Authority

Effective `2026-07-19`, the selected asynchronous-delay thesis has a frozen
problem-verification protocol. The external Prior is always labelled
`MECHANISM_FAITHFUL_A2C2_LOCAL_PORT`, never an official reproduction. Its
official source is pinned to `k1000dai/a2c2-libero` commit
`54dd088302a0ef3f50c4add3ec927ab94d76a406`.

The matched evaluation panel is LIBERO Spatial task ids `0`, `4`, and `8`,
official init-state ids `0..4`, with 15 episodes per arm. Base competence uses
`e=10,d=0`; the claim-specific Base and Prior condition uses `e=40,d=10`.
The queue semantics, 220-step cap, base-noise schedule, training identities,
40,000-step Prior-module budget, numerical gates, and seven-decision mapping
are frozen before any model preflight or empirical result.

Execution order is `SETUP_PREFLIGHT`, `CACHED_FEATURE_PROBE`,
`PRIOR_MODULE_TRAINING`, matched Base `VLA_CLOSED_LOOP_ROLLOUT`, trained-Prior
`VLA_CLOSED_LOOP_ROLLOUT`, then report-only adjudication. Neither feature
generation nor Prior training is VLA training. SmolVLA and the Prior's
ResNet-18 stay frozen during Prior-module optimization. Expert actions are
training supervision only and are forbidden at live inference.

No Ours method may be generated, selected, trained, or rolled out unless the
frozen adjudication returns `VERIFIED_PRIOR_RESIDUAL`. If it returns
`PRIOR_SATURATES_PROBLEM`, close the thesis without designing Ours. The
authoritative contract is
`reports/a2c2_prior/problem_verification_protocol.json` and its readable
companion is `reports/a2c2_prior/problem_verification_protocol.md`.

The first frozen setup preflight stopped on an
`INFRASTRUCTURE_NULL_DEFECT`: LeRobot `0.4.4` calls the VLM module's bound
`forward` directly and therefore bypasses a PyTorch `Module.__call__` hook.
The failed attempt is preserved. The single authorized repair observes the
same unchanged bound-forward return through a temporary wrapper and restores
the original method afterward. It changes no scientific graph, panel,
identity, condition, action value, budget, threshold, or decision rule.

The first cached-feature attempt exposed a distinct `DATA_PIPELINE_DEFECT`
before any cache row or model forward: the runnable LeRobot `0.4.4` dataset
does not expose the LeRobot `0.2` `episode_data_index` field. Its root-bounded
repair derives the same subset-local half-open boundaries from the
authoritative `hf_dataset` episode-index column and requires exact agreement
with all 40 frozen episode IDs. No scientific contract changes.

The repaired cache run was resource-stopped when Windows host RAM reached
87.93%, despite a WSL-local reading of 23.3%. This distinct
`RESOURCE_COMPATIBILITY_DEFECT` exposed no comparator outcome. The flushed
cache remained valid at 384 anchors and 1,525 rows. With no research worker
left, a clean WSL shutdown returned host RAM to 65.34%; the same frozen cache
stage resumes only missing anchors without changing the contract.

The clean-VM verification exposed a distinct WSL default-allocation root:
Windows RAM rose from 65.78% to 88.93% under WSL2's default 50%-of-host memory
limit, while the minimum model RSS was 2.57 GiB. The durable cache remained
valid at 533 anchors and 2,115 rows. The previously absent global WSL2 config
is bounded to 3,584 MiB with swap disabled and immediate cache reclaim. This
environment-only correction changes no scientific field and must be verified
on the same actual path before the cache may continue.

The first Base rollout was OOM-killed during initial LIBERO construction,
before any episode or model forward. This distinct simulator-path
`RESOURCE_COMPATIBILITY_DEFECT` showed about 2.77 GiB anonymous RSS plus 256
MiB WSLg shared memory under the 3,584 MiB cache cap. The one bounded repair
raises the cap to 4,096 MiB, retains zero swap, and disables unused WSL GUI
support; EGL simulator semantics and every scientific field remain unchanged.

The simulator-memory verification reached the first Base episode but stopped
before persisting its row: WSL RAM was 95.8% and observed Windows RAM was
83.17%, both above the frozen 82% ceiling. The same resource root therefore
persisted after its one verified correction. A2C2 problem verification closes
as `PRIOR_INFRASTRUCTURE_BLOCKED`; no success value is counted, the scientific
delay/A2C2 hypothesis is unadjudicated, and no Ours is authorized. The next
campaign action is exactly-two-candidate `PIVOT_EPOCH_2` selection outside
both wrist dropout and asynchronous-delay correction.

## Strategic Pivot Epoch 2 And Terminal Decision

Effective `2026-07-19`, `PIVOT_EPOCH_2` evaluated exactly two theses outside
both wrist dropout and asynchronous-delay correction:

1. active-view goal disambiguation; and
2. autonomous retry/recovery after execution failure.

Neither passed every hard filter. Active-view disambiguation lacks released
official ActiveVLA code/checkpoints, a locally runnable active-camera Base,
and a twelve-hour local path. Retry/recovery lacks a runnable official FLARE
or ReTVL recovery prior; SAFE's public implementation is a detector rather
than a recovery controller; the thesis also overlaps multiple closed local
safety, repair, verifier, retrieval, and recovery families. Neither candidate
meets local feasibility `F >= 4`.

The exact epoch decision is `NO_DEFENSIBLE_PIVOT_FOUND`; the terminal campaign
and paper-readiness status is `NO_DEFENSIBLE_LOCAL_RESEARCH_PATH_FOUND`.
Do not create Pivot Epoch 3, a new local method acronym, Ours, Stage 0/A/B, or
a paper package. Preserve every prior positive, null, negative, and
infrastructure result without reinterpretation. Further research requires
explicit user redirection or materially new executable official artifacts and
resources.

The authoritative records are
`reports/strategic_pivot_epoch2_selection_result.json` and
`reports/strategic_pivot_final_decision.json`.

## A2C2 Resource-Feasibility Continuation Outcome

Effective `2026-07-19`, the user explicitly reopened exactly one
resource-feasibility continuation of the unchanged frozen A2C2
prior-verification stage. The authority source SHA-256 is
`B0F2999C80165CD3A9B96494FD90DB7AE10A0E8BE97916248476FF8044DCDE71`.
This did not reopen Pivot Epoch 3 or wrist dropout and did not authorize Ours,
physical or active-camera work, CPU/disk offload, swap-backed evaluation, or
any scientific-contract change. The historical
`NO_DEFENSIBLE_LOCAL_RESEARCH_PATH_FOUND` and `PRIOR_INFRASTRUCTURE_BLOCKED`
records remain preserved.

The exact Base actual-path resource smoke ran sequentially under verified 6,
8, 10, and 12 GB WSL caps, with zero swap, no WSLg process, permitted sub-70%
launch baselines (although above the 65% target), no active game, and the
frozen 82% Windows ceiling. All four runs
crossed that ceiling during model loading before a complete episode or child
trace could be persisted. Pagefile current usage did not grow, no page writes
were observed, no success was persisted or counted, and no scientific outcome
was exposed. The monitor tore down WSL after each failure and no research
worker remained. The optional 14 GB smoke was unauthorized because the
required `<=40%` cleaned baseline was not achieved and the observed failures
were host-ceiling failures rather than isolated low-WSL-cap failures.

The current exact decision is
`A2C2_RESOURCE_FUNDAMENTALLY_BLOCKED_ON_CURRENT_24GB_HOST`. This is a local
resource result, never an A2C2 method failure. The A2C2 delay hypothesis and
trained Prior remain scientifically unadjudicated, the full Base/Prior panel
remains unexecuted, and Ours and Pivot Epoch 3 remain unauthorized. The
original absent `.wslconfig` state is restored and WSL is shut down.

Measured peak physical use implies 32 GB installed RAM as the practical
minimum for the unchanged single workload, with 48 GB preferred for safe host
headroom. 64 GB adds little for this one sequential evaluation but may help
larger or concurrent workloads. No purchase is automatic. Any retry after a
manual upgrade requires new explicit authority and must preserve the exact
frozen protocol.

Authoritative records are
`reports/a2c2_prior/resource_continuation_authorization.json` and
`reports/a2c2_prior/resource_feasibility_continuation_result.json`.

## A2C2 Clean-Host Continuation And Scientific Closure

Effective `2026-07-19`, the user explicitly authorized one clean-host
continuation of the unchanged trained A2C2 path through a steer with SHA-256
`15771F9CE6074790D43CDA65A17A264E85B3A6FA1CED905CDD5B1114C7C587B0`.
This authority superseded the prior current-host resource conclusion only for
this bounded continuation; it did not erase the historical
`PRIOR_INFRASTRUCTURE_BLOCKED`,
`A2C2_RESOURCE_FUNDAMENTALLY_BLOCKED_ON_CURRENT_24GB_HOST`, or
`NO_DEFENSIBLE_LOCAL_RESEARCH_PATH_FOUND` records. It did not authorize
retraining, Ours, Pivot Epoch 3, wrist dropout, physical/real-camera work,
offload, swap, or any frozen scientific change.

After safe background cleanup, Windows RAM was 38.33%. Strict one-policy,
one-environment actual-path smokes ran sequentially at 8, 10, and 12 GB with
swap zero, no prefetch, no video/observation cache, no duplicate model
residency, and no scientific outcome persistence. The 8 and 10 GB attempts
were rejected by the exact no-pagefile-growth smoke gate. The corrected 12 GB
attempt completed 76 simulator steps and eight Base forwards with Windows RAM
peaking at 72.54%, zero pagefile growth/writes, no OOM/offload, and successful
teardown. Therefore 12 GB was the smallest passing cap and 14 GB was not run.

At that cap, the frozen scientific panel completed sequentially with one full
backbone and no prefetch or parallel task execution. All 45 unique frozen rows
were atomically persisted with matched identities, official reset states,
finite actions, zero exceptions, and no live expert actions. Results were:

- Base standard `e=10,d=0`: `10/15`;
- Base delayed `e=40,d=10`: `4/15`; and
- trained A2C2 Prior delayed `e=40,d=10`: `3/15`.

The Base-competence and repeatable-delay-gap gates passed. The Prior executed
2,936 live module forwards and nonzero corrections, recovered one delayed
failure, but regressed two delayed Base successes. It failed the frozen prior
improvement gate requiring at least +2 successes, at least two recoveries,
and at most one regression. The unchanged frozen adjudicator returned
`NO_DIAGNOSTIC_HEADROOM`; under the clean-host steer's exact report vocabulary
this maps without changing any threshold to
`A2C2_PRIOR_NO_LOCAL_IMPROVEMENT`.

This is only a negative result for the trained
`MECHANISM_FAITHFUL_A2C2_LOCAL_PORT` on the frozen local panel. It is not an
official A2C2 reproduction and does not disprove the paper's method. Close the
A2C2 thesis locally. Do not design Ours or start Pivot Epoch 3. The original
absent `.wslconfig` state must remain restored, WSL shut down, and further
research must wait for explicit user direction.

Authoritative records are
`reports/a2c2_prior/clean_host_resource_smoke_result.json` and
`reports/a2c2_prior/clean_host_prior_verification_result.json`.

## A2C2 Fidelity/Strong-Prior Continuation

Effective `2026-07-19`, the user explicitly authorized one focused
asynchronous-delay continuation through a steer with SHA-256
`5932431D45911ED562272E7BEF1184579696A96A39026B2F2C734F3E3AFC754D`.
This later authority preserves every v1 row and decision but supersedes the
instruction to wait. It does not reopen wrist dropout or Pivot Epoch 3 and
does not authorize physical-robot work, required real-camera work, premature
Ours, or a paper package before `PAPER_CANDIDATE_GO`.

The required first stage is complete as a report-only primary-source audit.
It found, independently of the v1 `3/15` outcome, that v1 used a different
third-party base instead of the author's paired Spatial-scratch base; missed
public author base/residual/dataset artifacts; and omitted the frozen official
evaluator's 180-degree rotation of both live RGB views. The reset
stabilization, action queue, per-step correction call, residual target, and
normalization/addition ordering were found faithful. The exact audit decision
is `A2C2_OBJECTIVE_FIDELITY_DEFECT_FOUND`.

Exactly one corrected path is now frozen under the label
`A2C2_FIDELITY_CORRECTED_LOCAL_PORT`. It pins the author's public
Spatial-scratch base and six-layer `add_vlm_context` residual checkpoint,
uses the exact live RGB orientation and released queue/integration semantics,
performs no residual retraining, and uses new development/verification reset
identities. It must not be called an official reproduction because the paper,
README, uploaded configuration, and released graph disagree on correction
steps, named checkpoint, and MLP width. No second fidelity correction is
allowed.

Run focused equivalence/load tests and one bounded actual-path smoke before
the new 45-row matched panel. Use the verified 12 GB temporary WSL cap,
`swap=0`, one full base residency, sequential execution, atomic persistence,
and missing-key-only resume. Do not repeat the 8/10/12 GB qualification. The
old init states `0..4` are forbidden for tuning; the corrected panel uses
tasks `0,4,8` and init states `5..9`.

Additional-Prior search is authorized only after corrected no-improvement.
Ours is authorized only if the corrected/additional Prior leaves a valid
residual (Route 1) or after A2C2 plus exactly one additional strong Prior both
fail while a repeated problem, diagnostic headroom, and a defensible novelty
gap remain (Route 2). A saturating Prior closes this problem without Ours.

Authoritative records are
`reports/a2c2_prior/fidelity_strong_prior_continuation_authorization.json`,
`reports/a2c2_prior/fidelity_gap_audit_result.json`, and
`reports/a2c2_prior/fidelity_corrected_protocol.json`.

## A2C2 Corrected-Path Preflight Closure

Effective `2026-07-19`, the one corrected implementation completed its
bounded outcome-suppressed actual-path preflight. The label remains
`A2C2_FIDELITY_CORRECTED_LOCAL_PORT`; it is not an official reproduction.
No residual training, Ours training, or scientific panel execution occurred.

The first load attempt preserved an objective checkpoint/source serializer
failure: the public prior stores a `[512,512]` image projection while the
later frozen source constructs `[512,512,1,1]`. Repository history uniquely
identified the immediately preceding author commit
`c197a011aabf070cf2c0b2b0705be5f33d178ad7`; the public checkpoint strict-loads
there without reshaping tensors or using a non-strict load. A second preserved
attempt exposed only the corresponding historical dataclass-config reporting
API. The author's `asdict` serialization route repaired that telemetry without
changing the model or protocol.

The completed development smoke used task id `2`, official init state `10`,
the pinned author Base and prior hashes, exact two-view RGB rotation, ten reset
stabilization steps, and the frozen standard/delayed queues. Base completed 10
model forwards. The delayed trace completed three Base forwards and 94 live
prior forwards; mean absolute prior correction was `0.091438221`. Both traces
were finite and exception-free. No task success was persisted or counted.

The frozen raw-action legality gate failed before any scientific panel row.
Base reached maximum absolute raw action `1.024949789`; Prior reached
`1.000505567`, both beyond `[-1,1]`. The released evaluator adds no explicit
clip. Robosuite clips internally at its controller boundary, but the frozen
protocol requires legal raw actions and forbids a post-hoc clipping rescue.
Do not relax or reinterpret this threshold.

The exact corrected decision is `CORRECTED_A2C2_EVALUATION_INVALID`. The
new 45-row panel was not started and contains zero scientific rows. This does
not establish improvement or no-improvement and does not alter v1's separate
`A2C2_PRIOR_NO_LOCAL_IMPROVEMENT` result. The steer authorizes an additional
Prior after corrected no-improvement, not after corrected invalidity; hence
another Prior, Ours, Stage 0/A/B, Pareto/generalization, and the paper package
remain unauthorized.

Peak allocated VRAM was `1532.542 MiB`, peak process RSS `4063.348 MiB`, and
Windows physical use peaked at `65.09%`, with zero swap, pagefile growth,
offload, or OOM. The temporary `.wslconfig` was removed, WSL was shut down,
and no worker remains.

The authoritative result is
`reports/a2c2_prior/fidelity_corrected_actual_path_smoke_result.json`.

## A2C2 Official Action-Semantics Correction

Effective `2026-07-19`, the user explicitly authorized exactly one evaluation-
semantics correction through a steer with SHA-256
`CDC674DB9E0EFDC85F3529FA7387D4E3A9BD31DF91A66B0D9B87C1279DA6C0B0`.
This authority does not erase or rewrite the prior
`CORRECTED_A2C2_EVALUATION_INVALID`; preserve that result as
`HISTORICAL_LOCAL_STRICT_RAW_BOUND_GATE_RESULT`. It authorizes no method,
checkpoint, task, reset, delay, timeout, success, outcome-threshold, or
scientific-panel change and no external wrapper action clipping.

The required primary-source audit returned
`OFFICIAL_ACTION_SEMANTICS_VERIFIED`. The author evaluator passes the
unnormalized 7-D policy output unchanged to LIBERO. The official wrapper passes
it unchanged to robosuite. Robosuite's `SingleArm` sends the first six values to
the OSC controller, whose native `scale_action` clips/scales to controller
output bounds, and sends the seventh to the Panda gripper's signed incremental
native saturation path. It then applies actuator mapping and torque limits.
Base and Prior have exactly matching action mean/std and traverse the same
post-policy environment path.

For the active correction, raw nominal-bound exceedance is required diagnostic
evidence and not an automatic invalidity. Validity instead requires finite 7-D
raw output, unchanged action delivery, controller acceptance, native arm and
gripper effective values within their official bounds, actuator and torque
bounds, and finite simulator state. The frozen practical Prior-instability rule
requires a Prior-minus-matched-Base maximum exceedance increase of at least
`0.05` on both development identities plus either a raw exceedance-fraction
increase of at least `0.02` or a native arm clip-step-fraction increase of at
least `0.10` on both identities. No outcome was used to set this rule.

The next stage is exactly four outcome-suppressed technical traces: matched
delayed Base/Prior on task/init `(2,11)` and `(6,11)`, each for 80 fixed steps.
The runner may not inspect, persist, or count success, done, or reward. Only
`CORRECTED_A2C2_OFFICIAL_SEMANTICS_SMOKE_PASS` opens the unchanged 45-row panel
on tasks `0,4,8`, official init states `5..9`, and the three previously frozen
conditions. Until that pass, do not run the panel, choose another Prior, design
Ours, or prepare a paper package.

Authoritative records are
`reports/a2c2_prior/official_action_semantics_continuation_authorization.json`,
`reports/a2c2_prior/official_action_semantics_audit_result.json`, and
`reports/a2c2_prior/official_action_semantics_protocol.json`.

The preregistered outcome-suppressed smoke subsequently completed as
`CORRECTED_A2C2_OFFICIAL_SEMANTICS_SMOKE_PASS`. It executed all four 80-step
technical traces, persisted zero scientific episode rows, did not inspect,
persist, or count success/done/reward, and had no exception. Every raw action
was finite and 7-D; controller acceptance, native arm/gripper bounds, actuator
and torque limits, and simulator-state finiteness all passed. Raw nominal-bound
exceedance occurred only in gripper dimension 6, native arm clipping occurred
on zero steps, and the Prior-specific practical-instability rule was false on
both matched identities. Swap, pagefile growth, offload, and OOM were zero.

The unchanged 45-row panel is therefore now authorized under its already-
frozen tasks, official init states, conditions, horizons, success predicate,
effect thresholds, atomic persistence, and missing-key-only resume contract.
No additional Prior, Ours, Stage 0/A/B, or paper package is authorized before
that panel's exact decision. The smoke record is
`reports/a2c2_prior/official_action_semantics_smoke_result.json`.

The first full panel attempt then completed all 45 rows internally, but its
scientific candidate decision is quarantined. The inherited monitor returned a
host failure solely because system-wide Windows pagefile `CurrentUsage` rose
from 72 to 78 MiB. All 872 samples recorded zero `PageWrites/sec` and zero
`PagesOutput/sec`; WSL swap, model offload, OOM, and host-ceiling termination
were zero/false, and memory release passed. The monitor had conflated a
reservation-counter drift with actual paging, a condition not frozen by the
active resource contract.

Under the steer's explicit non-scientific logging/telemetry repair authority,
one minimal repair is frozen before outcomes may be adopted: persist
`CurrentUsage` drift as a diagnostic, detect pagefile activity from nonzero
sampled `PageWrites/sec` or `PagesOutput/sec`, and otherwise retain the exact
12 GB, swap-zero, one-residency, no-offload, no-OOM, 82% host-ceiling, and
memory-release gates. The same 45 frozen rows must rerun from zero under a new
run id. No model, action path, task, identity, delay, timeout, success rule,
outcome threshold, or adjudicator may change. The failed attempt and repair
protocol are `reports/a2c2_prior/official_action_semantics_panel_host_telemetry_failed_attempt.json`
and `reports/a2c2_prior/official_action_semantics_panel_host_telemetry_repair_protocol.json`.
