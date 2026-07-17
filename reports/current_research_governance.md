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
