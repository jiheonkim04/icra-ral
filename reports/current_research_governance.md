# Current Research Governance

Date: 2026-07-12 KST

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
