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

## Post-PSE Research Design Governance

This section applies to all method candidates after the valid PSE-VLA adjudication. It is not retroactive to PSE.

Future method cycles must be problem-first, novelty-aware, mechanism-explicit, mathematically justified, and external-prior-early.

Before choosing a method, identify:

1. the concrete unresolved failure or assumption;
2. evidence that the failure matters in closed-loop control;
3. how the closest external prior addresses or fails to address it;
4. the proposed technical mechanism;
5. the falsifiable path from mechanism to policy behavior to closed-loop outcome;
6. the smallest experiment that can test the hypothesis.

Every candidate must declare exactly one contribution type:

- `PRIOR_EXTENSION`
- `IMPLICIT_GAP_SOLUTION`
- `CROSS_PAPER_SYNTHESIS`
- `NEW_DEPLOYMENT_PROBLEM`

Generate exactly three candidates and score them on:

- provisional method novelty: `30%`
- importance of unresolved problem: `20%`
- technical mechanism quality: `20%`
- external-prior comparison feasibility: `15%`
- decisive local experiment feasibility: `15%`

The score selects what to test. It is not a prediction of empirical success.

For close literature, do not treat abstracts, contribution lists, limitations, future-work text, or discussion framing as authoritative. Reconstruct papers from equations, algorithms, architecture, training supervision, inference procedure, data generation, code when available, appendices, ablations, failure cases, benchmark conditions, omitted comparisons, and fixed experimental variables.

Each close-paper record must separate:

- `AUTHOR_STATED`: what the authors explicitly claim as contribution, novelty, limitation, or future work;
- `INDEPENDENTLY_INFERRED`: what follows from the actual method and evidence;
- `CROSS_PAPER_SYNTHESIZED`: what becomes visible only after comparing multiple papers.

For the closest literature, build a mechanism map covering observation/input, learned representation, supervision, objective, policy component changed, action-generation mechanism, inference-time intervention, assumed feedback, benchmark condition, primary metric, actual demonstrated causal link, and untested causal link.

Every selected method must include `reports/<method>/mathematical_mechanism_audit.md` before implementation. The audit must define variables and tensor shapes, mathematical formulation, representation learned, exact policy component affected, training objective, inference algorithm, data and supervision source, gradient path, expected behavioral effect, expected closed-loop consequence, closest mathematical alternative, simplest equivalent baseline, key ablation, and known failure mode.

Every proposed module and loss term must state what quantities are compared, why the discrepancy is appropriate, where it is used, which parameters receive gradients, what behavior it should induce, what simpler alternative could replace it, and which ablation proves it matters.

Do not add KL divergence, entropy, contrastive learning, mutual information, consistency losses, or regularization merely because they sound sophisticated. KL may be used only when both arguments are valid probability distributions or justified density approximations, support and normalization are defined, the KL direction is justified, and the estimator is reliable. SmolVLA flow outputs are not automatically normalized action probability distributions. Do not compute KL directly between deterministic 7D action vectors.

Before implementation, identify the closest external prior, strongest recent method on the same claim axis, official code or checkpoint availability, exact backbone and benchmark compatibility, required modifications for fair comparison, and whether a faithful local proxy is possible. The closest prior must enter no later than the first confirmatory closed-loop comparison.

Future first serious prototypes should normally compare exactly:

1. unmodified backbone;
2. closest external prior or a faithful transparently labeled proxy;
3. full proposed method;
4. key ablation;
5. one strongest simple reviewer-killer baseline.

No more than one mandatory simple killer baseline is required at the initial prototype stage. Additional internal controls are allowed only when they correspond to a concrete reviewer objection, test a genuinely different trivial explanation, could change the scientific decision, and are cheaper than proceeding directly to the prior comparison.

Use this future experiment order:

1. Stage 0: small problem diagnostic.
2. Stage 1: mechanism smoke.
3. Stage 2: early paper comparison on one matched manifest: Base, closest external prior or proxy, Ours, key ablation, and one strongest simple baseline.
4. Stage 3: confirmatory paired expansion only when Ours is not clearly inferior.
5. Stage 4: after success, second backbone or second condition, clean retention, statistics, and efficiency.

Pre-implementation rejection is allowed only for near-exact prior-art duplication, obvious mathematical equivalence to a trivial method, mathematically invalid formulation, essential unavailable resource, no concrete falsifiable mechanism, no feasible fair comparison with the closest prior, or no plausible connection between the intervention and policy behavior. Unknown empirical performance is not a rejection reason.

Before any future terminal decision, `scripts/check_current_research_governance.py` must pass.
