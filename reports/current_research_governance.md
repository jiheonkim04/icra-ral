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

Before any future terminal decision, `scripts/check_current_research_governance.py` must pass.
