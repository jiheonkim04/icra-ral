# Project State

Date: 2026-07-08

Branch:

`codex/research-reset-target-grounded-actionmap-scout`

Current decision:

`NEED_ACTIONMAP_ANCHOR_REPRO_FIRST`

## Execution Boundary For This Pass

- Experiments happened: no.
- Training happened: no.
- Rollout/replay happened: no.
- Loss computation happened: no.
- Downloads happened: no.
- GPU use happened: no.
- OpenVLA-OFT happened: no.
- Local proxy diagnostics happened: no.
- New NumPy surrogate happened: no.
- Method implementation happened: no.

## Repository Start State

- Starting branch: `main`.
- Starting status: clean.
- Starting latest commit: `fd873ea Add SafeLoRA-VLA feasibility gate`.
- Working branch created: `codex/research-reset-target-grounded-actionmap-scout`.

## Reports Read

Requested reports were read or inspected:

- `reports/ral_strategy_reset.md`
- `reports/all_killed_routes_summary.md`
- `reports/simple_baseline_failure_patterns.md`
- `reports/next_topic_selection_criteria.md`
- `reports/next_topic_anti_baseline_prescreen.md`
- `reports/project_state.md`
- `reports/next_actions.md`
- `reports/decision_log.md`
- `reports/target_prior_tca_kill_summary.md`
- `reports/css_shield_kill_summary.md`
- `reports/execspec_kill_summary.md`
- `reports/amp_gd_kill_summary.md`
- `reports/resetspec_kill_summary.md`
- `reports/phase_locked_retiming_kill_summary.md`
- `reports/tl_chunkrepair_kill_summary.md`
- `reports/contacttube_aug_kill_summary.md`
- `reports/prism_vla_kill_summary.md`
- `reports/contactset_vla_kill_summary.md`
- `reports/safetrace_vla_kill_summary.md`
- `reports/safelora_vla_state1_decision.md`
- `reports/safelora_vla_lora_feasibility.md`
- `reports/safelora_vla_source_feasibility.md`

Additional relevant local anchor reports were inspected:

- `reports/latest_anchor_paper_matrix.md`
- `reports/actionmap_anchor_state1_result.md`
- `reports/actionmap_anchor_related_work_matrix.md`

## Consolidated Killed Routes

Dead as current RA-L routes:

- Target-Prior TCA-Map as originally formulated.
- TCA-Select.
- Weak 7D MLP TCA head.
- CSS-Shield.
- ExecSpec-Repair.
- AMP-GD.
- ResetSpec-Retarget.
- Phase-Locked Retiming.
- TL-ChunkRepair.
- ContactTube-Aug.
- PRISM-VLA.
- ContactSet-VLA.
- Local ActionMap mini-anchor extension.
- SafeTrace-VLA.
- SafeLoRA-VLA as topic novelty.

The reusable pieces are diagnostics, split/audit tooling, object and target resolvers, LIBERO-Para/PRIDE metrics, ActionMap/TCA comparison scaffolds, and simple-baseline discipline.

## Salvage State

Only one family is salvageable:

**Target-Prior TCA family -> Target-Grounded ActionMap / Language-Grounded Action Heatmap.**

The salvage is conditional. It cannot proceed from the failed local ActionMap mini-anchor. It requires ActionMap anchor reproduction first.

## Latest Literature State

Primary-source scan checked ActionMap, Direct Action-Head Injection of a Grounded 3D Point, LIBERO-Para, OpenVLA-OFT, LoRA-SP, QVLA, ActQuant, DyQ-VLA, CAC-VLA, w2 VLA, GuidedVLA, RoVLA, OA-WAM, Gaze2Act, VLA-Corrector, A2C2, and RTC.

Key conclusion:

Target-grounded heatmap novelty is possible but narrow. Adjacent work already covers semantic conditioning, object grounding, consistency, latent action conditioning, and single-point injection. The candidate must specifically contribute semantic target prior conditioning of an ActionMap-style voxel action heatmap with counterfactual target sensitivity and object lexical/paraphrase robustness.

## Current Recommended Direction

Recommended direction:

Target-Grounded ActionMap / Language-Grounded Action Heatmap for VLA Manipulation.

Current decision:

`NEED_ACTIONMAP_ANCHOR_REPRO_FIRST`

Estimated kill risk:

`0.65` before ActionMap anchor reproduction.

## New Reports Created Or Updated

- `reports/research_prompt_cleanup.md`
- `reports/route_salvage_matrix.md`
- `reports/latest_anchor_paper_matrix.md`
- `reports/final_research_direction_recommendation.md`
- `reports/target_grounded_actionmap_feasibility.md`
- `reports/target_grounded_actionmap_experiment_plan.md`
- `reports/target_grounded_actionmap_kill_criteria.md`
- `reports/project_state.md`
- `reports/next_actions.md`
- `reports/decision_log.md`
