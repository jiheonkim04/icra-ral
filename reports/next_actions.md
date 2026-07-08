# Next Actions

Date: 2026-07-08

Current decision:

`NEED_ACTIONMAP_ANCHOR_REPRO_FIRST`

## Immediate Next Action

Do not implement Target-Grounded ActionMap yet.

The next valid task is an ActionMap anchor reproduction planning pass that decides whether the ActionMap-style heatmap substrate can be made credible against mean action, linear/L1, and simple MLP baselines without candidate collapse.

## Next Prompt

Use this next prompt if continuing from this state:

```text
You are working in C:\Users\jiheo\tca_map.

Do not implement Target-Grounded ActionMap yet.
Do not train.
Do not run experiments.
Do not rollout.
Do not download.
Do not use GPU.
Do not run OpenVLA-OFT.

Goal:
Plan the minimum ActionMap anchor reproduction gate required before Target-Grounded ActionMap can be considered.

Read:
- reports/final_research_direction_recommendation.md
- reports/latest_anchor_paper_matrix.md
- reports/target_grounded_actionmap_feasibility.md
- reports/target_grounded_actionmap_experiment_plan.md
- reports/target_grounded_actionmap_kill_criteria.md
- reports/actionmap_anchor_state1_result.md
- reports/actionmap_anchor_related_work_matrix.md
- reports/route_salvage_matrix.md

Create:
- reports/actionmap_anchor_repro_first_plan.md
- reports/actionmap_anchor_repro_first_kill_criteria.md
- reports/actionmap_anchor_repro_first_source_audit.md
- reports/project_state.md
- reports/next_actions.md
- reports/decision_log.md

Required decision:
Choose exactly one:
- READY_FOR_ACTIONMAP_ANCHOR_REPRO_STATE1
- SOURCE_BLOCKED_ACTIONMAP_ANCHOR
- NO_GO_ACTIONMAP_TOO_WEAK_LOCALLY
- RETURN_TO_OFFICIAL_ANCHOR_REPRODUCTION

No method implementation in this run.
```

## If Anchor Reproduction Later Goes Green

Only after the ActionMap anchor beats mean action, linear/L1, and simple MLP without collapse, the next possible decision may become:

`GO_TARGET_GROUNDED_ACTIONMAP_STATE1`

At that point, the exact next prompt should be a Stage 1 feasibility-only task, not a training or method implementation task.

## Standing Stop Rules

Stop if the next task requires:

- downloads;
- GPU;
- full VLA training;
- OpenVLA-OFT;
- simulator rollout;
- benchmark-scale evaluation;
- local proxy-first method invention;
- LoRA as novelty;
- target labels or future actions as inference-time information.
