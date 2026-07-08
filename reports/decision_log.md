# Decision Log

## 2026-07-08: Research Reset And Target-Grounded ActionMap Scout

Decision:

`NEED_ACTIONMAP_ANCHOR_REPRO_FIRST`

Reason:

All previous killed routes were consolidated. The only salvageable family is the old target-prior/action-decoder family, but only after reframing it as Target-Grounded ActionMap / Language-Grounded Action Heatmap. The old TCA-Select and weak 7D heads remain killed.

The candidate has a real paper gap:

- ActionMap provides voxel action heatmaps but lacks explicit semantic target/object heatmap conditioning.
- Direct grounded 3D point injection provides single-point action-head grounding but not a target-conditioned action heatmap distribution.
- LIBERO-Para exposes object lexical/paraphrase fragility but does not solve it with action decoder design.

However, the local ActionMap mini-anchor failed mean-action and cheap-MLP gates, and recent adjacent work such as CAC-VLA, w2 VLA, GuidedVLA, and RoVLA raises the novelty threshold for any generic semantic-conditioning or consistency claim.

Consequence:

Target-Grounded ActionMap remains the recommended direction, but the next valid step is ActionMap anchor reproduction first. Do not implement the target-grounded method yet.

Execution boundary:

- experiments happened: no;
- training happened: no;
- rollout/replay happened: no;
- loss computation happened: no;
- downloads happened: no;
- GPU use happened: no;
- OpenVLA-OFT happened: no;
- local proxy diagnostics happened: no;
- new method implementation happened: no.

Reports created or updated:

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
