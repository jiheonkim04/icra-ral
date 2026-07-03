# Low-Compute Distributional Stack Integration

## Integrated Source Branches

This branch integrates the validated planning and method work from:

1. `codex/local-papergrade-runner`
2. `codex/no-large-openvla-lowcompute-protocol`
3. `codex/distributional-tca-select-action-decoder-study`

The GitHub branch already existed when this integration pass started. Its tree matched current `main` for repository contents, and the three source branches were already ancestors of `main`. This report records the conservative integration policy and keeps a concrete integration commit on `codex/integrate-lowcompute-distributional-stack`.

## Conflict Policy

No content conflicts needed manual line-level resolution in this connector-based integration pass. The resolved policy is:

- keep strict no-large-OpenVLA local rules,
- keep SmolVLA-first as the first real-adapter path,
- keep frozen backbone, cached features, head-only TCA-Map, low-resolution/coarse-to-fine heatmaps as the local path,
- keep Distributional TCA-Select as the main inference-time method contribution,
- keep heuristic TCA-Select as an ablation only,
- keep LoRA/QLoRA as required experimental tracks after head-only validation, not the main novelty,
- preserve Windows PowerShell scripts and Linux/WSL shell companions,
- preserve compute-budget enforcement before new pilot commands,
- preserve local paper-grade runner scripts and cloud handoff plans,
- preserve tests and dummy/smoke validation paths.

## Required Files Confirmed

The integration branch contains:

- `configs/compute_budget.yaml`
- `configs/distributional_tca_select.yaml`
- `reports/no_large_openvla_strategy.md`
- `reports/final_method_spec_distributional_tca_map.md`
- `reports/local_papergrade_plan.md`
- `reports/action_decoder_landscape.md`
- `reports/mg_select_vs_distributional_tca_select.md`
- `tests/test_distributional_tca_select.py`
- `tests/test_lora_config_guards.py`

## Local Safety Boundary

This branch does not authorize GPU jobs, dataset/checkpoint downloads, simulator rollouts, heavy VLA imports, or OpenVLA-OFT execution. Those remain behind explicit gates and separate approval.
