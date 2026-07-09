# Official SmolVLA Baseline Naming Audit

Date: 2026-07-10 KST

## Finding 1: MoIRA-Style Router

Current labels seen in reports/code:

- `moira_style_instruction_task_router`
- `MoIRA-style task/instruction router`
- `MoIRA router`

Audit answer:

- Is it an official MoIRA reproduction? no
- Is it a local task/instruction routing proxy? yes

Reason: the current router selects between frozen/base and rank-4 LoRA using local task/instruction routing logic derived from existing predictions. It does not run the official MoIRA implementation, train MoIRA experts, use MoIRA's official router, or reproduce MoIRA's official evaluation.

Required future name:

`task_or_instruction_router_proxy`

Allowed display label:

`task/instruction router proxy (not official MoIRA)`

Disallowed future wording:

- `official MoIRA`
- `MoIRA reproduction`
- `MoIRA baseline`, unless the official MoIRA method is actually run

## Finding 2: Static Merge / Adapter Soup

Current labels seen in reports/code:

- `adapter_soup_static_merge`
- `static merge`
- `static mix`
- `static_mix_val_selected`

Audit answer:

- Is it adapter-weight merging? no
- Is it action-space static interpolation between base and LoRA predictions? yes

Reason: current code evaluates weighted action predictions after inference. It does not merge LoRA adapter weights, checkpoint tensors, or PEFT adapter parameters.

Required future name:

`validation_selected_action_space_static_mix`

Allowed display label:

`validation-selected action-space static mix`

Disallowed future wording:

- `adapter soup`
- `adapter-weight merge`
- `weight soup`
- `merged adapter`

These names are allowed only if adapter weights are actually merged and evaluated.

## Naming Corrections For Future Reports

| old / risky name | corrected future name | reason |
| --- | --- | --- |
| `moira_style_instruction_task_router` | `task_or_instruction_router_proxy` | local proxy, not official MoIRA |
| `MoIRA router` | `task/instruction router proxy (not official MoIRA)` | avoids implying official reproduction |
| `adapter_soup_static_merge` | `validation_selected_action_space_static_mix` | action interpolation, not adapter-weight soup |
| `static merge` | `validation-selected action-space static mix` | avoids weight-merge ambiguity |
| `frame oracle` | `frame oracle upper bound` | oracle uses labels and is not realistic |
| `task oracle` | `task oracle upper bound` | oracle uses labels and is not realistic |

## Result Impact

The naming issues do not invalidate the stored offline action-L2 metrics. They do require correction in all future reports and any paper-facing summary before official rollout or RA-L evidence claims.
