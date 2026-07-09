# Official SmolVLA Baseline Naming Policy

Date: 2026-07-10 KST

Status: `FROZEN_FOR_FUTURE_REPORTS`

This policy applies prospectively. Historical metric files are preserved as artifacts and must not be silently rewritten.

## Canonical Names

| Canonical name | Human label | Meaning | Reportable as realistic rollout policy |
| --- | --- | --- | --- |
| `frozen_base` | Frozen SmolVLA-LIBERO Base Policy | Official frozen `lerobot/smolvla_libero` checkpoint with official preprocessing and postprocessing. | Yes |
| `rank4_lora` | Standard Rank-4 LoRA | Standard PEFT LoRA adapter policy trained from the official SmolVLA-LIBERO base under the fixed split/protocol. | Yes, only with persisted adapter checkpoint |
| `validation_selected_action_space_static_mix` | Validation-Selected Base-LoRA Action-Space Static Mix | `a_mix = alpha * a_lora + (1 - alpha) * a_base`, where `alpha` is selected only on validation data. | Yes, only with base and persisted LoRA adapter |
| `task_or_instruction_router_proxy` | Task/Instruction Router Proxy | Local proxy router evidence. | No official MoIRA claim |
| `frame_oracle_upper_bound` | Frame Oracle Upper Bound | Per-frame oracle selection bound. | No |
| `task_oracle_upper_bound` | Task Oracle Upper Bound | Per-task oracle selection bound. | No |

## Required Interpretations

`validation_selected_action_space_static_mix` is action-space interpolation. It is not adapter-weight merge, not model soup, and not adapter soup.

`task_or_instruction_router_proxy` is a local proxy. It is not an official MoIRA reproduction and must not be labeled as one.

`frame_oracle_upper_bound` and `task_oracle_upper_bound` are upper bounds. They are not realistic deployable policies and must not be compared as rollout-ready baselines.

## Legacy Mapping

Future writing should map legacy names as follows:

| Legacy or informal name | Canonical future name |
| --- | --- |
| `adapter_soup` | `validation_selected_action_space_static_mix` |
| `static_merge` | `validation_selected_action_space_static_mix` |
| `MoIRA proxy` | `task_or_instruction_router_proxy` |
| `oracle router` | `frame_oracle_upper_bound` or `task_oracle_upper_bound`, depending on granularity |

This mapping is explanatory only. It does not authorize editing old result JSON or changing historical metric values.

## Policy

All paper-facing and rollout-facing summaries after this protocol fix must use the canonical names. If an old artifact uses a legacy name, cite the artifact as-is and add the legacy mapping in surrounding text.
