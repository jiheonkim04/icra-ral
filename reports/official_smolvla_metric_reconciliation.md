# Official SmolVLA Metric Reconciliation

Date: 2026-07-09 KST

Branch: `codex/official-smolvla-libero-failure-mining`

## Question

Why did the previous official baseline show action L2 improvement while eval loss worsened?

## What Eval Loss Measures

Installed `SmolVLAPolicy.forward()` returns `(loss, loss_dict)`.

The scalar loss is:

- produced by `self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)`;
- a normalized action-chunk flow/training objective;
- optionally masked by `actions_id_pad` / `action_is_pad`;
- sliced to `self.config.max_action_dim`;
- reduced as `losses.mean()`.

It is not simulator success, not raw 7D one-step action L2, and not a language/token prediction loss.

## What Action L2 Measures

Action L2 in these reports is:

- `policy.select_action(batch)` for a single current-frame action;
- passed through the official postprocessor/unnormalizer;
- compared against the raw official `action` label for that current frame;
- reported in raw postprocessed 7D action space.

This makes action L2 an offline action-quality proxy, not a task-success metric.

## Split Alignment

Official `lerobot/libero` exposes only `{"train": "0:1693"}` in the downloaded metadata. There is no official eval split in the local asset.

The failure-mining run used:

- train episode: `0`;
- held-out diagnostic episodes: `1, 4, 2, 3, 7, 9, 8, 13, 14, 15`;
- held-out task groups: `5`;
- held-out frames: `200`.

This is a deterministic official-data diagnostic split, not a benchmark.

## Reconciled Interpretation

Action L2 and eval loss can disagree because they inspect different objects:

- action L2: current raw action after official postprocessing;
- eval loss: normalized future action chunk flow objective.

The broad failure-mining result no longer repeats the small-sample "action L2 improved but eval loss worsened" pattern in aggregate. Instead:

- frozen/base action L2: `0.106514960`;
- rank-4 LoRA action L2: `0.118024259`;
- frozen/base eval loss: `0.011978370`;
- rank-4 LoRA eval loss: `0.012148290`.

On 200 held-out frames, rank-4 LoRA worsened both aggregate action L2 and aggregate eval loss, while still helping `98` frames and hurting `102` frames. The issue is therefore better described as mixed low-data adapter interference than pure metric conflict.

## Recommendations

Primary offline metric:

- postprocessed held-out action L2;
- always include translation L2, rotation L2, gripper error/sign accuracy, per-task breakdown, per-phase breakdown, and mean-action prior.

Secondary stability metric:

- normalized action-chunk flow eval loss from `SmolVLAPolicy.forward()`.

Required warning:

- if action L2 and eval loss disagree, report both and do not call the run a success.

Current gate:

- action L2 and eval loss do not conflict strongly enough to block method selection;
- both show that the broader rank-4 LoRA adapter is not a reliable aggregate improvement;
- method readiness, if pursued, should target task/frame-specific adapter interference while retaining frozen/base as a mandatory anchor.
