# RL4IL Official Code Fidelity Audit

- Decision: `RL4IL_OFFICIAL_RELEASE_PRIOR_ROLLOUT_NOT_ARMED_CONSTANT_LABEL_SUPERVISION`
- Official clone: `/mnt/c/assets/repos/RL4IL-Missing-Camera`
- Official clone HEAD: `e1dd5b741ebb6b392bfd1f8cbb61bad82417e9bd`
- Dataset root checked: `/mnt/c/assets/data/libero`
- Training, optimizer, checkpoint, simulator rollout, and Ours work: none.

## Result

RL4IL remains the closest paper-level external prior for complete in-hand/wrist camera dropout, but the cloned official release should not be armed as the prior comparator as-is.

The released demo loaders assign `label=1` to every loaded demonstration, while retrieval/fusion training uses `tr_labels` as its supervised target. With a single label value, validation accuracy and PPO candidate rewards can become degenerate. A rollout might still replay retrieved demo actions, but it would not establish the paper’s learned action-signal retrieval/fusion mechanism.

## Dynamic label audit

| suite | demos checked | unique labels |
|---|---:|---|
| `libero_goal` | 50 | `[1.0]` |
| `libero_object` | 50 | `[1.0]` |
| `libero_spatial` | 50 | `[1.0]` |

No checkpoint files (`.pt`, `.pth`, `.ckpt`, `.bin`, `.safetensors`) were found in the official clone.

## Static code evidence

- Constant label assignment appears in all five scripts checked.
- `train_policy` is present in all five scripts and consumes `tr_labels`.
- The fusion head target is `float(tr_labels[i])` in the suite scripts.
- The rollout path replays `train_demos[best_tr_idx]["actions"]`, but the candidate-selection training signal is still the constant label.

## Next action

Do not run an official-release RL4IL comparator rollout as-is. Either acquire/fix official checkpoints, obtain an upstream correction, or implement a mechanism-faithful local port with an action-sequence oracle before using RL4IL as the external prior comparator.
