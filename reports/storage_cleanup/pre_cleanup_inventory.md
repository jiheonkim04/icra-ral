# Pre-cleanup Storage Inventory

Captured: `2026-07-20T12:51:53.921658+09:00`
Branch / HEAD: `codex/epoch6-ral-submission-convergence-v2` / `a621a993234847894c0bb54b05ac091a70729aaa`
Windows free: **262.966 GB**
WSL internal free: **932.575 GB**
WSL VHDX physical size: **94.967 GB**
Manifested deletion allocation: **84.299 GB** across **784** exact targets.

## Safety refresh

- Relevant worker active: `False`.
- Branch, local/remote HEAD, ancestry, worktree, candidate mtimes, selected snapshot blobs, and high-risk hashes were refreshed after the overlap pause.
- All Epoch 6 overlap-window state is `KEEP_OR_REVIEW`; it is absent from the delete manifest.
- The two protected rollout directories remain `PROTECTED`.

## Large retained or review items

| Classification | Path | Allocated GB | Reason |
|---|---|---:|---|
| KEEP | `C:\assets\data\libero` | 100.443 | benchmark data |
| KEEP | `C:\Users\jiheo\AppData\Local\wsl\{180ce51e-73f9-4a6b-91f0-3a4f1842ad61}\ext4.vhdx` | 94.967 | verified WSL virtual disk; compaction only |
| USER_DECISION_REQUIRED | `/home/jiheon/assets/checkpoints/lightvla` | 15.455 | local checkpoint modifications |
| USER_DECISION_REQUIRED | `/home/jiheon/assets/checkpoints/openpi` | 12.439 | non-immutable GCS source |
| KEEP_OR_REVIEW | `C:\Users\jiheo\tca_map\runs` | 10.311 | unique scientific outputs/checkpoints |
| AMBIGUOUS | `C:\Users\jiheo\AppData\Local\Temp\DiagOutputDir` | 10.196 | unrelated/ambiguous diagnostic ownership |
| KEEP_OR_REVIEW | `C:\assets\checkpoints` | 3.758 | referenced Windows checkpoints |
| KEEP | `/home/jiheon/assets/checkpoints/xvla_hf_cache` | 3.522 | selected runnable checkpoint closure |
| KEEP | `C:\assets\datasets\lerobot_libero` | 1.936 | retained dataset dependency |
| KEEP_OR_REVIEW | `C:\Users\jiheo\tca_map\rollouts` | 0.008 | includes two explicitly protected directories |

## Protected rollout sizes

- `C:\Users\jiheo\tca_map\rollouts\2026_07_17`: 5.144 MB
- `C:\Users\jiheo\tca_map\rollouts\2026_07_18`: 0.925 MB

The complete candidate fields, exact deletion targets, source revisions, hashes, and dependency closure are in the JSON manifests.
