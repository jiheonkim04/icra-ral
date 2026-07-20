# VLA Storage Cleanup Report

Terminal status: `CLEANUP_COMPLETE_COMPACTION_PENDING`

The audited cleanup is complete. All **784** exact `VERIFIED_DISPOSABLE`
targets were deleted, with **0** manifest targets remaining. C: gained
**49.831 GB** of immediately usable space. WSL gained **32.190 GB** internally,
but the Windows VHDX has not yet shrunk, so that second amount is not counted as
host-space recovery yet.

## Storage result

| Measure | Before | After | Change |
|---|---:|---:|---:|
| Windows C: free | 262.966 GB | 312.797 GB | **+49.831 GB** |
| WSL internal free | 932.575 GB | 964.765 GB | **+32.190 GB** |
| Ubuntu-22.04 VHDX physical | 94.967 GB | 94.967 GB | **0 GB** |

The manifest estimated 84.299 GB of allocated targets: 50.276 GB on Windows
and 34.024 GB inside WSL. Actual filesystem deltas are authoritative; Conda
hardlinks and allocation accounting explain the difference from the estimate.

The archive commit scope is 15 files / 41,865 lines / 1.930 MB. It exceeds the
5,000-line review threshold because `delete_manifest.json` alone contains
34,136 lines and the two execution ledgers retain per-target outcomes for all
784 permanent deletions. Those exact rows are required safety/audit evidence;
collapsing them would lose target identity, hashes/revisions, and deletion
status.

## What was removed

| Category | Exact targets | Manifest GB |
|---|---:|---:|
| VLA WSL crash dumps | 2 | 39.943 |
| Public closed-route OpenVLA-OFT checkpoint | 1 | 15.941 |
| uv cache | 10 | 8.582 |
| pip caches | 7 | 7.129 |
| Conda package caches, never environments | 748 | 4.253 |
| Clean, pinned source-audit-only clones | 2 | 3.117 |
| Nonselected public Hugging Face model caches | 2 | 4.070 |
| Closed RL4IL CLIP feature cache | 1 | 1.214 |
| Torch/Xet/lock metadata | 16 | 0.051 |

The two crash dumps were rehashed after the concurrent Goal paused:

- 20.145 GB — `9D17164F0C822CC56D9FEDB32155FFB94A4F683B624CDFE2BA6B229310AE28AF`
- 19.798 GB — `8F3A0CA82FA76E07CB17F1A8CA40C57B00DC8CB85E02CBFE26CFF67820FF0198`

The removed OpenVLA-OFT checkpoint remains reacquirable from the retained
public revision `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`; all nine key-file hashes
are preserved in `delete_manifest.json`. The clean PCD and VLA-Arena audit
clones likewise retain their exact remote URLs and Git revisions.

## Concurrent-Goal safety refresh

Before deletion, branch, local/remote HEAD, ancestry, Git status, candidate
mtimes, high-risk hashes, workers, GPU processes, and open handles were
rechecked. No VLA, LIBERO, MuJoCo, CUDA, training, rollout, or download worker
remained. No deletion candidate had a write in the conservative overlap
window. All latest Epoch 6 reports, source, tests, and run output were assigned
`KEEP_OR_REVIEW` and excluded.

Windows checked every regular target file with exclusive `FileShare.None`.
WSL scanned `/proc/*/fd`; both handle checks passed. Both deletion executors
then repeated their checks immediately before deleting.

## Preservation result

The retained runnable path is:

- model: `2toINF/X-VLA-Libero` at revision
  `129e71460678b7236cee6fc9707f09d9fa0c3590`;
- snapshot: `/home/jiheon/assets/checkpoints/xvla_hf_cache/transformers/models--2toINF--X-VLA-Libero/snapshots/129e71460678b7236cee6fc9707f09d9fa0c3590`;
- source: `C:\assets\repos\X-VLA` at
  `6bc2513f5f1cbec715cc668b414392a6cae5c671`;
- environment: `/home/jiheon/miniconda3-official/envs/official-smolvla-libero`;
- LIBERO runtime: `/home/jiheon/assets/repos/LIBERO`;
- benchmark data: `C:\assets\data\libero`.

All eight snapshot links resolve to retained blobs, including the 3.519 GB
weight blob, and every blob was SHA-256 verified after cleanup. Offline
`AutoConfig` resolved as `XVLAConfig` without loading model weights. A new VLA
topic using this existing X-VLA/LIBERO path can therefore start without a
model download; no new thesis search was started here.

The explicitly protected rollout directories remain present:

- `rollouts/2026_07_17`: 5.144 MB;
- `rollouts/2026_07_18`: 0.925 MB.

Large items intentionally left for later review include the modified LightVLA
checkpoint (15.455 GB), non-immutable OpenPI checkpoint (12.439 GB), all unique
scientific `runs` (10.311 GB), ambiguous Remote Desktop diagnostics
(10.196 GB), and referenced Windows checkpoints (3.758 GB).

## VHDX compaction pending

The exact verified VHDX is
`C:\Users\jiheo\AppData\Local\wsl\{180ce51e-73f9-4a6b-91f0-3a4f1842ad61}\ext4.vhdx`.
WSL 2.7.3 exposes `--set-sparse`, but it refused this distribution unless
`--allow-unsafe` was supplied. That unsafe flag was not used. This session is
not elevated and `Optimize-VHD` is unavailable, so compaction stopped safely.

An administrator may later run `wsl --shutdown`, reverify that exact path, and
use elevated DiskPart only against it:

```text
select vdisk file="C:\Users\jiheo\AppData\Local\wsl\{180ce51e-73f9-4a6b-91f0-3a4f1842ad61}\ext4.vhdx"
attach vdisk readonly
compact vdisk
detach vdisk
exit
```

Do not interrupt compaction, use `--allow-unsafe`, or unregister the distro.
The newly freed 32.190 GB inside WSL is the practical pending host-reclaim
amount.

## Verification

- 22 focused Epoch 6 tests passed.
- Current research governance and scaffold-tree checks passed.
- Cleanup JSON parsing and `git diff --check` passed.
- X-VLA source remained clean at its pinned HEAD.
- All protected, benchmark, environment, snapshot, and blob paths remained.
- No full model load, simulator episode, training, rollout, or download ran.

## Research transition

Epoch 6 schedule-invariance evidence is archived and resumable, not disproven.
Four-shard closed-loop outcomes remain unobserved (`0`). The original route can
resume on a clean host with at least 48 GB RAM. A later research epoch must
select a different VLA/robotics thesis through a fresh literature, artifact,
feasibility, and paperability audit; no failed old route may be silently
renamed as the new topic.
