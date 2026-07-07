# TL-ChunkRepair Kill Summary

## Original TL-ChunkRepair Hypothesis

Temporal-Logic-Guided Action Chunk Repair should inspect an already proposed 7D manipulation action chunk, monitor temporal manipulation properties, identify the causal violation boundary, and minimally patch the chunk before execution. The route targeted violations such as grasp-before-lift, keep-grasp-until-placement, no early release, no open-gripper transport, unsafe-contact ordering, and mechanism onset order.

The intended RA-L-stable claim would require replay/control utility beyond clipping-only, safety-only filtering, gripper-only timing fixes, fixed delay/shift, linear time warp, abort-to-stop, repeat-last/hold, and no-repair baselines.

## Strongest Positive Evidence

- STATE 1 produced real bounded LIBERO/RoboSuite exact-init replay/control metrics.
- Exact-init expert replay infrastructure worked on the selected LIBERO task.
- The diagnostic ran `73` variants and `19803` simulator steps.
- Eight temporal perturbations were tested.
- Seven of eight perturbations degraded replay.
- TL-ChunkRepair reduced symbolic temporal violations on `8 / 8` perturbations.
- The temporal property monitor, perturbation runner, baseline table, and replay harness all produced usable diagnostic evidence.

## Decisive Negative Evidence

- TL safe-success was `0 / 8`.
- TL reward/success was `0.0 / 0`.
- The best single simple baseline was `no_repair`, with reward/success `1.0 / 1`.
- TL-ChunkRepair did not beat the best single simple baseline.
- TL-ChunkRepair did not beat the best per-failure-mode simple baseline.
- The improvement was symbolic/property-level and did not translate into reward, success, done index, safe-success, or enough replay/control utility.

## Exact Kill Criteria Triggered

Triggered hard kill criteria:
- TL-ChunkRepair did not beat the best single simple baseline.
- TL-ChunkRepair did not beat the best per-failure-mode simple baseline.
- TL-ChunkRepair improved symbolic temporal-property scores without improving real replay/control utility.
- Utility cost was unacceptable because a simpler baseline produced nonzero reward/success while TL-ChunkRepair produced none.

## Why Symbolic Violation Reduction Alone Is Not Enough

Temporal property satisfaction is only useful for this route if it preserves or improves robot execution. A repair can make a chunk look temporally valid by holding, delaying, or suppressing actions while still destroying task progress. In STATE 1, TL-ChunkRepair made the monitor happy on all perturbations but produced zero safe-success, zero success, and zero reward. For RA-L-stable continuation, safety/property constraints must be tied to real replay/control outcomes.

## Which Simple Baseline Killed It

The decisive best single simple baseline was `no_repair`, which achieved aggregate reward/success `1.0 / 1` while TL-ChunkRepair achieved `0.0 / 0`.

Additional weakening baselines included `clipping_only`, `safety_only_one_step_filter`, and `repeat_last_hold`, each with aggregate reward `1.0`, plus per-perturbation simple timing baselines such as `fixed_delay_shift`.

## Reusable Artifacts

- temporal perturbation runner in `tca_map.tl_chunkrepair.state1_diagnostic`,
- gated exact-init replay diagnostic script: `scripts\181_tl_chunkrepair_state1_diagnostic.ps1`,
- finite-state temporal property monitor,
- violation metrics and per-property summaries,
- baseline suite for no-repair, clipping, safety-only filter, gripper timing, fixed shift, linear time warp, abort-to-stop, repeat-last/hold, and TL repair,
- compact result report: `reports\tl_chunkrepair_state1_result.md` and `.json`,
- focused tests: `tests\test_tl_chunkrepair_state1.py`.

## Why Not Continue As RA-L-Stable

TL-ChunkRepair found a real observable safety/property axis, but the method failed the route's own hard gate. The route is not RA-L-stable because the central contribution reduced symbolic violations while losing to a simple baseline on real replay/control utility. Continuing would reward the exact failure pattern this repository now excludes: proxy or symbolic improvement without a best-simple-baseline win on robot execution.
