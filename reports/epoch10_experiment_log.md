# Epoch 10 execution log

Date: 2026-07-21 KST  
Branch: `codex/epoch10-icae-vla-evaluation`

## Completed phases

1. Verified the authoritative Epoch 9E commit `2463ef93f9326fb780f7c053fe40c0051aca901c`, created the required branch, audited the untracked rollout trees without modifying them, and recorded host/WSL/GPU resources.
2. Preserved the exact Epoch 9E negative scope boundary and completed the required exact-query novelty audit through 2026-07-21. No exact prior collision was found.
3. Inventoried the frozen model, dataset, LIBERO simulator, and historical checkpoint identities. The existing four identities were insufficient for the prospective panel.
4. Trained four new standard rank-4 LoRA lineages (`101`, `202`, `303`, `404`) serially for 100 optimizer steps with snapshots at `10`, `30`, and `100`. All 12 adapter bundles were saved and freshly reloaded; generation took 204.0 seconds and opened no simulator outcome.
5. Froze four tasks across all four LIBERO suites, whole-demo state splits, whole-seed checkpoint splits, and disjoint development/official closed-loop initial-state indices.
6. Exact-state preflight Attempt 1 restored states but failed all 128 audit rows because native goal metadata was requested from the outer wrapper. The invalid file is retained.
7. Applied the sole repair: use the inner LIBERO task for native goal predicates. Attempt 2 reran all 128 rows in 34.832 seconds; all state restores had L2 exactly zero, all goal operands were accessible, and both camera streams regenerated at 256×256.
8. Ran the complete checkpoint-free mechanics panel: 32 states × 3 horizons, with nominal reference, independently restored nominal sham, small perturbation, and medium perturbation branches in SHA-256-fixed randomized order. The 96 registered horizon rows required 221.96 seconds.
9. Adjudicated all three horizons as failures and stopped before checkpoint simulator ranking or closed-loop outcomes, as required by the frozen assay gate.

## Commands

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_epoch10_checkpoint_panel.py
wsl.exe -e bash -lc 'cd /mnt/c/Users/jiheo/tca_map && export MUJOCO_GL=egl && /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/preflight_epoch10_icae_exact_states.py'
wsl.exe -e bash -lc 'cd /mnt/c/Users/jiheo/tca_map && export MUJOCO_GL=egl && /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_epoch10_icae_mechanics.py'
```

## Outcome-access ledger

- Historical Epoch 9E and earlier official outcomes: known only as documented development context.
- New seed `101/202` simulator outcomes: never opened.
- New seed `303/404` official held-out outcomes: never opened.
- Mechanics expert demonstrations and native goal predicates: used only to validate the assay.
- Synthetic perturbations: used only as mechanics controls, never checkpoint identities or ranking evidence.
- Positive paper: not authorized and not written.

## Resource and integrity notes

Training and simulator jobs were serialized. Peak checkpoint-training host RAM was 51.0%; peak CUDA allocation was 1105.569 MiB. The mechanics worker remained within the frozen WSL environment and no resource stop occurred. The pre-existing untracked `rollouts/2026_07_17/` and `rollouts/2026_07_18/` trees remain untracked and were not added, deleted, or edited by the Epoch 10 commits.

## Final verification

- Epoch 10 JSON parsing and 24-file evidence-index hash verification: pass.
- External adapter hash verification: 12/12 pass.
- Python compilation for all three Epoch 10 runners: pass.
- Repository governance checker: pass.
- Governance tests excluding one demonstrably stale assertion: 5 passed, 1 deselected.
- Full governance-test file: 5 passed, 1 failed. The failure asserts `current_epoch == 4`; both the untouched authoritative starting commit `2463ef9` and the current file store `current_epoch == 5`, so it predates Epoch 10 and was not rewritten to manufacture a green test.
- `git diff --check`: pass.
- Protected untracked rollout trees: `27 / 5,143,751 bytes` and `10 / 924,633 bytes`, exactly matching the starting inventory.
