# Epoch 8 Change-Scope Justification

Scope frozen at: 2026-07-20T21:31:40+09:00

This checkpoint intentionally contains 70 new Epoch 8 files, approximately
32,851 inserted lines, and 5,494,570 bytes. It exceeds the repository's 50-file
and 5,000-line review thresholds because the unit of work is one complete,
auditable research campaign rather than a source-only feature:

- frozen protocols, split manifests, scripts, raw machine-readable outcomes,
  and human adjudications must travel together so every decision can be
  reproduced and audited;
- four approximately 1 MB PCAT adapter checkpoints are retained because their
  hashes and exact learned states are part of the valid Stage 0 evidence;
- both the invalid and repaired resource/probe attempts are preserved so the
  repair boundary is reviewable rather than overwritten;
- generated ledger and split JSON account for most of the text-line volume and
  are required inputs to the terminal evidence index.

The scope is limited to `reports/epoch8*` and `scripts/*epoch8*`. The untracked
protected evidence directories `rollouts/2026_07_17/` and
`rollouts/2026_07_18/` are explicitly excluded from staging. No inherited
tracked file is modified by this checkpoint.

Verification before staging:

- all 12 Epoch 8 Python scripts compile;
- all 39 Epoch 8 JSON artifacts parse;
- the current-research governance checker passes;
- 25 relevant governance/Epoch 6/Epoch 7 tests pass;
- the complete targeted command has one inherited stale assertion: the source
  commit and worktree both record Epoch 5/cycle 0, while
  `test_active_state_records_amp_selection_and_rap_stage_0_failure` still
  expects Epoch 4/cycle 39. Epoch 8 does not modify either file.
- protected rollout inventories remain 27 files / 5,143,751 bytes and 10
  files / 924,633 bytes, matching their recorded baselines.
