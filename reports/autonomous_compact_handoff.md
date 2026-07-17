# Autonomous Compact Handoff

Updated: 2026-07-17 KST

## Current State

- Branch: `codex/epoch5-official-prior-first`
- Current epoch/cycle: `5 / 0`
- Current stage: `epoch_5_xvla_prior_residual_mining_after_br_xvla_no_pass_complete`
- Current decision: `X_VLA_POST_BRXVLA_RESIDUAL_SCAN_FOUND_FAILURES_BASE_MATCH_PENDING`
- Latest pushed source commits in this segment:
  - `b90b26b` record BR-XVLA closed-loop no-pass and launcher escaping fix
  - `62713d5` add detached X-VLA prior failure-scan launcher
  - `5a90911` repair X-VLA optional import boundary
  - `5835ef3` repair X-VLA runner path import
- Audit report: `reports/autonomous_research_full_history_audit.md`
- Audit accepted as evidence; the embedded Cycle 39 prompt is not active.
- Paper status: no PROTOTYPE_GO, no official-prior Ours win, no second-backbone Ours result.

## Standing Governance

- Official-prior-first remains active.
- Do not rescue or retune MCI/CSPR/R2R/CR-LightVLA/ATCD/BR-XVLA.
- Do not generate generic local heads, residual gates, memory, verifiers, cached-feature probes, or proxy-only methods.
- LoRA/QLoRA is infrastructure only, not the contribution.
- Do not treat prior success, prior failure, uniform-ablation success, or diagnostic headroom as Ours.
- Do not design Ours until matched Base/Prior residual and recoverable headroom are established.
- Keep this file under 250 lines.

## Prior Sequence Summary

OpenVLA-OFT INT4:

- Hard slice: OpenVLA-OFT INT4 20/20 vs SmolVLA base 11/20, saturated.
- Residual diagnostic: OpenVLA-OFT INT4 14/16 vs SmolVLA base 7/16.
- R2R-OFT was designed after base/prior/headroom gates, trained under a frozen two-arm spec, and failed offline selection. No closed-loop Ours GO.

LightVLA:

- Official LightVLA-Libero10 checkpoint loaded and ran locally.
- It was complementary on task 8 but did not produce a usable Ours route.
- CR-LightVLA Stage 0 and ATCD teacher-signal audit closed. Do not rescue.

X-VLA:

- Official X-VLA-Libero loaded from cached HF assets.
- Task-8 residual was solved by X-VLA; no Ours target there.
- Task-1 shared residual led to BR-XVLA; BR-XVLA is now closed as a validation no-pass.

## BR-XVLA Closed-Loop No-Pass

- Scope: `libero_10/task_1`, identity `20260727`, initial-state index `16`.
- Frozen manifest: `runs/xvla_prior/epoch5_br_xvla_closed_loop_residual_20260727/closed_loop_manifest.json`
- Manifest SHA-256: `ea222a6014e2cda6a8f7428bdf2d0f0105e1773e0f7a0c6ba3ce5bb74f01dc63`
- Result: `runs/xvla_prior/epoch5_br_xvla_closed_loop_residual_20260727/closed_loop_result.json`
- Result SHA-256: `472904b03472c8b1017aad2080c57e49c0b1064816b430670051330dd970b64f`
- Policies:
  - same-run X-VLA prior: failed, 900 steps, reward 0.0
  - BR-XVLA primary: failed, 900 steps, reward 0.0
  - uniform ablation: succeeded, 479 steps, reward 1.0
- Interpretation: selected BR weighting failed while key ablation solved the identity. Archive BR-XVLA; no retune.

## Post-BR-XVLA X-VLA Prior Scan

Completed repaired scan:
`runs/xvla_prior/failure_scan_libero10_identity20260725_post_brxvla_repaired2_20260717T2022KST`

- Purpose: official-prior residual mining only.
- Commit: `5835ef3bafad1027e9e4ed6dcf5943383d2a9714`
- Policy/suite: `X-VLA-Libero` / `libero_10`
- Reset identity: `20260725`, initial-state index `14`
- Tasks: `0..9`
- Horizon/settle/denoise: `900 / 10 / 10`
- Exit code: `0`
- Finished: `2026-07-17T20:22:49+09:00`
- Manifest SHA-256: `5adbc60144dde3f49a1c8cd82f5bcdc2f82c184447d5fb799843a0fbeef3eacc`
- Summary SHA-256: `c2ff073b74efb5e9af9db0bc6254aaa9dd735aaaf0c6635fcf93dfe35d07a16a`
- Training/optimizer/checkpoint/Ours design/closed-loop Ours: false/false/false/false/false

Results:

| Task | Success | Steps | Note |
|---:|---:|---:|---|
| 0 | true | 273 | prior succeeds |
| 1 | false | 900 | known X-VLA regression; SmolVLA base previously solved identity `20260725` |
| 2 | true | 238 | prior succeeds |
| 3 | true | 221 | prior succeeds |
| 4 | true | 219 | prior succeeds |
| 5 | true | 180 | prior succeeds |
| 6 | false | 900 | fresh prior failure; base/headroom pending |
| 7 | true | 270 | prior succeeds |
| 8 | true | 368 | prior succeeds |
| 9 | true | 244 | prior succeeds |

Invalid attempts to preserve:

- `...post_brxvla_20260717T2010KST`: invalid infrastructure block, `No module named 'fastapi'`.
- `...post_brxvla_repaired_20260717T2018KST`: invalid infrastructure block, `No module named 'tca_map'`.

## Immediate Next Gate

Run a matched Base/Prior diagnostic for the fresh task-6 identity-`20260725`
X-VLA failure, or record a no-go if the structure is unusable.

Required before Ours:

- same task/reset semantics;
- unmodified Base and selected prior;
- no training or retuning from BR-XVLA;
- confirm condition is not floor/saturated;
- run recoverable headroom if base/prior residual structure is usable.

If task 6 lacks Base/Prior/headroom structure, broaden residual mining under
official-prior-first rather than inventing a local proxy.

## Report Set

- Ecosystem selection: `reports/epoch5_prior_ecosystem_selection.md`
- Reproduction plan: `reports/epoch5_prior_reproduction_plan.md`
- Reproduction result: `reports/epoch5_prior_reproduction_result.md`
- Reproduction result JSON: `reports/epoch5_prior_reproduction_result.json`
- Task-1 candidate design: `reports/epoch5_task1_ours_candidate_design.md`

## Current Validation Status

- JSON parse: pass via `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m json.tool`
- Handoff line count: 134, under the 250-line cap
- Scan launcher syntax: pass via WSL `bash -n`
- X-VLA runner py-compile: pass via official WSL env
- Focused scan tests: none found
- `git diff --check`: pass with LF/CRLF warnings only
- `scripts/99_tree_check.ps1`: pass via one-shot PowerShell execution-policy bypass

Do not add `rollouts/2026_07_17/` or ignored run directories.
