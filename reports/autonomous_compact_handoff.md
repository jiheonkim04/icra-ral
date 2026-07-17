# Autonomous Compact Handoff

Updated: 2026-07-17 KST

## Current State

- Branch: `codex/epoch5-official-prior-first`
- Current epoch/cycle: `5 / 0`
- Current stage: `epoch_5_xvla_task6_ours_candidate_selected`
- Current decision: `TASK6_MPR_XVLA_SELECTED_AFTER_SECOND_PRIOR_RESIDUAL_SURVIVED`
- Latest pushed source commits in this segment:
  - `b90b26b` record BR-XVLA closed-loop no-pass and launcher escaping fix
  - `62713d5` add detached X-VLA prior failure-scan launcher
  - `5a90911` repair X-VLA optional import boundary
  - `5835ef3` repair X-VLA runner path import
  - `19177fe` record repaired post-BR-XVLA X-VLA prior scan
  - `e1d67fb` record task6 matched residual headroom
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
- Task-6 now has matched Base/Prior residual structure, task-level headroom,
  spatial data-health, OpenVLA-OFT INT4 second-prior no-solve evidence, and a
  selected candidate design.

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
| 6 | false | 900 | fresh prior failure; matched task-6 diagnostic now complete |
| 7 | true | 270 | prior succeeds |
| 8 | true | 368 | prior succeeds |
| 9 | true | 244 | prior succeeds |

Invalid attempts to preserve:

- `...post_brxvla_20260717T2010KST`: invalid infrastructure block, `No module named 'fastapi'`.
- `...post_brxvla_repaired_20260717T2018KST`: invalid infrastructure block, `No module named 'tca_map'`.

## Task-6 Matched Base/Prior + Headroom

Task: `libero_10/task_6`, identities `20260724..20260731`, indices `13..20`.

Matched window:

- X-VLA prior: 6/8, failures `20260725`, `20260731`.
- X-VLA result: `runs/xvla_prior/diagnostic_xvla_libero10_task6_id20260724_20260731_20260717T2043KST/result.json`
- X-VLA result SHA-256: `d18356bf1a18e4f2053596142d9af13983ffc1ed0ccc74fa525ad4d802ac25aa`
- SmolVLA base: 3/8, failures `20260724`, `20260725`, `20260727`, `20260730`, `20260731`.
- Base result: `runs/xvla_prior/diagnostic_smolvla_base_libero10_task6_id20260724_20260731_officialenv_20260717T2047KST/result.json`
- Base result SHA-256: `749fbc0f25f075902de9e2172c602e99cde020d4b4be735accedbb80c45556c8`
- Base manifest SHA-256: `19733d8a5490350beba7d4444810e73c90af48c47e21471dca2b5257e0874f89`
- Shared residuals: `20260725`, `20260731`.
- X-VLA-only successes: `20260724`, `20260727`, `20260730`.
- No training, optimizer step, checkpoint, Ours design, or Ours rollout happened.

Headroom script:
`scripts/epoch5_expert_headroom.py`

- Script SHA-256: `7339d16a9b70665064b437eb7d007d81f6bc99246f0fe28a46b2e33ee321b8b0`
- `20260725`: positive nearest-demo expert replay, selected `demo_24`, first success 235, same-reset HDF5 matches 0.
- `20260725` artifact SHA-256: `68b61e5802f6d672d44ab58ee26170cad724fce6c8cc4870065e2b4b2dc7cccd`
- `20260731`: positive nearest-demo expert replay, selected `demo_6`, first success 217, same-reset HDF5 matches 0.
- `20260731` artifact SHA-256: `5dac493d0443bb1237b69ca0c0d5c69b2a00259c697de39fe2364550b9d9f49d`
- Zero-action and default-reset expert replay controls failed for both.
- Interpretation: recoverable task-level headroom exists, but same-reset HDF5 headroom is unavailable.

Invalid launcher caveat:

- `runs/xvla_prior/diagnostic_xvla_libero10_task6_id20260724_20260731_20260717T2040KST`: invalid launcher no-result/no-rollout due WSL background session teardown.

## Task-6 Data Audit, Second-Prior Screen, and Candidate

Spatial data audit:
`runs/xvla_prior/diagnostic_task6_spatial_data_audit_20260717T2115KST/result.json`

- Result SHA-256: `71178809c5290ae6b4083e34fdf3aa49a4b259bb42f26b2561628acaeb3800fd`
- Source/test SHA-256:
  `0accb6887839178fca565b18d8a691ed78fa2b515b90f8cf5d986085e1b779c8` /
  `6486a6a83bc2c6354a7e64a962a55c505b48ccf45bf5f798c91a6d1a43bb1155`
- 50 demos, 12,756 steps, 12,406 chunks; train/val mug-done-pudding-remaining
  chunks 5,518 / 1,372.
- All demos mug-first; red mug stays off-plate; residual init overlap 0.
- Privileged simulator state is training-label-only; inference remains X-VLA
  RGB/proprio/instruction.

Quantized OpenVLA-OFT INT4 second-prior screen:
`runs/openvla_oft_int4/diagnostic_task6_residual_openvla_int4_20260725_20260731_openvlaenv_20260717T2114KST/result.json`

- Result SHA-256: `c897000b299d2d8fd356bb467a574971dd8d11843c0d06ecdd7698d765cd233b`
- Runtime: `/home/jiheon/venvs/openvla-oft-int4-rtx5080/bin/python`
- Completed 2/2, successes 0/2, infra failures 0, elapsed 208.769s.
- `20260725`: false, 530 steps, reward 0.0.
- `20260731`: false, 530 steps, reward 0.0.
- Invalid wrong-runtime attempts:
  - `...20260717T2130KST`: missing `json_numpy`, no result.
  - `...repaired_20260717T2135KST`: missing TensorFlow, no result.

Selected task6 candidate:
`reports/epoch5_task6_ours_candidate_design.md`

- Exactly two candidates generated: `MPR-XVLA` selected, `PRC-XVLA` not selected.
- `MPR-XVLA`: Mug-placed / Pudding-right Reweighted X-VLA.
- Core objective: upweight HDF5 chunks where mug-on-plate is true and
  pudding-right is false; LoRA/QLoRA only as infrastructure.
- Mandatory first spec arms: primary `MPR-XVLA` and uniform-weight X-VLA
  LoRA/QLoRA ablation.
- No task6 optimizer step, checkpoint, training, or closed-loop Ours evaluation
  has happened.

## Immediate Next Gate

Freeze a no-training task6 `MPR-XVLA` training spec with exactly two arms:
primary `MPR-XVLA` and uniform-weight X-VLA LoRA/QLoRA ablation.

Still prohibited: BR-XVLA rescue/retune, broad search, generic local heads,
residual gates, memory, verifiers, cached-feature probes, proxy-only methods,
and closed-loop Ours evaluation before an offline gate.

## Report Set

- Ecosystem selection: `reports/epoch5_prior_ecosystem_selection.md`
- Reproduction plan: `reports/epoch5_prior_reproduction_plan.md`
- Reproduction result: `reports/epoch5_prior_reproduction_result.md`
- Reproduction result JSON: `reports/epoch5_prior_reproduction_result.json`
- Task-1 candidate design: `reports/epoch5_task1_ours_candidate_design.md`
- Task-6 candidate design: `reports/epoch5_task6_ours_candidate_design.md`

## Current Validation Status

- JSON parse: pass via `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m json.tool`
- Handoff line count: 210, under the 250-line cap.
- Task-6 candidate-design validation:
  - JSON parse: pass via `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m json.tool`
  - py_compile: pass for OpenVLA gate, task6 data audit, and focused tests
  - pytest: `8 passed`
  - `git diff --check`: pass with LF/CRLF warnings only
  - `scripts/99_tree_check.ps1`: pass
- Scan launcher syntax: pass via WSL `bash -n`
- X-VLA runner py-compile: pass via official WSL env
- Focused scan tests: none found
- `git diff --check`: pass with LF/CRLF warnings only
- `scripts/99_tree_check.ps1`: pass via one-shot PowerShell execution-policy bypass

Do not add `rollouts/2026_07_17/` or ignored run directories.
