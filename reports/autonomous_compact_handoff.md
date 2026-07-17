# Autonomous Compact Handoff

Updated: 2026-07-17 KST

## Current State

- Branch: `codex/epoch5-official-prior-first`
- Current epoch/cycle: `5 / 0`
- Current stage: `epoch_5_post_mpr_xvla_identity_grid_no_fresh_target`
- Current decision: `POST_MPR_XVLA_IDENTITY_GRID_NO_FRESH_TARGET`
- Latest pushed source commits in this segment:
  - `b90b26b` record BR-XVLA closed-loop no-pass and launcher escaping fix
  - `62713d5` add detached X-VLA prior failure-scan launcher
  - `5a90911` repair X-VLA optional import boundary
  - `5835ef3` repair X-VLA runner path import
  - `19177fe` record repaired post-BR-XVLA X-VLA prior scan
  - `e1d67fb` record task6 matched residual headroom
  - `d387d8a` record task6 second-prior candidate design
  - `4cdb49f` freeze task6 MPR-XVLA training spec
  - `f5efa5c` pass task6 MPR-XVLA preoptimizer gates
  - `58a97d5` add task6 MPR-XVLA training gate
- `5faed4d` fix task6 MPR-XVLA offline validation spec scope
  - `cc03266` add foreground X-VLA prior scan worker
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
- Task-6 has matched Base/Prior residual, task-level headroom, spatial data
  health, OpenVLA-OFT INT4 no-solve evidence, and MPR-XVLA offline no-pass;
  PRC-XVLA was not elevated due lack of red-mug confusion evidence.

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

MPR-XVLA candidate/training:

- Design: `reports/epoch5_task6_ours_candidate_design.md`; exactly two
  candidates, `MPR-XVLA` selected and `PRC-XVLA` not selected.
- Frozen spec: `runs/xvla_prior/epoch5_mpr_xvla_training_spec_v1.json`
  (`5ee2b5d49887d187f5da81cb0d14d0e48feaab2e46dc8ff1ac65bd671808cc98`).
- Preoptimizer smokes passed:
  data adapter `62668c2483ab060aaa1a8e1f5d6153dd4f220fbd331658f587acb38003625ee8`;
  gradient `97b6bbc9a9cd1a2e0e471587196b8f3ad11b5e958f07d66f3eaa6dae60dad552`.
- One-step two-arm debug gate passed:
  `272b30f25528ffa9925f71e5fdde1ef09b03bac6d69fe3a6f0731ab40fc94f0e`.
- Full 64-step two-arm training completed and wrote checkpoints:
  primary result `168b1720b8911f77700a78057de21d8aad3f08a55ccac7384589f02bf946dcb0`;
  uniform result `918bbf6c4d1e941b6827b28cc314f91a72982501399b48ce46722601c39d48c7`.
- Initial full gate failed only during offline validation due
  `NameError: spec`; invalid artifact
  `b8ec1ac9fb739877f73f53e76eac8ef00ebf3899f6f77823ceade2f57164fa97`.
- Repaired offline validation artifact:
  `runs/xvla_prior/epoch5_mpr_xvla_offline_validation_step0064_repaired_20260717T2200KST.json`
  (`ede498006a5832e3b2101de41fd344438b1c2dc4cdbd21f1355d8564e03fc59f`).
- Offline metrics on 24 chunks: prior phase-1 loss 2.999218; MPR 0.878535892;
  uniform 0.878535837. MPR passed absolute health but did not beat uniform.
- Closed-loop MPR-XVLA Ours evaluation did not happen and is disallowed.

## Immediate Next Decision

Post-MPR scans: X-VLA `libero_10` grid exhausted for fresh targets
(`20260731` only archived task6; `20260726..30` saturated). `libero_goal`
and `libero_object` identity `20260724` saturated 10/10 (summary SHAs
`77065303dfecc5fd170d9ca00fae1dfa95ea2fb1f87143eab1b5663bc94281f4`,
`5c521e9f229a5e046e06d389e33bd72888e354a4e98e794bff3c7c10097c2808`).

Fresh residual: `libero_spatial`, identity `20260724`, task 5 failed:
`pick up the black bowl on the ramekin and place it on the plate`, 900 steps,
reward 0.0, result SHA
`847317ad60f499dddc3d8f372a47281031f92c8e68137429ca46a220d65207ba`.
Spatial summary SHA `596ecd11212f4de4019b171b2809b0242b1d8e734ec9807fe45cb0ab176ec4fd`;
manifest SHA `4c48f9bd4ddafa43879d3524d7506e353a22649f087815a23cb3a5af74c989a8`.
Task 7 stalled >15min, was SIGTERM'd, exit 143, no result; classify as infra,
not policy failure. No cross-suite scan trained, optimized, checkpointed, or
evaluated Ours.

Matched diagnostics: SmolVLA base full scan on same spatial identity is 3/10
(`322cd732b2c2aec3e1dec1e56918fc073318c13ce425c320c47073641ffea8c9`);
focused base task5 is 0/1, reward 0.0
(`064a5819755df6aed742a57a666e784f3985f35909b55a2b692ceb798a4ce5db`).
On comparable tasks excluding X-VLA infra-stalled task7, X-VLA is 8/9 vs base
3/9; shared residual is task5. Expert headroom is task-level positive but
same-reset HDF5 unavailable: nearest `demo_9`, L2 2.984425805, SHA
`4b1107cdeda0044cf53bb0b3656c3b52ca516c6c52b8a7ac1ae991bfb1d0ebdc`.
Quantized OpenVLA-OFT INT4 solves exact task5 residual: 1/1 success, 136
steps, reward 1.0, SHA
`6da4ea2e072b0a227d19dc25aa18c1f0a61ad22602be22a68f5d59a3f23740e4`.
Post-second-prior `libero_90` identity `20260724` saturated tasks 0..49:
0..9 summary `2b0fb345a88dd7fbe0baab488cfed35a924e8e54e7291144e8df0d488825af5f`;
10..29 summary `7616acc53feffd88c9dc342781fab162db1938b34d97924afef4d1b3994452ed`
(`tasks10_19` label actually task_count=20); 30..49 summary
`9ee57921eb488b677c826431628a2f87093bf1454f756b918296bb17268f15de`.

Do not design Ours for spatial task5. Do not retune/rescue MPR-XVLA. Next:
select a new preregistered residual source or prior ecosystem.

## Report Set

- Ecosystem selection: `reports/epoch5_prior_ecosystem_selection.md`
- Reproduction plan: `reports/epoch5_prior_reproduction_plan.md`
- Reproduction result: `reports/epoch5_prior_reproduction_result.md`
- Reproduction result JSON: `reports/epoch5_prior_reproduction_result.json`
- Task-1 candidate design: `reports/epoch5_task1_ours_candidate_design.md`
- Task-6 candidate design: `reports/epoch5_task6_ours_candidate_design.md`

## Current Validation Status

- Repaired offline fix validation before commit `5faed4d`:
  - py_compile: pass for `offline_validate.py` and `test_mpr_xvla_offline_validate.py`
  - pytest: `5 passed` for offline validator + training gate tests
  - `git diff --check`: pass with LF/CRLF warnings only
  - `scripts/99_tree_check.ps1`: pass
- Repaired offline run completed under official WSL env in 50.325s.
Do not add `rollouts/2026_07_17/` or ignored run directories.
