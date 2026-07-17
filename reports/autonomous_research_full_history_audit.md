# Autonomous VLA Research — Full History Audit

This is the mandatory Phase A audit before any resumed VLA research. It refreshes the earlier audit at `b0ecb6e`, the BR-XVLA audit refresh at `330230f`, and the current pushed Epoch 5 state at `cd3853285a5dfabdee1ab21524392acf9ad2bc64`, plus ignored/local task75 artifacts that existed before this audit pause. No new training, rollout, or second-prior screen was launched during this Phase A run. Evidence precedence: current local git/HEAD, `reports/current_research_governance.md`, current result artifacts, current campaign-state JSON, git history across branches, project state / next actions / decision logs, historical reports, then old prompts. Missing facts are recorded as `NOT_RECORDED`.

## 1. Executive Summary

No paper-ready method exists. No valid `PROTOTYPE_GO` method exists. The repository contains substantial reusable infrastructure, many valid scientific kills or bounded no-pass closures, many invalid or pre-rollout stops, and a much stronger official-prior-first diagnostic trail, but it does not contain `READY_TO_DRAFT_RAL_PAPER_PACKAGE`.

This audit finds 95 distinct research routes or diagnostic prior routes: the prior 89-route ledger plus post-BR-XVLA X-VLA task6 residual mining, selected `MPR-XVLA`, unselected `PRC-XVLA`, X-VLA cross-suite/spatial task5 diagnostics, X-VLA LIBERO-90 tasks81/83 diagnostics, and the ignored/local task75 diagnostic thread. Formal selected Ours methods are 50 after adding selected `MPR-XVLA` to the previous 49-method count. Implemented route count is 84/95 using code, runner, or local execution evidence. Trained/checkpointed route count is 34 after BR-XVLA and MPR-XVLA wrote bounded adapters. Closed-loop Stage A count remains 17 formal autonomous methods, or 19 route-level methods when historical prototypes are included; BR-XVLA had a one-identity closed-loop residual-manifest screen, not a full Stage A. Stage B count remains 10 formal methods, or 11 route-level methods with the repaired PhaseBarrier prototype. Second-backbone Ours count remains 0.

Outcome totals: 28 valid scientific kills or bounded no-pass closures, 43 non-scientific failures or resource/preimplementation blockers, 13 underpowered/unresolved/unfinished results, and 11 diagnostic/no-claim rows. The decisive changes since the BR-XVLA audit refresh are: BR-XVLA passed gradient smoke, bounded training, and offline validation but failed closed-loop because the selected primary failed while the uniform LoRA ablation succeeded; task6 became a matched X-VLA/Base residual with task-level headroom and OpenVLA-OFT INT4 second-prior no-solve evidence; `MPR-XVLA` was selected, trained, and stopped because it did not beat its uniform-weight ablation in repaired offline validation; `PRC-XVLA` was not elevated because no independent red-mug/distractor confusion evidence appeared; spatial task5 was solved by Quantized OpenVLA-OFT INT4 and is not an Ours target; LIBERO-90 tasks81/83 were shared X-VLA/Base residuals but failed the headroom gate; and local task75 is unfinished because its second-prior screen had not run when the audit pause started.

The strongest Ours result remains `CAVM-VLA`: full 24/58 versus nearest-success memory 23/58, Base 22/58, and no-contrast 21/58 after one allowed expansion. It is a near-miss, not paper-ready: the advantage is one episode, no third expansion is allowed, and there is no second-backbone or official-prior confirmation. The strongest official-prior results are X-VLA solving the earlier task8 residual 8/8 and OpenVLA-OFT INT4 solving spatial task5 1/1; both remove Ours targets rather than creating a paper method.

Current active state: Phase A audit pause on branch `codex/epoch5-official-prior-first` at pushed HEAD `cd38532`. Pushed reports say no Ours design is authorized after LIBERO-90 tasks81/83 failed headroom; the ignored/local task75 thread found one new shared X-VLA/Base residual with task-level headroom but has not completed the required second-prior screen. Previous selected Ours methods `BR-XVLA` and `MPR-XVLA` are no-pass and must not be retuned. The next scientific action, after user review only, is to either finish the already-started task75 second-prior screen or choose a new residual source/identity/prior ecosystem; no Ours design is authorized until the prior/Base/headroom/second-prior gates are complete.

Main reasons the campaign is not paper-ready: no Ours method beats Base, closest prior/proxy, key ablation, and simple reviewer-killer control in a valid Stage B; official-prior comparison arrived late and is still diagnostic; many late routes failed from data/headroom/objective-scale/resource issues before rollout; the search repeatedly favored small frozen-SmolVLA attachments; and no same-method Ours evidence exists on Quantized OpenVLA-OFT INT4 or another second backbone.

## 2. Audit Snapshot

| Field | Value |
|---|---|
| Snapshot timestamp | `2026-07-18T00:48:48+09:00` |
| Repository | `C:\Users\jiheo\tca_map` |
| Current branch | `codex/epoch5-official-prior-first` |
| Scientific HEAD | `cd3853285a5dfabdee1ab21524392acf9ad2bc64` |
| HEAD subject | `Record libero90 tasks81-83 no-headroom diagnostics` |
| Git status | `## codex/epoch5-official-prior-first...origin/codex/epoch5-official-prior-first`; untracked `rollouts/2026_07_17/`, `rollouts/2026_07_18/`; ignored `runs/xvla_prior/*task75*` artifacts present |
| `main` HEAD | `8dc4de2fdbf576ace8bdf3699d190b761553c1fa` |
| Active Windows research Python | none detected |
| Active WSL research Python | none detected; only Ubuntu service Python daemons were listed |
| Worker classification | `NO_ACTIVE_SCIENTIFIC_WORKER_AT_AUDIT_SNAPSHOT` |
| CUDA/GPU snapshot | RTX 5080, 16,303 MiB total, 4,300 MiB used, 24% utilization; no research Python compute process detected |
| RAM snapshot | Windows 23.16 GiB total / 7.43 GiB free |
| Disk snapshot | Windows C: 688.96 GiB used / 241.57 GiB free |
| Current epoch/cycle/stage | Epoch 5, cycle 0; pushed handoff `epoch_5_post_mpr_xvla_identity_grid_no_fresh_target`; local task75 second-prior pending |
| Current prior sequence | OpenVLA-OFT first, LightVLA second, X-VLA third |
| Current decision | pushed `POST_MPR_XVLA_IDENTITY_GRID_NO_FRESH_TARGET` / `LIBERO90_TASKS81_83_HEADROOM_NOT_VERIFIED_NO_OURS_TARGET`; local task75 `TASK75_TASK_LEVEL_EXPERT_HEADROOM_POSITIVE_SAME_RESET_UNAVAILABLE` but no second-prior result |
| Current next action | after user review only, complete the task75 second-prior screen or select a new residual source; no Ours design, optimizer, checkpoint, or closed-loop Ours rollout |
| Current reports | `reports/epoch5_prior_reproduction_result.md`, `reports/epoch5_prior_reproduction_result.json`, `reports/autonomous_compact_handoff.md` |
| Current X-VLA result | LIBERO-90 tasks70..89 identity `20260724`: `runs/xvla_prior/failure_scan_libero90_identity20260724_tasks70_89_post_secondprior_20260718T001938KST/scan_summary.json`, SHA-256 `86c4f3f4d35adb1581620c07d4287c4b52767baf3c4a404d86a4dfe7d4ec0708` |
| Current Base/headroom result | tasks81/83 Base `runs/xvla_prior/diagnostic_smolvla_base_libero90_tasks81_83_id20260724_officialenv_20260718T003018KST/result.json`, SHA-256 `b23ea884a89e3952e2166c29c924bd3eee0ae520ca8218b00ff051edef708d00`; headroom not verified for both |
| Local task75 result | X-VLA scan `runs/xvla_prior/failure_scan_libero90_identity20260725_tasks70_89_post_noheadroom_20260718T003659KST/scan_summary.json`, SHA-256 `74a54dad9417edf50a8a8a506ca27fbe462b85dca9a175e410092a630c544d50`; task75 result SHA-256 `8270a32c8eb4829db4cb75191f7a55fbf68e2b4db04c68c4424f0c55f56a9bb2` |
| Local task75 Base/headroom | Base result SHA-256 `18d6925c257f5ae231d25e39c539e564c0b1b43c9538fc2ec1c4e994d974b0e1`; headroom result SHA-256 `768d82bbdc89c3a7bc1a3d11103076a5ecb392f45c878dc90a550d2b313aade0` |
| State-file caveat | `reports/autonomous_until_paper_state.json` and `reports/autonomous_ral_campaign_state.json` still contain stale `current_commit` values from `b0ecb6e`; live HEAD and current artifacts are authoritative |

Current pushed diagnostic: LIBERO-90 identity `20260724` is covered across tasks0..89; tasks81 and 83 are clean X-VLA/Base shared residuals but expert headroom was not verified, so they do not authorize Ours design. Current ignored/local diagnostic: LIBERO-90 identity `20260725` tasks70..89 completed 20/20 with 19 successes and only task75 failed; matched SmolVLA Base task75 also failed, and nearest-demo expert replay gives task-level headroom with same-reset unavailable (`demo_18`, L2 `0.308533477`, exact replay success true, default reset false). The task75 second-prior screen did not run before this audit pause, so task75 is unfinished local work, not an Ours candidate.

## 3. Major Infrastructure Milestones

| Milestone | Commit | Artifact | Evidence | Current validity | Paper-method contribution |
|---|---|---|---|---|---|
| Official SmolVLA/LeRobot loader | `83e88a7` | `reports/official_smolvla_lerobot_model_load_status.md` | official checkpoint loaded locally | valid | infrastructure only |
| Official LIBERO assets | `2a4cad2` | `reports/official_smolvla_libero_asset_verification.md` | dataset/checkpoint verification | valid | infrastructure only |
| Official 8D state / 7D action semantics | `2efdd9e` | `reports/official_smolvla_libero_action_schema.md` | action/state schema audit | valid | infrastructure only |
| WSL/Linux CUDA rollout | `54a80ff` | `tca_map/smolvla/official_wsl_libero_rollout.py` | official simulator path and closed-loop pilot | valid | infrastructure only |
| Exact-init replay stabilization | `2b80cfb` | `reports/exact_init_expert_replay_stabilization.md` | eligible expert successes 6/6, adapter 0/6 | valid | infrastructure only |
| Deterministic evaluation | `d12ef6d` | `reports/official_smolvla_eval_determinism_check.md` | fixed-seed exact max diff 0 | valid when seeded | infrastructure only |
| Persisted LoRA adapters | `15649d6` | `reports/official_smolvla_lora_checkpoint_manifest.json` | seed bundles/checksums persisted | valid but not paper method | infrastructure/control |
| Quantized OpenVLA-OFT INT4 | `5c2a364` | `reports/openvla_oft_quantized_hard_slice_result.md` | hard-slice prior 20/20 | valid diagnostic | prior diagnostic only |
| Cross-backbone exact-state evaluation | `5c2a364` | `reports/openvla_oft_quantized_hard_slice_result.md` | OpenVLA did not share the same SmolVLA hard failures | valid diagnostic | blocks overclaiming |
| Official-prior-first pivot | `d268a83` | `reports/epoch5_prior_ecosystem_selection.md` | selected executable priors before Ours | valid process correction | governance/process |
| OpenVLA residual manifest control | `ffb55f5`, `22469ce` | `reports/epoch5_prior_reproduction_plan.md`, `runs/openvla_oft_int4/*residual*` | OpenVLA 14/16 vs SmolVLA 7/16 | valid, later superseded as task8 target by X-VLA | prior diagnostic |
| R2R-OFT bounded training path | `0af0269`..`9db6abf` | `runs/openvla_oft_int4/epoch5_r2r_oft_training_spec_v1.json` | data audit, QLoRA smoke, 64-step arms, offline no-pass | valid negative development evidence | stopped Ours extension |
| LightVLA and ATCD diagnostics | `44bcdd6`, `9d1d785`, `ed6837f` | `runs/lightvla_prior/*` | LightVLA 6/8, CR 6/8, ATCD signal below threshold | valid diagnostic | not paper method |
| X-VLA official-prior runner | `e81387c`..`312baa2` | `scripts/epoch5_xvla_libero10_task8_eval.py`, `runs/xvla_prior/*` | load/smoke/task8/task1 matched diagnostics | valid diagnostic | prerequisite before Ours |
| Task1 headroom/data gates | `3e49cac`, `ae9505e` | `runs/xvla_prior/diagnostic_task1_expert_headroom_20260727_20260717T180914KST/result.json`, `runs/xvla_prior/diagnostic_task1_basket_data_audit_20260727_20260717T181823KST/result.json` | task-level expert replay succeeds; data audit finds 4,607 train and 1,079 validation one-target chunks | valid but same-reset headroom unavailable | prerequisite before Ours |
| BR-XVLA bounded training and residual screen | `04c5239`..`b90b26b` | `runs/xvla_prior/epoch5_br_xvla_training_spec_v1.json`, `runs/xvla_prior/epoch5_br_xvla_training/*/result.json`, `runs/xvla_prior/epoch5_br_xvla_closed_loop_residual_20260727/closed_loop_result.json` | gradient smoke, 64-step two-arm training, offline validation, and frozen residual screen completed; primary failed while uniform ablation succeeded | valid bounded no-pass | selected method closed |
| X-VLA prior failure-scan worker | `62713d5`..`cc03266` | `scripts/run_xvla_prior_failure_scan_identity_worker.sh`, `runs/xvla_prior/failure_scan_*` | durable foreground WSL worker, manifest/heartbeat/exit-code/status files, cross-suite task-count support | valid diagnostic infrastructure | prior mining only |
| MPR-XVLA bounded training/offline gate | `4cdb49f`..`f07d602` | `runs/xvla_prior/epoch5_mpr_xvla_training_spec_v1.json`, `runs/xvla_prior/epoch5_mpr_xvla_training/*/result.json`, `runs/xvla_prior/epoch5_mpr_xvla_offline_validation_step0064_repaired_20260717T2200KST.json` | preoptimizer smokes and two-arm 64-step training completed; primary phase-1 loss `0.8785358916` did not beat uniform `0.8785358369` | valid offline no-pass | selected method closed before rollout |
| Cross-suite X-VLA residual/headroom screens | `9117b00`..`cd38532` | `reports/epoch5_prior_reproduction_result.json`, `runs/xvla_prior/failure_scan_libero_*`, `runs/xvla_prior/diagnostic_*headroom*` | LIBERO goal/object/spatial/90 scans, spatial task5 OpenVLA second-prior solve, LIBERO-90 tasks81/83 no-headroom decision | valid diagnostic infrastructure | no Ours method evidence |
| Local task75 ignored artifacts | local uncommitted/ignored | `runs/xvla_prior/failure_scan_libero90_identity20260725_tasks70_89_post_noheadroom_20260718T003659KST`, `runs/xvla_prior/diagnostic_smolvla_base_libero90_task75_id20260725_officialenv_20260718T004412KST/result.json`, `runs/xvla_prior/diagnostic_libero90_task75_expert_headroom_20260725_20260718T004553KST/result.json` | X-VLA task75 and Base task75 both failed; task-level headroom positive; second-prior screen not run | unfinished local diagnostic | not an Ours candidate yet |
| Partial/resume/durable worker infrastructure | many | `reports/*partial*.json`, `runs/*heartbeat*`, `*.pid` | 52 tracked report partials, 69 heartbeat files, 93 PID-like files | valid operational infrastructure | infrastructure only |

Tracked inventory at audit refresh: 2,698 tracked files; 1,777 under `reports`, 284 under `scripts`, 257 under `tests`, 225 under `tca_map`, 103 under `runs`, and 21 under `rollouts`. Machine-readable inventory from the prior audit remains the latest normalized count for typed artifacts: 152 tracked `reports/*result.json`, 72 report manifests, 52 report partials, 196 `reports`/`runs` result JSONs, 57 exit-code files, 69 heartbeat files, and 93 PID/PID-like files.

## 4. Master Method Ledger

Legend: `Impl`, `Train`, `GPU`, `Sim`, `S0`, `SA`, `SB`, and `2BB` mean implementation, training/checkpoint, GPU use, official simulator rollout, Stage 0, Stage A, Stage B, and second-backbone Ours. `Closure` is permanent scientific closure, not an operational stop. Every row has at least one evidence path.

| # | Epoch/cycle | Method/route | Core idea | Class | Closest prior | Branch/commit | Impl | Train | GPU | Sim | S0 | SA | SB | 2BB | Principal metric/result | Final decision | Evidence class | Closure | Reopen | Evidence |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | historical | TCA-Map | target-conditioned action mapping | representation | TCA-style target conditioning | historical | Yes | Yes | Yes | No | NA | No | No | No | target-prior positive but 7D head lost to mean | archive | VALID_HISTORICAL | Yes | No | `reports/final_project_state.md` |
| 2 | historical | TCA-Select | candidate selection | verifier | target selection | historical | Yes | No | No | No | NA | No | No | No | no meaningful gain; collapse risk | archive | INCONCLUSIVE | No | No | `reports/killed_routes_summary.md` |
| 3 | historical | ActionMap mini-anchor | action heatmap approximation | representation | ActionMap | historical | Yes | No | No | No | NA | No | No | No | 0.52993 vs mean 0.46677, MLP 0.50193 | killed | VALID_HISTORICAL | Yes | No | `reports/actionmap_mini_anchor_state1_result.json` |
| 4 | historical | CSS-Shield | semantic safety shield | action filtering | safety shield | historical | Yes | No | No | No | NA | No | No | No | no wrong-target gain, intervention rate 1.0 | killed | VALID_HISTORICAL | Yes | No | `reports/css_shield_autopilot_state.json` |
| 5 | historical | ExecSpec-Repair | executable-spec action repair | action repair | spec repair | historical plus unmerged state1 branch | Yes | No | No | No | NA | No | No | No | full 17/19, diagonal affine 17/19 | killed | VALID_HISTORICAL | Yes | No | `reports/execspec_state3_5_baseline_dominance_audit.json` |
| 6 | historical | AMP-GD | gradient action planning | optimization | action planning | historical | Yes | No | No | Partial | NA | No | No | No | toy pass, LIBERO tiny not above safety/random | archive | INCONCLUSIVE | No | No | `reports/amp_gd_state2_report.json` |
| 7 | historical | ResetSpec-Retarget | reset/object retargeting | retargeting | object retargeting | historical | Yes | No | No | No | NA | No | No | No | global scale success 1, object-relative 0 | archive | INCONCLUSIVE | No | No | `reports/resetspec_state1_result.json` |
| 8 | historical | Phase/event retiming | retime action phases | temporal | phase retiming | historical | Yes | No | No | No | NA | No | No | No | 0/9 recovery | killed | VALID_HISTORICAL | Yes | No | `reports/phase_locked_retiming_state1_result.json` |
| 9 | historical | TL-ChunkRepair | chunk repair | temporal repair | localization repair | historical | Yes | No | No | No | NA | No | No | No | safe success 0/8; no-repair best 1/1 | killed | VALID_HISTORICAL | Yes | No | `reports/tl_chunkrepair_state1_result.json` |
| 10 | historical | ContactTube-Aug | contact tube augmentation | contact representation | contact-tube imitation | historical | Yes | No | No | No | NA | No | No | No | missing HDF5 pose; simple retarget better | measurement failure | INVALID_QUARANTINED | No | No | `reports/contacttube_aug_state1_result.json` |
| 11 | historical | PRISM-VLA | visual canonicalization | visual TTA | PRISM-like transforms | historical | Yes | No | No | No | NA | No | No | No | canonical 0.474066 > best PRISM 0.436356 | killed | VALID_HISTORICAL | Yes | No | `reports/all_killed_routes_summary.md` |
| 12 | historical | ContactSet-VLA | contact-set geometry | contact representation | contact-set policies | historical | Yes | No | No | No | NA | No | No | No | full 1.1050 worse than single/no-geom | archive | INCONCLUSIVE | No | No | `reports/all_killed_routes_summary.md` |
| 13 | historical | SafeTrace-VLA | safety preference traces | supervision | DPO/safety VLA | historical | Yes | No | No | No | NA | No | No | No | 800 pairs but only 10 nontrivial | data failure | INVALID_QUARANTINED | No | No | `reports/safetrace_vla_state1_result.json` |
| 14 | historical | SafeLoRA-VLA | safety LoRA | adapter | SafeLoRA | historical | No | No | No | No | NA | No | No | No | no local experiment | preimplementation | UNKNOWN | No | No | `reports/all_killed_routes_summary.md` |
| 15 | historical | PatchGuard-VLA | guarded patch adapter | adapter | patch/adversarial LoRA | historical | Yes | Yes | Yes | No | NA | No | No | No | 0.13356 vs adv LoRA 0.142803, cutout 0.02973 | killed | VALID_HISTORICAL | Yes | No | `reports/patchguard_vla_state1_result.json` |
| 16 | historical | SmolVLA LoRA baseline | official PEFT baseline | infrastructure | LoRA/PEFT | `0a15424`, `54a80ff` | Yes | Yes | Yes | Yes | NA | Partial | No | No | Base 75%, LoRA seeds 83.3/66.7/75 over 12 each | baseline only | SUPERSEDED | No | No | `reports/official_smolvla_libero_baseline_scaleup_result.json` |
| 17 | historical | Custom SmolVLA 7D adapter | fixed 7D adapter | adapter | PEFT adapter | historical | Yes | Yes | Yes | Yes | NA | Partial | No | No | offline rank8 0.494959, exact replay 0/6 | implementation/control gap | INVALID_QUARANTINED | No | No | `reports/tg7d_adapter_state_gate.json` |
| 18 | historical | TG-VLA | trajectory-generation VLA | representation | trajectory VLA | historical | No | No | No | No | NA | No | No | No | no experiment | preimplementation | UNKNOWN | No | No | `reports/killed_routes_summary.md` |
| 19 | historical | TG-7D | trajectory 7D adapter | representation | TG adapter | historical | Yes | Yes | Yes | No | NA | No | No | No | L2 0.740922 vs canonical 0.587661 | killed | VALID_HISTORICAL | Yes | No | `reports/tg7d_adapter_state_gate.json` |
| 20 | historical | Post-canonical residual mining | residual after canonicalization | residual | canonical residual | historical | Yes | No | No | No | NA | No | No | No | oracle headroom -0.137013 | no headroom | VALID_HISTORICAL | No | No | `reports/final_autonomous_method_decision.md` |
| 21 | historical | FCAR | feature-conditioned residual | residual | residual adapters | historical | Yes | Yes | Yes | No | NA | No | No | No | full 0.100145 vs static 0.09118, LoRA 0.07619 | killed | VALID_HISTORICAL | Yes | No | `reports/final_autonomous_method_decision.md` |
| 22 | historical | ECHO | effect-conditioned headroom | verifier | effect repair | `8dc4de2` | Yes | No | No | No | NA | No | No | No | Base/oracle/random all 0.8333; recoverable 0 | no headroom | VALID_HISTORICAL | No | No | `reports/implementation_v2_final_decision.md` |
| 23 | historical+epoch5 | OpenVLA-OFT INT4 diagnostic | second-backbone prior diagnostic | prior diagnostic | OpenVLA-OFT | `5c2a364`, `22469ce` | Yes | No | Yes | Yes | NA | No | No | Diagnostic | hard slice 20/20; residual 14/16 vs Base 7/16 | diagnostic only | VALID_CANONICAL_PARTIAL | No | No | `reports/openvla_oft_quantized_hard_slice_result.md`, `reports/epoch5_prior_reproduction_result.md` |
| 24 | historical | PhaseBarrier-VLA | phase barrier intervention | action filter | phase barrier | historical | Yes | Yes | Yes | Yes | NA | Yes | Yes | No | valid repair full 0/20 vs Base 8/20, ablation 9/20 | killed | VALID_CANONICAL | Yes | No | `reports/phase_barrier_bounded_repair_result.json` |
| 25 | historical | CensorCredit | censored correction | supervision | credit/censoring | `06cf915` | Yes | Yes | Yes | Partial | NA | Partial | No | No | apparent positive invalidated by identical labels/heads | data failure | INVALID_QUARANTINED | No | No | `reports/censor_credit_empirical_postmortem.md` |
| 26 | historical | ISAC-VLA | implicit safety/action correction | action filter | safety/action correction | historical | No | No | No | No | NA | No | No | No | not implemented locally | preimplementation | UNKNOWN | No | No | `reports/all_killed_routes_summary.md` |
| 27 | epoch/auto | DICD | distributional intervention | residual | NOT_RECORDED | `a2154c2` | Yes | Yes | Yes | Yes | Yes | Yes | No | No | 1/10 vs 2/10 | underpowered archive | INCONCLUSIVE | No | No | `reports/autonomous_cycle_01_action_conditioning_kill.md` |
| 28 | epoch/auto | FEDO | fault/execution domain optimization | action filter | APEX proxy | `b2f7b50` | Yes | Yes | Yes | Yes | Yes | Yes | No | No | clean full 0 vs frozen 4 | killed | VALID_CANONICAL | Yes | No | `reports/autonomous_cycle_02_censored_correction_kill.md` |
| 29 | epoch/auto | GCAP | goal-conditioned action prior | goal conditioning | GCAP-like | `e24a6a1` | Yes | Yes | Yes | Yes | Yes | Yes | No | No | target full 3 vs Sobel 5; clean positive | underpowered archive | INCONCLUSIVE | No | No | `reports/autonomous_cycle_03_contact_barrier_kill.md` |
| 30 | epoch4/c1 | PTC-VLA | policy trajectory correction | action repair | NOT_RECORDED | `ce7d455` | Yes | Yes | Yes | Yes | Yes | Yes | No | No | full 0/10 vs Base 3/10 | killed | VALID_CANONICAL | Yes | No | `reports/ptc_vla/stage_a_result.json` |
| 31 | epoch4/c2 | SACF-VLA | safety/action correction field | action filter | CAG/null proxy | `fc0fb1e` | Yes | Yes | Yes | Yes | Yes | Yes | No | No | full 0/10 vs Base 7/10 | killed | VALID_CANONICAL | Yes | No | `reports/sacf_vla/stage_a_result.json` |
| 32 | epoch4/c3 | OCFN-VLA | observation corruption filter | visual TTA | visual robustness | `c183d15` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | full 26/80 vs zero-noise 27/80 | killed | VALID_CANONICAL | Yes | No | `reports/ocfn_vla/stage_b_result.json` |
| 33 | epoch4/c4 | CBFD-VLA | contrastive behavior distillation | supervision | distillation | `c0fbca2` | Yes | Yes | Yes | Yes | Yes | Yes | No | No | full 0/10 vs Base 7/10 | killed | VALID_CANONICAL | Yes | No | `reports/cbfd_vla/stage_a_result.json` |
| 34 | epoch4/c5 | SCVC-VLA | semantic canonical view correction | visual TTA | visual canonicalization | `2c733a3` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | full 11/40 vs shifted Base 20/40 | killed | VALID_CANONICAL | Yes | No | `reports/scvc_vla/stage_b_result.json` |
| 35 | epoch4/c6 | PSE-VLA | predictive state estimator | temporal | state estimator | `c4607b8` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | full 50/80 vs bright-single 51/80 | killed | VALID_CANONICAL | Yes | No | `reports/pse_vla/stage_b_result.json` |
| 36 | epoch4/c7 | RCV-VLA | retrieval correction | memory | retrieval control | `3a8a815` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | full 20/40, no-context/stateless 24/40 | killed | VALID_CANONICAL | Yes | No | `reports/rcv_vla/stage_2b_result.json` |
| 37 | epoch4/c8 | CAVM-VLA | contrastive action-value memory | memory | success memory | `e69f64f` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | full 24/58 vs nearest 23/58, Base 22/58 | unresolved near-miss | INCONCLUSIVE | No | No | `reports/cavm_vla/stage_2b_expansion_result.json` |
| 38 | epoch4/c9 | FANG-VLA | failure-aware negative guidance | residual | failure/action field | `de08f34` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | full 11/40 vs Base 16/40, ablation 11/40 | killed | VALID_CANONICAL | Yes | No | `reports/fang_vla/stage_b_result.json` |
| 39 | epoch4/c10 | EvoState-VLA | action-evolved state | temporal | dynamics/state prior | `a2e94c1` | Yes | No | No | No | Yes | No | No | No | 4,221 pairs, improvement 0.024689 < 0.05 | design stop | VALID_CANONICAL | Yes | No | `reports/evostate_vla/development_audit.json` |
| 40 | epoch4/c11 | RAC-VLA | reflective action correction | action filter | Reflective proxy | `adf3d07` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | full/Base/proxy 1/40; ablation/inverse 2/40 | killed | VALID_CANONICAL | Yes | No | `reports/rac_vla/stage_b_result.json` |
| 41 | epoch4/c12 | MTF-VLA | memory/temporal filter | memory | FrameSkip proxy | `0824ba8` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | full 26/40 vs no-retention 32/40 | killed | VALID_CANONICAL | Yes | No | `reports/mtf_vla/stage_b_result.json` |
| 42 | epoch4/c13 | DAGR-VLA | demonstration/action-guided retrieval | memory | DAM proxy | `1853080` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | full 18/40 vs Base 28/40 | killed | VALID_CANONICAL | Yes | No | `reports/dagr_vla/stage_b_result.json` |
| 43 | epoch4/c14 | MARC-VLA | model action residual correction | residual | OpenVLA L1 proxy | `0d5648d` | Yes | Yes | Yes | Yes | Yes | Yes | No | No | full 0/10 vs Base 8/10 | killed | VALID_CANONICAL | Yes | No | `reports/marc_vla/stage_a_result.json` |
| 44 | epoch4/c15 | PESA-VLA | prior entropy scheduler | verifier | uncertainty/AAC adjacent | `f6b65e6` | Yes | No | No | No | Yes | No | No | No | query probe below majority | design/data failure | INVALID_QUARANTINED | No | No | `reports/pesa_vla/development_audit.json` |
| 45 | epoch4/c16 | EAC-VLA | adaptive action chunking | action filter | AAC proxy | `5e7bbdd` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | full 29/40 vs Base/AAC/ablation 30/40 | killed | VALID_CANONICAL | Yes | No | `reports/eac_vla/stage_b_result.json` |
| 46 | epoch4/c17 | G3P-VLA | goal-gradient/proxy prior | residual | NOT_RECORDED | `9079a25` | Yes | No | No | No | Yes | No | No | No | Stage 0 weak/failed | data failure | INVALID_QUARANTINED | No | No | `reports/g3p_vla/development_audit.json` |
| 47 | epoch4/c18 | CALA-VLA | causal action latent alignment | representation | NOT_RECORDED | `65fd947` | Yes | Yes | Yes | No | Yes | No | No | No | full RMSE 3.1988 vs history-only 3.14397 | possible false negative | INCONCLUSIVE | No | Maybe | `reports/cala_vla/development_audit.json` |
| 48 | epoch4/c19 | RAR-VLA | residual action refinement | residual | NOT_RECORDED | `fbffc0a` | Yes | Yes | Yes | No | Yes | No | No | No | full RMSE 0.171954 vs zero-residual 0.165597 | possible false negative | INCONCLUSIVE | No | Maybe | `reports/rar_vla/development_audit.json` |
| 49 | epoch4/c20 | COVI-VLA | contrastive visual intervention | visual TTA | visual intervention | `6027d10` | Yes | Yes | Yes | No | Yes | No | No | No | repaired valid 0.2, only two nonzero objectives | implementation failure | INVALID_QUARANTINED | No | No | `reports/covi_vla/stage_0_result.json` |
| 50 | epoch4/c21 | LIFT-VLA | latent intervention finetuning | adaptation | LIFT | `b1adc67` | Yes | No | No | No | Yes | No | No | No | compute infeasible | resource failure | INVALID_QUARANTINED | No | No | `reports/lift_vla/stage_0_result.json` |
| 51 | epoch4/c22 | IARC-VLA | inverse action-region correction | residual | inverse correction | `aee7867` | Yes | Yes | Yes | No | Yes | No | No | No | action validity/data range invalid 0.3 | implementation/data failure | INVALID_QUARANTINED | No | No | `reports/iarc_vla/stage_0a_result.json` |
| 52 | epoch4/c23 | FAMR-VLA | feature-aligned model residual | residual | model residual | `5725a68` | Yes | Yes | Yes | No | Yes | No | No | No | endpoint training; no headroom/data failure | implementation/data failure | INVALID_QUARANTINED | No | No | `reports/famr_vla/stage_0a_result.json` |
| 53 | epoch4/c24 | PCAV-VLA | policy-conditioned verifier | verifier | action verifier | `787fc7c` | Yes | No | No | No | Yes | No | No | No | no usable headroom | no headroom | INVALID_QUARANTINED | No | No | `reports/pcav_vla/stage_0a_result.json` |
| 54 | epoch4/c25 | SPARC-VLA | sparse action correction | residual | sparse correction | `660db5a` | Yes | No | No | No | Yes | No | No | No | action-validity failure | implementation failure | INVALID_QUARANTINED | No | No | `reports/sparc_vla/stage_0a_result.json` |
| 55 | epoch4/c26 | NICE-VLA | invariant context encoder | temporal | invariant context | `a814a7a` | Yes | Yes | Yes | No | Yes | No | No | No | Stage0a pass then Stage0b1 collapsed contrast | data failure | INVALID_QUARANTINED | No | No | `reports/nice_vla/stage_0b1_result.json` |
| 56 | epoch4/c27 | HEST-VLA | history state transformer | temporal | history transformer | `5124db4` | Yes | No | No | No | Yes | No | No | No | support/optimization failure | implementation failure | INVALID_QUARANTINED | No | No | `reports/hest_vla/stage_0a_result.json` |
| 57 | epoch4/c28 | HASTE-VLA | history action-state estimator | temporal | history estimator | `4c0a015` | Yes | No | No | No | Yes | No | No | No | support/optimization failure | implementation failure | INVALID_QUARANTINED | No | No | `reports/haste_vla/stage_0a_result.json` |
| 58 | epoch4/c29 | KITE-VLA | keyframe intervention transform | temporal repair | keyframe intervention | `500dfb1` | Yes | No | No | No | Yes | No | No | No | support failure | implementation failure | INVALID_QUARANTINED | No | No | `reports/kite_vla/stage_0a_result.json` |
| 59 | epoch4/c30 | VDR-VLA | visual discrepancy repair | visual TTA | visual repair | `52dc9ae` | Yes | No | No | No | Yes | No | No | No | action-validity/pre-manifest failure | implementation failure | INVALID_QUARANTINED | No | No | `reports/vdr_vla/stage_0a_result.json` |
| 60 | epoch4/c31 | RAP-VLA | retrieval-augmented policy | memory | retrieval policy | `cc20e20` | Yes | No | No | No | Yes | No | No | No | launcher/preflight failure | implementation failure | INVALID_QUARANTINED | No | No | `reports/rap_vla/stage_0_result.json` |
| 61 | epoch4/c32 | AMP-VLA | action-manifold projection | action filter | manifold projection | `ef1f20a` | Yes | No | No | No | Yes | No | No | No | optimization failure | implementation failure | INVALID_QUARANTINED | No | No | `reports/amp_vla/stage_0_result.json` |
| 62 | epoch4/c33 | CFR-VLA | counterfactual failure repair | residual | counterfactual repair | `bdec3dc` | Yes | No | No | No | Yes | No | No | No | no usable headroom | no headroom | INVALID_QUARANTINED | No | No | `reports/cfr_vla/stage_0_result.json` |
| 63 | epoch4/c34 | TSC-VLA | temporal state correction | temporal | temporal correction | `e4a5abc` | Yes | No | No | No | Yes | No | No | No | no usable headroom | no headroom | INVALID_QUARANTINED | No | No | `reports/tsc_vla/stage_0_result.json` |
| 64 | epoch4/c35 | CCIF-VLA | cross-context intervention filtering | action filter | intervention filtering | `77a470e` | Yes | No | No | No | Yes | No | No | No | design failure | design failure | INVALID_QUARANTINED | No | No | `reports/ccif_vla/stage_0_result.json` |
| 65 | epoch4/c36 | URF-VLA | uncertainty residual filtering | residual | uncertainty residual | `466511f` | Yes | No | No | No | Yes | No | No | No | no usable headroom | no headroom | INVALID_QUARANTINED | No | No | `reports/urf_vla/stage_0_result.json` |
| 66 | epoch4/c37 | S2C-VLA | state-to-contact conversion | contact | contact conversion | `73dc3c0` | Yes | No | No | No | Yes | No | No | No | cache/data failure | data failure | INVALID_QUARANTINED | No | No | `reports/s2c_vla/stage_0_result.json` |
| 67 | epoch4/c38 | LCG-VLA | latent contact guidance | contact | latent contact | `9d48f36` | Yes | No | No | No | Yes | No | No | No | design/no-headroom failure | design failure | INVALID_QUARANTINED | No | No | `reports/lcg_vla/stage_0_result.json` |
| 68 | epoch4/c33 | AFID-VLA | action-frequency invariant descriptor | representation | invariant descriptor | `f392a33` | Yes | No | No | No | Yes | No | No | No | objective-scale failure | implementation failure | INVALID_QUARANTINED | No | No | `reports/afid_vla/stage_0_result.md` |
| 69 | epoch4/c34 | BRID-VLA | behavior residual identity decomposition | residual | residual decomposition | `7348dd9` | Yes | No | No | No | Yes | No | No | No | design/no-headroom failure | design failure | INVALID_QUARANTINED | No | No | `reports/brid_vla/stage_0_result.json` |
| 70 | epoch4/c35 | MHS-VLA | multi-horizon state supervision | temporal | multi-horizon prior | `48c8239` | Yes | No | No | No | Yes | No | No | No | cache/data failure | data failure | INVALID_QUARANTINED | No | No | `reports/mhs_vla/stage_0_result.json` |
| 71 | epoch4/c36 | DCCG-VLA | dense cross-context guidance | visual/action | dense context guidance | `bdeed30` | Yes | No | No | No | Yes | No | No | No | cache/data failure | data failure | INVALID_QUARANTINED | No | No | `reports/dccg_vla/stage_0_result.json` |
| 72 | epoch4/c37 | CSPR-VLA | consistency-scaled policy residual | residual | consistency residual | `dcfcfa6` | Yes | No | No | No | Yes | No | No | No | gradient-scale implementation failure | implementation failure | INVALID_QUARANTINED | No | No | `reports/cspr_vla/stage_0_result.json` |
| 73 | epoch4/c38 | MCI-VLA | multi-consistency invariance | representation | ROVLA multi-consistency proxy | `bc15132` | Yes | No | No | No | Yes | No | No | No | weighted gradient norm ratio 324.58 > 100 | implementation failure | INVALID_QUARANTINED | No | No | `reports/mci_vla/stage_0_result.json` |
| 74 | epoch5 | R2R-OFT | remaining-object weighted OFT | prior extension | OpenVLA-OFT | `0af0269`..`9db6abf` | Yes | Yes | Yes | No | Yes | No | No | No | 64-step arms; fixed offline validation failed action-delta gate | offline no-pass | INVALID_QUARANTINED | No | No | `reports/epoch5_prior_reproduction_result.json` |
| 75 | epoch5 control | 4-step OpenVLA requery | shorter action chunk control | simple control | OpenVLA-OFT | `cdb7404` | Yes | No | Yes | Yes | NA | No | No | Diagnostic | 5/8 vs original 6/8 on task8 | not selected | VALID_CANONICAL | No | No | `runs/openvla_oft_int4/epoch5_task8_short_requery4_openvla_int4.json` |
| 76 | epoch5 prior | OpenPI pi0.5 | official prior fallback | resource check | pi0.5/OpenPI | `b5a295d` | No | No | No | No | NA | No | No | Diagnostic | resource-blocked | resource blocker | INVALID_QUARANTINED | No | No | `reports/epoch5_prior_reproduction_result.json` |
| 77 | epoch5 prior | PCD/PCD-LeRobot | official prior fallback | resource check | PCD | `8e55279` | No | No | No | No | NA | No | No | Diagnostic | dependency/checkpoint/multi-GPU blocked | resource blocker | INVALID_QUARANTINED | No | No | `reports/epoch5_prior_reproduction_result.json` |
| 78 | epoch5 prior | LightVLA | lightweight OpenVLA prior | prior diagnostic | LightVLA | `44bcdd6` | Yes | No | Yes | Yes | NA | No | No | Diagnostic | 6/8, complementary failures vs OpenVLA 6/8 | prior diagnostic | VALID_CANONICAL | No | No | `runs/lightvla_prior/diagnostic_lightvla_libero10_task8_all_20260717T1535KST/result.json` |
| 79 | epoch5 | CR-LightVLA | collision-rescue token pruning | prior extension | LightVLA | `9d1d785` | Yes | No | Yes | Yes | Yes | No | No | No | 6/8; regression `20260718`; no prototype GO | no-pass | INVALID_QUARANTINED | No | No | `runs/lightvla_prior/cr_lightvla_task8_all_20260717T1600KST/result.json` |
| 80 | epoch5 | ATCD | action-teacher complementarity distillation audit | prior extension | LightVLA + OpenVLA | `ed6837f` | Yes | No | Yes | No | Yes | No | No | No | oracle relative gain 0.025896 < 0.03 | teacher signal not enough | INVALID_QUARANTINED | No | No | `runs/lightvla_prior/atcd_teacher_signal_20260717T1620KST/atcd_teacher_signal_result_v2.json` |
| 81 | epoch5 prior | RIPT-VLA | RL fine-tuning prior fallback | resource check | RIPT-VLA | `d762c78` | No | No | No | No | NA | No | No | Diagnostic | source import ok; assets/task/resources not comparable | resource blocker | INVALID_QUARANTINED | No | No | `reports/epoch5_prior_reproduction_result.json` |
| 82 | epoch5 prior | VLA-GSE | grounded search/exploration prior fallback | resource check | VLA-GSE | `d762c78` | No | No | No | No | NA | No | No | Diagnostic | no trained checkpoint; 8-GPU/80k-step reference | resource blocker | INVALID_QUARANTINED | No | No | `reports/epoch5_prior_reproduction_result.json` |
| 83 | epoch5 prior | X-VLA task8 | official X-VLA prior | prior diagnostic | X-VLA | `e81387c` | Yes | No | Yes | Yes | NA | No | No | Diagnostic | task8 8/8, solves prior residual | target removed | VALID_CANONICAL | No | No | `runs/xvla_prior/diagnostic_xvla_task8_all_20260717T1705KST/result.json` |
| 84 | epoch5 prior | X-VLA identity scan | one-identity LIBERO-10 scan | residual mining | X-VLA | `f74734d` | Yes | No | Yes | Yes | NA | No | No | Diagnostic | 10/10 successes at identity `20260724` | no candidate | VALID_CANONICAL | No | No | `runs/xvla_prior/failure_scan_libero10_identity20260724_20260717T1716KST` |
| 85 | epoch5 prior | X-VLA task1 residual | matched Base/Prior residual | prior diagnostic | X-VLA | `af25589`, `312baa2`, `3e49cac` | Yes | No | Yes | Yes | NA | No | No | Diagnostic | X-VLA 6/8 vs Base 3/8; shared failure `20260727`; task-level headroom positive, same-reset unavailable | continued into BR-XVLA | INCONCLUSIVE | No | Yes | `runs/xvla_prior/diagnostic_xvla_libero10_task1_id20260724_20260731_20260717T1729KST/result.json`, `runs/xvla_prior/diagnostic_smolvla_base_libero10_task1_id20260724_20260731_officialenv_20260717T1739KST/result.json`, `runs/xvla_prior/diagnostic_task1_expert_headroom_20260727_20260717T180914KST/result.json` |
| 86 | epoch5 prior | VLA-0 | third-pass unselected prior | resource check | VLA-0 | `312baa2` report context | No | No | No | No | NA | No | No | Diagnostic | HF asset 21.459 GiB; not selected over lighter X-VLA | unselected | INVALID_QUARANTINED | No | No | `reports/epoch5_prior_reproduction_result.json` |
| 87 | epoch5 prior | VLA-JEPA | third-pass unselected prior | resource check | VLA-JEPA | `312baa2` report context | No | No | No | No | NA | No | No | Diagnostic | HF asset 22.961 GiB; not selected over lighter X-VLA | unselected | INVALID_QUARANTINED | No | No | `reports/epoch5_prior_reproduction_result.json` |
| 88 | epoch5 | BR-XVLA | basket-remaining reweighted X-VLA | prior extension | X-VLA-Libero | `ae9505e`..`b90b26b` | Yes | Yes | Yes | Yes | Yes | No | No | No | closed-loop residual: prior failed, BR primary failed, uniform ablation succeeded | validation no-pass | VALID_CANONICAL | Yes | No | `reports/epoch5_task1_ours_candidate_design.md`, `runs/xvla_prior/epoch5_br_xvla_training/*/result.json`, `runs/xvla_prior/epoch5_br_xvla_closed_loop_residual_20260727/closed_loop_result.json` |
| 89 | epoch5 candidate | OCB-XVLA | object-contrast basket X-VLA | prior extension candidate | X-VLA-Libero | `ae9505e` | No | No | No | No | No | No | No | No | candidate score 73/100, not selected | not selected | INCONCLUSIVE | No | No | `reports/epoch5_task1_ours_candidate_design.md` |
| 90 | epoch5 prior | X-VLA task6 residual | matched Base/Prior residual | prior diagnostic | X-VLA + OpenVLA-OFT | `19177fe`..`d387d8a` | Yes | No | Yes | Yes | NA | No | No | Diagnostic | X-VLA 6/8 vs Base 3/8; task-level headroom positive; OpenVLA INT4 0/2 on residuals | continued into MPR-XVLA | INCONCLUSIVE | No | No | `runs/xvla_prior/diagnostic_xvla_libero10_task6_id20260724_20260731_20260717T2043KST/result.json`, `runs/xvla_prior/diagnostic_smolvla_base_libero10_task6_id20260724_20260731_officialenv_20260717T2047KST/result.json`, `runs/openvla_oft_int4/diagnostic_task6_residual_openvla_int4_20260725_20260731_openvlaenv_20260717T2114KST/result.json` |
| 91 | epoch5 | MPR-XVLA | mug-placed/pudding-right reweighted X-VLA | prior extension | X-VLA-Libero | `4cdb49f`..`f07d602` | Yes | Yes | Yes | No | Yes | No | No | No | primary phase-1 loss 0.8785358916 did not beat uniform 0.8785358369 | offline no-pass | VALID_CANONICAL | Yes | No | `reports/epoch5_task6_ours_candidate_design.md`, `runs/xvla_prior/epoch5_mpr_xvla_training/*/result.json`, `runs/xvla_prior/epoch5_mpr_xvla_offline_validation_step0064_repaired_20260717T2200KST.json` |
| 92 | epoch5 candidate | PRC-XVLA | pudding-right contrast X-VLA | prior extension candidate | X-VLA-Libero | `d387d8a`, `cb7f379` | No | No | No | No | No | No | No | No | candidate score 74/100; not elevated after MPR no-pass due no independent red-mug/distractor evidence | not selected / not elevated | INCONCLUSIVE | No | No | `reports/epoch5_task6_ours_candidate_design.md`, `reports/epoch5_prior_reproduction_result.json` |
| 93 | epoch5 prior | X-VLA spatial task5 | cross-suite residual and second-prior screen | prior diagnostic | X-VLA + OpenVLA-OFT | `93feebc`..`ca5f8cf` | Yes | No | Yes | Yes | NA | No | No | Diagnostic | X-VLA shared residual task5; Base 0/1; task-level headroom positive; OpenVLA INT4 solved 1/1 | target removed | VALID_CANONICAL | Yes | No | `runs/xvla_prior/failure_scan_libero_spatial_identity20260724_post_mpr_20260717T230355KST/scan_summary.json`, `runs/xvla_prior/diagnostic_smolvla_base_libero_spatial_task5_id20260724_officialenv_20260717T232734KST/result.json`, `runs/openvla_oft_int4/diagnostic_spatial_task5_openvla_int4_20260724_openvlaenv_20260717T234111KST/result.json` |
| 94 | epoch5 prior | X-VLA LIBERO-90 tasks81/83 | LIBERO-90 residual/headroom screen | prior diagnostic | X-VLA-Libero | `8da1108`..`cd38532` | Yes | No | Yes | Yes | NA | No | No | Diagnostic | X-VLA 18/20 with clean failures 81/83; Base 0/2; expert headroom not verified | no Ours target | VALID_CANONICAL | Yes | No | `runs/xvla_prior/failure_scan_libero90_identity20260724_tasks70_89_post_secondprior_20260718T001938KST/scan_summary.json`, `runs/xvla_prior/diagnostic_smolvla_base_libero90_tasks81_83_id20260724_officialenv_20260718T003018KST/result.json`, `runs/xvla_prior/diagnostic_libero90_task81_expert_headroom_20260724_20260718T003247KST/result.json`, `runs/xvla_prior/diagnostic_libero90_task83_expert_headroom_20260724_20260718T003343KST/result.json` |
| 95 | epoch5 local | X-VLA LIBERO-90 task75 | ignored/local residual/headroom thread | prior diagnostic | X-VLA-Libero | local ignored artifacts after `cd38532` | Yes | No | Yes | Yes | NA | No | No | Diagnostic | X-VLA 19/20 with task75 failure; Base 0/1; task-level headroom positive; second-prior not run | unfinished local diagnostic | INCONCLUSIVE | No | Yes | `runs/xvla_prior/failure_scan_libero90_identity20260725_tasks70_89_post_noheadroom_20260718T003659KST/scan_summary.json`, `runs/xvla_prior/diagnostic_smolvla_base_libero90_task75_id20260725_officialenv_20260718T004412KST/result.json`, `runs/xvla_prior/diagnostic_libero90_task75_expert_headroom_20260725_20260718T004553KST/result.json` |

## 5. Detailed Chronological Timeline

| Date KST | Commit | Action | Method | Result | Scientific meaning |
|---|---|---|---|---|---|
| 2026-06-27 | `07c823d` | repository initialized | workspace | agent instructions and scaffold | governance/evidence discipline starts |
| 2026-07-04..08 | many | early offline/local routes | TCA, CSS, ExecSpec, AMP-GD, retiming, ContactSet, ActionMap | mostly negative/invalid | exposed interface/data/action-head gaps before official simulator stabilization |
| 2026-07-09 | `83e88a7`, `2a4cad2`, `2efdd9e` | official SmolVLA/LIBERO loading and action semantics | infrastructure | official model/data/action schema usable | corrected early interface uncertainty |
| 2026-07-10..11 | `15649d6`, `54a80ff`, `5c2a364` | LoRA persistence, official closed-loop, OpenVLA INT4 | infrastructure/prior | Base/LoRA pilot, OpenVLA hard-slice 20/20 | simulator and second-backbone diagnostics become real |
| 2026-07-12 | `a2154c2`, `b2f7b50`, `e24a6a1` | first autonomous methods | DICD/FEDO/GCAP | DICD/GCAP underpowered, FEDO killed | early closed-loop evidence negative or inconclusive |
| 2026-07-12..15 | `ce7d455`..`5e7bbdd` | main Stage A/B campaign | PTC through EAC | 10 formal Stage B, no GO | valid kills and near-misses accumulated |
| 2026-07-14 | `e69f64f` | expansion result | CAVM | 24/58 vs nearest 23/58 | strongest near-miss, governance-closed |
| 2026-07-15..17 | many | Stage0-heavy late epoch | G3P through MCI | data/headroom/implementation failures | no paper method; many non-scientific stops |
| 2026-07-17 | `bc15132` | MCI adjudicated | MCI | gradient norm ratio 324.58 > 100 | previous Ours method stops as implementation failure |
| 2026-07-17 | `b0ecb6e` | first full-history audit | audit | 73 routes, no GO | recommended strategy reset |
| 2026-07-17 | `d268a83`, `ffb55f5`, `22469ce` | Epoch 5 official-prior-first | OpenVLA-OFT | hard-slice 20/20; residual 14/16 vs Base 7/16; task-level headroom positive | prior-positive residual discovered on task8 |
| 2026-07-17 | `0af0269`..`9db6abf` | Ours extension attempted after initial prior gate | R2R-OFT | bounded training completed; offline validation/action-delta no-pass | no closed-loop Ours rollout allowed |
| 2026-07-17 | `cdb7404` | simple control | OpenVLA 4-step requery | 5/8 vs original 6/8 | no-training requery did not explain/fix residual |
| 2026-07-17 | `fd68eaf`, `b5a295d`, `8e55279` | fallback prior preflight | OpenPI/pi0.5, PCD | blocked | resource/checkpoint limits recorded |
| 2026-07-17 | `44bcdd6`, `9d1d785`, `ed6837f` | second-pass prior and methods | LightVLA, CR-LightVLA, ATCD | LightVLA 6/8 complementary; CR 6/8; ATCD below signal threshold | no prototype GO; fallback required |
| 2026-07-17 | `d762c78` | second-pass fallback preflight | RIPT-VLA, VLA-GSE | source/HF checked, resource/comparability blocked | no local executable prior selected |
| 2026-07-17 | `e81387c` | third-pass prior diagnostic | X-VLA task8 | 8/8 | task8 residual solved by official prior; no Ours target remains there |
| 2026-07-17 | `f74734d`, `af25589`, `312baa2` | X-VLA residual search and matched Base/Prior | LIBERO-10 task1 | X-VLA 6/8 vs Base 3/8; shared failure `20260727` | residual candidate confirmed |
| 2026-07-17 | `3e49cac`, `ae9505e` | task1 headroom/data audit and candidate design | BR-XVLA/OCB-XVLA | task-level headroom positive; data audit passed; BR-XVLA selected | Ours design allowed only after prior/headroom gates |
| 2026-07-17 | `ed9becd`..`b90b26b` | BR-XVLA gradient/training/closed-loop residual screen | BR-XVLA | primary failed while uniform ablation succeeded on identity `20260727` | selected configuration archived; no retune |
| 2026-07-17 | `19177fe`..`e1d67fb` | post-BR X-VLA task6 residual and headroom | X-VLA task6 | X-VLA 6/8 vs Base 3/8; task-level headroom positive; OpenVLA INT4 0/2 | task6 became a prior-grounded candidate |
| 2026-07-17 | `d387d8a`..`f07d602` | candidate design and bounded training/offline gate | MPR-XVLA | primary did not beat uniform ablation; no closed-loop Ours | MPR-XVLA archived as offline no-pass |
| 2026-07-17 | `cb7f379` | post-MPR candidate governance | PRC-XVLA | no independent red-mug/distractor evidence | PRC not elevated |
| 2026-07-17..18 | `9f9b6f6`..`ca5f8cf` | post-MPR X-VLA residual scans | LIBERO-10/goal/object/spatial | goal/object saturated; spatial task5 solved by OpenVLA INT4 | no Ours target |
| 2026-07-18 | `8da1108`..`cd38532` | LIBERO-90 identity `20260724` scan and headroom | X-VLA LIBERO-90 | tasks81/83 failed X-VLA/Base but headroom not verified | no Ours target |
| 2026-07-18 | local ignored | LIBERO-90 identity `20260725` task75 diagnostic | X-VLA task75 | X-VLA/Base shared residual and task-level headroom; second-prior missing | unfinished local work recorded; no new run in audit |

## 6. Valid Scientific Kills

The 28 valid scientific kills or bounded no-pass closures are valid only at their scoped claims. None proves a broad family impossible.

| Method | Exact metric | Strongest baseline/control | Ablation evidence | Sample size | Confidence/paired evidence | Why closure is justified |
|---|---|---|---|---|---|---|
| TCA-Map | target-prior positive, 7D head lost to mean | mean/MLP style baseline | action head failed | offline historical | NOT_RECORDED | representation did not survive action/control path |
| ActionMap | method 0.52993 vs mean 0.46677, MLP 0.50193 | MLP/mean | top1 0.0185 | 1008 train, 432 eval | offline split | learned selector worse than simple controls |
| CSS-Shield | no wrong-target gain; intervention 1.0 | safety-only | over-intervention | historical | NOT_RECORDED | shield collapsed to acting everywhere |
| ExecSpec-Repair | full 17/19, affine 17/19 | diagonal affine | full no better | 19 exact-init replays | paired replay | spec repair not better than simple calibration |
| Phase retiming | 0/9 recovery | raw/best simple | no recovery | 9 cases | direct | no case recovered |
| TL-ChunkRepair | violations down 8/8, safe success 0/8 | no repair 1/1 | repair harms success | 8 | direct | metric improvement did not translate to success |
| PRISM | canonical 0.474066 vs best PRISM 0.436356 | canonicalization | sensitivity weakened | historical | NOT_RECORDED | transform did not beat simpler canonical route |
| PatchGuard | 0.13356 vs adv LoRA 0.142803 and cutout 0.02973 | generic controls | cutout strong | historical | NOT_RECORDED | guarded patch not competitive |
| TG-7D | L2 0.740922 vs canonical 0.587661 | canonical/LoRA | residual worse | heldout | offline split | adapter failed action axis |
| FCAR | full 0.100145 vs static 0.09118, rank4 0.07619 | rank4/static | tiny gate insufficient | 120/40/40 | offline split | not robust to controls |
| PhaseBarrier | full 0/20, Base 8/20, ablation 9/20 | Base/ablation | ablation beats full | 100 paired total | repaired canonical | original positive invalid; repair killed component |
| FEDO | clean full 0 vs frozen 4 | clean frozen | retention collapse | Stage A | paired | catastrophic retention failure |
| PTC | full 0/10 vs Base 3/10 | Base | mechanism active | 10 | paired | full failed while Base succeeded |
| SACF | full 0/10 vs Base 7/10 | Base | mechanism active | 10 | paired | catastrophic harm |
| OCFN | full 26/80 vs zero-noise 27/80 | zero-noise | no useful gain | 80 | expanded paired | useful gain excluded |
| CBFD | full 0/10 vs Base 7/10 | Base | mechanism valid | 10 | paired | catastrophic harm |
| SCVC | full 11/40 vs shifted Base 20/40 | shifted Base | full worse | 40 | CI [-0.425,-0.025] | positive direction excluded |
| PSE | full 50/80 vs bright-single 51/80 | bright-single | no useful gain | 80 | upper CI excludes useful gain | no improvement after expansion |
| RCV | full 20/40, stateless/no-context 24/40 | stateless/no-context | ablations beat full | 40 | paired | retrieval explanation not needed |
| FANG | full 11/40, Base 16/40, AFIL 15/40, ablation 11/40 | Base/AFIL | ablation ties full | 40 | paired | guidance did not add value |
| EvoState | 4,221 pairs, improvement 0.024689 < 0.05 | actionless baseline | mechanism/data valid | 4,221 | preregistered threshold | scoped design stop |
| RAC | full/Base/proxy 1/40, ablation/inverse 2/40 | ablation/inverse | controls beat | 40 | paired | reflective correction not useful |
| MTF | full 26/40 vs no-retention 32/40 | no-retention | ablation beats full | 40 | CI [-0.275,-0.025] | retention harmful |
| DAGR | full 18/40 vs Base 28/40, heuristic 24/40 | Base/heuristic | ablation/prior proxy weaker | 40 | CI full-base [-0.4,-0.1] | guidance lost to Base/simple |
| MARC | full 0/10 vs Base 8/10, no-gate/static 7/10 | Base | simple controls strong | 10 | paired | catastrophic |
| EAC | full 29/40 vs Base/AAC/ablation 30/40 | Base/AAC/ablation | ablation beats by one | 40 | paired | Stage A promise did not survive fair Stage B |
| BR-XVLA | closed-loop residual screen: prior failed, BR primary failed, uniform ablation succeeded | uniform X-VLA LoRA ablation | uniform solved the same identity | 1 exact residual identity after offline gate | frozen manifest | selected weighting failed the mechanism-specific claim and lost to its required ablation |
| MPR-XVLA | repaired offline selector: primary phase-1 loss 0.8785358916 vs uniform 0.8785358369 | uniform X-VLA LoRA ablation | primary did not beat uniform | 24 validation chunks after 64-step two-arm training | frozen selector | offline gate blocked closed-loop Ours; retuning is disallowed |

## 7. Non-Scientific Failures

These 43 failures/blockers are not proof that a broad scientific family is impossible.

| Class | Routes | Why not a scientific kill |
|---|---|---|
| IMPLEMENTATION_OR_OPTIMIZATION_FAILURE | custom SmolVLA 7D adapter, COVI, IARC, FAMR, SPARC, HEST, HASTE, KITE, VDR, RAP, AMP, AFID, CSPR, MCI, R2R-OFT | local realization, objective scale, or validation/action-delta gate failed before a fair closed-loop formulation test |
| DATA_OR_SUPERVISION_FAILURE | SafeTrace, CensorCredit, G3P, PESA, NICE, S2C, MHS, DCCG | labels, contrast, caches, or targets collapsed |
| CONDITION_TOO_SEVERE_OR_RESOURCE | LIFT, OpenPI/pi0.5, PCD, RIPT-VLA, VLA-GSE, VLA-0, VLA-JEPA | compute/checkpoint/resource comparability blocked local fair execution |
| NO_USABLE_HEADROOM | PCAV, CFR, TSC, URF | claimed condition did not expose useful residual improvement |
| MEASUREMENT_INVALIDITY | ContactTube | measurement depended on missing/invalid pose or clipping semantics |
| ENVIRONMENT/DEPENDENCY FAILURE | invalid BR-XVLA/X-VLA/OpenVLA attempts, invalid task6 X-VLA launcher, spatial task7 stall | wrong runtime, missing optional packages, WSL background teardown, or long-tail no-result conditions; repaired artifacts are preserved separately |
| DESIGN_OR_PREPROTOTYPE_NO_GO | CCIF, LCG, BRID, CR-LightVLA, ATCD, PRC-XVLA | design or pre-rollout mechanism did not define enough evidence for prototype GO; PRC-XVLA lacked independent red-mug/distractor evidence after MPR no-pass |
| CONTEXT/SESSION INTERRUPTION | prior exhausted Codex thread and this Phase A resumption | operational interruption, not scientific evidence |

## 8. Underpowered, Ambiguous, or Potentially Misclassified Results

| Route | Evidence | Decision | False-negative risk | Bounded reopen? |
|---|---|---|---|---|
| TCA-Select | no meaningful gain, no later official closed-loop | correctly archived | low | No |
| AMP-GD | toy success 1.0 but LIBERO tiny not above safety/random | correctly archived | medium | No |
| ResetSpec-Retarget | global scale success 1, object-relative 0 | correctly archived | medium | No |
| ContactSet | 6 demos; full worse than simple variants | underpowered archive | medium | No |
| DICD | 1/10 vs 2/10 | underpowered Stage A | medium | No |
| GCAP | target-axis full 3 vs Base 4 and Sobel 5; clean full 5 vs Base 1 | underpowered/mixed | medium | No |
| CAVM | full 24/58 vs nearest 23/58 | strongest unresolved near-miss | medium-high | No under current governance |
| CALA | full RMSE 3.1988 vs history-only 3.14397 | possible false negative | medium | Maybe scientifically; not strategically recommended |
| RAR | full RMSE 0.171954 vs zero-residual 0.165597 | possible false negative | medium | Maybe scientifically; not strategically recommended |
| X-VLA task1 residual | X-VLA 6/8 vs Base 3/8; shared failure `20260727`; task-level expert replay positive but same-reset unavailable | correctly continued into BR-XVLA only with caveat | medium | No: BR-XVLA no-pass closed this path |
| BR-XVLA | gradient/training/offline passed, but closed-loop primary failed while uniform ablation succeeded | correctly archived as bounded no-pass | low for frozen BR claim | No retune |
| X-VLA task6 residual | X-VLA 6/8 vs Base 3/8; OpenVLA INT4 0/2; task-level headroom positive | correctly continued into MPR-XVLA only with caveat | medium | No: MPR no-pass closed the selected path |
| MPR-XVLA | primary did not beat uniform ablation in repaired offline selector | correctly archived before rollout | low for frozen MPR claim | No retune |
| X-VLA spatial task5 | X-VLA/Base shared residual with task-level headroom, but OpenVLA INT4 solved exact residual 1/1 | correctly removed as Ours target | low | No |
| X-VLA LIBERO-90 tasks81/83 | X-VLA/Base shared residuals but expert headroom not verified | correctly removed as Ours target | medium only if headroom method is later repaired | No under current evidence |
| Local X-VLA LIBERO-90 task75 | X-VLA/Base shared residual, task-level headroom positive, same-reset unavailable; second-prior screen missing | unfinished local diagnostic, not a method | medium | Yes: finish second-prior gate only |

## 9. Positive Signals and Near-Misses

| Rank | Route | Positive evidence | Novelty strength | Closed-loop evidence | Strongest comparison | Why not advanced | Reusable? |
|---:|---|---|---|---|---|---|---|
| 1 | CAVM | best Ours point estimate 24/58 | moderate memory mechanism | expanded Stage B | nearest 23/58, Base 22/58 | one-episode gain, no third expansion, no second backbone | concept reusable; result closed |
| 2 | MPR-XVLA task6 path | matched Base/Prior residual, task-level headroom, OpenVLA INT4 no-solve, data audit passed | modest prior-extension novelty | no closed-loop Ours; offline only | uniform ablation slightly better | offline selector blocked rollout | reusable as evidence that uniform controls are essential |
| 3 | Local task75 path | X-VLA/Base shared residual plus task-level headroom on LIBERO-90 | unknown until second-prior screen | prior/Base only; no Ours | second-prior not run | unfinished; cannot rank as paper result | reusable only if second-prior fails cleanly |
| 4 | EAC | Stage A promise, Stage B near-tie | moderate adaptive controller | Stage B | Base/AAC/ablation 30/40 vs full 29/40 | controls beat by one | useful control set |
| 5 | OpenVLA/LightVLA/X-VLA complementarity | official priors solved or exposed different residuals | prior-comparison signal | prior diagnostics | X-VLA task8 8/8; OpenVLA spatial task5 1/1 | prior successes remove Ours targets | informs prior-first search |

Positive offline metrics alone are not ranked as paper results. CensorCredit's apparent positive prototype is invalid because later evidence found identical censored/uncensored labels and heads.

## 10. External-Prior Comparison Audit

Among the now 50 selected formal Ours methods, official external-prior reproduction count remains 0 as a successful Ours-vs-official-prior result. Proxy comparison count remains 26 for the original 47 formal methods; `R2R-OFT` used a real OpenVLA-OFT prior condition but failed before closed-loop Ours comparison; `BR-XVLA` used a real X-VLA prior and failed its frozen closed-loop residual screen; `MPR-XVLA` used a real X-VLA prior plus OpenVLA INT4 second-prior no-solve evidence and failed its offline selector. `OCB-XVLA` and `PRC-XVLA` are counted as unselected candidate routes, not selected formal methods. No-external-prior experiment count for formal Ours methods is therefore still large and reviewer-facing.

Epoch 5 materially improved process quality: OpenVLA-OFT, LightVLA, RIPT-VLA, VLA-GSE, X-VLA, VLA-0, and VLA-JEPA were inspected or run before new Ours design. The important result is negative for overclaiming: X-VLA solved the task8 residual 8/8, OpenVLA-OFT INT4 solved spatial task5 1/1, and MPR-XVLA did not beat its uniform ablation. Current local task75 is a prior-grounded residual candidate only in the diagnostic sense; it still lacks the required second-prior screen and has no Ours evidence.

Published numbers were generally not treated as direct baselines in later reports, but many methods used transparent proxies rather than official code/checkpoints. The current branch is the first serious correction toward official-prior-first comparison, and its lesson is stricter than the previous audit: prior success, prior failure, uniform-ablation success, and diagnostic headroom are not Ours evidence.

## 11. LoRA / Low-Compute Strategy Audit

LoRA/QLoRA was intended as compute infrastructure, not the contribution. The campaign partially respected this, but it also underused official LoRA early and overemphasized lightweight frozen-policy attachments later.

Verified adapter facts: official SmolVLA rank-4 LoRA targeted `lm_expert` q/v plus state/action projections and had 185,664 trainable parameters, 0.0412% of 450,231,840. Rank-16 feasibility had 742,656 trainable. PatchGuard rank-4 had 9,984 trainable parameters. Custom 7D rank-4/rank-8 adapters had 128,007/131,975 trainable parameters. TG-7D rank-4 had 295,623 trainable. `R2R-OFT` used QLoRA/LoRA as implementation infrastructure for a prior extension but failed validation before closed-loop rollout. `BR-XVLA` and `MPR-XVLA` used X-VLA PEFT rank 8 / alpha 16 two-arm LoRA gates with primary weighting and mandatory uniform-weight ablations; both failed the mechanism-specific comparison. Quantized OpenVLA-OFT INT4 and X-VLA are inference diagnostics, not QLoRA training results.

Failure attribution: many failures were not caused by LoRA capacity because they never reached a fair capacity test. Early failures were often action-interface and checkpoint persistence problems; later failures often came from data/headroom/objective-scale collapse. In Epoch 5, LoRA was used more appropriately as controlled infrastructure, but BR-XVLA and MPR-XVLA show a new reviewer-killer: uniform LoRA adaptation can match or beat the proposed weighting. Future LoRA use should be conditional: only after Base/Prior/headroom/second-prior are proven and only when the scientific mechanism is separable from PEFT and from uniform adaptation.

## 12. Research-Funnel Statistics

| Funnel quantity | Count | Notes |
|---|---:|---|
| Total routes/diagnostic routes in ledger | 95 | previous 89 plus task6 residual, `MPR-XVLA`, `PRC-XVLA`, spatial task5, LIBERO-90 tasks81/83, and local task75 |
| Formal selected Ours methods | 50 | previous 49 plus selected `MPR-XVLA`; `OCB-XVLA` and `PRC-XVLA` are unselected candidate routes |
| Current local residual candidates | 1 unfinished | task75 has Base/Prior residual and task-level headroom but no second-prior screen |
| Implemented routes | 84 | code/runner/local execution evidence; local task75 is diagnostic only |
| Trained/checkpointed routes | 34 | includes bounded `R2R-OFT`, `BR-XVLA`, and `MPR-XVLA`; prior diagnostics without training excluded |
| Formal Stage A count | 17 | unchanged |
| Formal Stage B count | 10 | unchanged |
| Route-level Stage-A-equivalent count | 19 | formal 17 plus PhaseBarrier/CensorCredit historical prototypes |
| Route-level Stage-B-equivalent count | 11 | formal 10 plus PhaseBarrier repaired prototype |
| Second-backbone Ours count | 0 | OpenVLA/X-VLA are priors, not Ours |
| Official-prior diagnostic routes | 16 | OpenVLA, OpenPI, PCD, LightVLA, RIPT, VLA-GSE, X-VLA, VLA-0, VLA-JEPA plus task1/task6/spatial/LIBERO-90 controls and scans |
| Paper-candidate GO count | 0 | no `PROTOTYPE_GO` |

Loss breakdown: 28 valid scientific kills or bounded no-pass closures; 43 non-scientific failures/blockers; 13 underpowered/unresolved/unfinished rows; 11 infrastructure/diagnostic/no-claim rows. Formal selected proposal to Stage A: 17/50 = 34.0%. Formal selected proposal to Stage B: 10/50 = 20.0%. Stage B to GO: 0/10 = 0%. Formal selected proposal to second-backbone Ours: 0/50 = 0%.

## 13. Compute and Operational Audit

Current campaign-state records 5.21 GPU hours and 14.845 GiB downloaded for an earlier autonomous slice, but repo-wide GPU hours are `NOT_RECORDED`. The repository spans from first commit `07c823d` on 2026-06-27 10:30:15+09:00 to this audit on 2026-07-18 00:48:48+09:00, about 20.6 wall-clock days. Git history contains 870 commits across all refs and 866 ancestors of audit HEAD before this report commit.

Simulator episode lower bound from final artifacts remains at least 3,604 completed non-quarantined route-level episodes before adding all invalid attempts. Epoch 5 adds OpenVLA/SmolVLA residual episodes, short-requery episodes, LightVLA/CR/X-VLA/task1/task6/spatial/LIBERO-90 scans, task-level expert replay, BR-XVLA closed-loop residual episodes, and local task75 diagnostics, but a globally normalized episode total is `NOT_RECORDED`.

Asset/storage notes: `C:\assets\data` and model/checkpoint caches dominate storage. Known large prior assets include OpenVLA-OFT around 15 GiB, X-VLA 3.28 GiB, VLA-0 21.46 GiB, VLA-JEPA 22.96 GiB, and SmolVLA under 1 GiB. Current untracked videos under `rollouts/2026_07_17/` include residual and short-requery OpenVLA videos.

Operational overhead includes context exhaustion, many documentation/report commits, durable worker launchers, invalid/repaired attempts, environment mismatch repairs, and branch proliferation. Duplicate/avoidable reruns include official LoRA drift/regeneration, PhaseBarrier invalid retrain then repair, COVI invalid v1 then repair, PCAV expansion resume, VDR self-worker confusion, RAP/KITE/SPARC launcher issues, wrong-env SmolVLA residual attempts, the X-VLA Base run first attempted in an incompatible OpenVLA environment, two BR-XVLA gradient-smoke dependency failures before model load, a task6 X-VLA WSL-background no-result, wrong-runtime OpenVLA task6 attempts, and the spatial task7 long-tail SIGTERM/no-result.

## 14. Repetition and Search-Space Audit

Recurring families: candidate ranking/verifiers (`TCA-Select`, `PESA`, `PCAV`, `ECHO`); post-hoc residual correction (`FCAR`, `RAR`, `COVI`, `FAMR`, `SPARC`, `CFR`, `URF`, `BRID`, `CSPR`, `MCI`, `R2R-OFT`, `BR-XVLA`, `MPR-XVLA`, `PRC-XVLA`); action filtering/damping (`CSS`, `ExecSpec`, `PTC`, `SACF`, `RAC`, `EAC`, `AMP`, `PhaseBarrier`, `CR-LightVLA`); memory/retrieval (`RCV`, `CAVM`, `MTF`, `DAGR`, `RAP`); visual canonicalization/TTA (`PRISM`, `OCFN`, `SCVC`, `FANG`, `VDR`, `COVI`); temporal/history heads (`DICD`, `PSE`, `CALA`, `HEST`, `HASTE`, `TSC`, `MHS`, `NICE`); supervision/credit (`SafeTrace`, `CensorCredit`, `FEDO`, `G3P`, `ATCD`); representation/action generation (`TCA-Map`, `ActionMap`, `TG-7D`, `CBFD`, `EvoState`, `AFID`, `MCI`).

Common failed assumptions: small frozen-policy attachments would produce publishable gains; action L2/offline probes would predict closed-loop success; retrieval/memory would beat stateless/simple controls; visual canonicalization would fix brittleness without clean-behavior disruption; proxy priors would satisfy reviewer-grade comparison; and in Epoch 5, hand-designed residual-phase weights would beat uniform LoRA adaptation. Epoch 5 improved this by running official priors first, but it has not yet produced Ours evidence.

## 15. Why the Campaign Is Not Finished

Ranked causes by impact:

1. Candidate quality and anchoring: 0/50 selected formal Ours methods have a positive completed official-prior Ours comparison. Evidence: ledger rows 27-95 and section 10.
2. No stable positive problem condition: many late routes found no usable headroom, collapsed labels, objective-scale failure, only task-level rather than same-reset headroom, or a second prior that solved the target. Evidence: PCAV/CFR/TSC/URF, NICE/CensorCredit, MCI, R2R-OFT, BR-XVLA, MPR-XVLA, spatial task5, LIBERO-90 tasks81/83.
3. Repeated narrow method families: residuals, gates, memories, and history heads were renamed more often than core assumptions changed. Evidence: section 14.
4. Pretrained-policy disruption/nonacting mechanisms: many Stage A/B methods lost to Base, ablation, or simple controls. Evidence: RCV, DAGR, MARC, MTF, EAC.
5. Late external-prior comparison: official priors became central only in Epoch 5. Evidence: OpenVLA/LightVLA/X-VLA sequence.
6. Low-compute strategy confusion: LoRA was useful infrastructure but often turned into a bias toward tiny local attachments; BR-XVLA and MPR-XVLA were better prior-anchored but lost to uniform LoRA controls. Evidence: section 11.
7. Underpowered early decisions: DICD, GCAP, CAVM, CALA, and RAR leave false-negative risk but no paper candidate. Evidence: section 8.
8. Documentation/state churn and context interruptions: state JSON lagged behind HEAD, and a previous Codex context exhausted. Evidence: snapshot caveat and this audit request.
9. Hardware/resource limits: RTX 5080 supports lightweight and INT4 diagnostics but not every large prior or full finetune. Evidence: LIFT, OpenPI/PCD/RIPT/VLA-GSE/VLA-0/VLA-JEPA blockers.

Scientific difficulty dominates, but process/governance mattered: broad search before stable official simulator evidence and late proxy-heavy comparisons generated many honest negatives without a reviewer-ready positive. The current local task75 condition is the most concrete live diagnostic thread, but it is still pre-Ours: same-reset headroom is unavailable and the second-prior gate is missing.

## 16. False-Negative Audit

Potential false negatives exist, but none should be reopened immediately except finishing the already-started task75 diagnostic gate. CAVM is strongest historically: 24/58 beat nearest 23/58, Base 22/58, and ablation 21/58, but no third expansion is allowed. CALA and RAR had small offline margins without closed-loop confidence. DICD and GCAP were underpowered Stage A archives, but later related methods tested richer variants.

Reviewer B overreach risk is real mainly for Stage 0 point-estimate or offline stops, not for completed Stage B kills. Later governance improved classification: MCI and R2R-OFT are implementation/validation failures, not scientific kills. BR-XVLA and MPR-XVLA should not be reopened merely because they were close to official-prior residuals: both were tested against required uniform ablations and did not pass. Do not reopen a route merely because the campaign lacks a positive result.

## 17. Paper-Readiness Checklist

Nearest live route for checklist: local task75 is the nearest unfinished problem condition, not a selected Ours method. It has a shared X-VLA/Base residual and task-level headroom, but the second-prior screen and any candidate design are missing.

| Requirement | Status | Gap |
|---|---|---|
| Defensible novelty | MISSING | task75 has no selected Ours mechanism yet |
| SmolVLA Base vs Base + Ours | MISSING | Base 0/1 exists for task75; no Ours |
| Closest prior vs Ours | MISSING | X-VLA failed task75; no Ours |
| Key ablation | MISSING | no selected method or ablation |
| Relevant simple control | MISSING | second-prior screen missing; no simple control selected |
| Clean retention | MISSING | not evaluated |
| Adequate paired statistics | MISSING | only single task75 Base/Prior/headroom diagnostic so far |
| Quantized OpenVLA-OFT INT4 + Ours | MISSING | task75 OpenVLA-OFT INT4 screen not run; no Ours |
| Second claim-specific condition | MISSING | not selected |
| Efficiency | MISSING | no trained/evaluated method |
| Reproducibility | PARTIAL | prior/Base/headroom artifacts exist; task75 second-prior/Ours artifacts do not |
| Figure/table-ready artifacts | MISSING | no paper package |

Exact gap to `READY_TO_DRAFT_RAL_PAPER_PACKAGE`: no `PROTOTYPE_GO`, no selected task75 Ours method, no second-prior screen for task75, no optimizer-step training for any live method, no offline validation pass for any live method, no closed-loop Ours result, no official-prior win, no positive Stage B, no second-backbone Ours result, no second condition, and no figure/table package.

## 18. Missed or Unreported Events

Events likely easy to miss because the previous thread exhausted context:

- CSPR and MCI completed just before the first full audit; both are implementation failures, not scientific kills.
- The first full audit at `b0ecb6e` recommended strategy reset; Epoch 5 then moved to official-prior-first.
- OpenVLA-OFT hard-slice was prior-positive but saturated at 20/20; residual OpenVLA-OFT was 14/16 vs SmolVLA Base 7/16.
- Task-level expert replay was positive for task8, but not same-reset.
- `R2R-OFT` was generated, selected, audited, trained for bounded steps, and stopped by offline/action-delta validation before closed-loop rollout.
- Short 4-step OpenVLA requery was run and did not beat original OpenVLA.
- OpenPI/pi0.5, PCD, RIPT-VLA, VLA-GSE, VLA-0, and VLA-JEPA were recorded as fallback or unselected prior/resource routes.
- LightVLA was runnable and complementary to OpenVLA on task8; `CR-LightVLA` and `ATCD` did not reach prototype GO.
- X-VLA solved the old task8 residual 8/8; this invalidates task8 as an Ours target.
- X-VLA task1 residual was then found and matched against SmolVLA Base; shared failure `20260727` became the clean headroom target.
- Task1 expert replay found positive task-level headroom on nearest HDF5 demo `demo_48`, but no same-reset HDF5 demo hash matched the benchmark reset.
- The task1 basket data audit passed and exactly two candidates were generated: selected `BR-XVLA` and unselected `OCB-XVLA`.
- BR-XVLA later passed the gradient smoke, trained two rank-8 X-VLA LoRA arms for 64 steps, passed an offline screen only by an extremely tiny margin, and then failed the frozen closed-loop residual screen while the uniform ablation succeeded.
- Post-BR-XVLA X-VLA scans found task6; task6 had matched Base/Prior residual, task-level expert headroom, a spatial data audit, and OpenVLA-OFT INT4 second-prior no-solve evidence.
- MPR-XVLA was selected over PRC-XVLA, passed preoptimizer and one-step smokes, trained both arms for 64 steps, then failed repaired offline selection because primary did not beat uniform.
- PRC-XVLA was explicitly not elevated after MPR no-pass because LIBERO-90 mug shards gave no independent red-mug/distractor confusion evidence.
- Post-MPR scans saturated LIBERO goal/object and most LIBERO-90 shards; spatial task5 was a shared residual but OpenVLA-OFT INT4 solved it, so it was removed as an Ours target.
- LIBERO-90 tasks81/83 were clean X-VLA/Base residuals but expert headroom was not verified; these are no-headroom diagnostics, not Ours targets.
- Local ignored task75 work exists after pushed HEAD: X-VLA/Base shared residual and task-level headroom are recorded, but no second-prior screen was run before this audit.
- The first SmolVLA task1 run used an incompatible OpenVLA environment and failed before rollout; the official-env rerun is the valid Base result.
- Untracked rollout videos remain under `rollouts/2026_07_17/` and `rollouts/2026_07_18/`; this audit did not move, stash, or delete them.

## 19. Recommended Strategic Decision

Recommendation: `CONTINUE_CURRENT_CYCLE`.

Justification: the prior `RESET_CANDIDATE_SELECTION_STRATEGY` recommendation has already been acted on by Epoch 5 official-prior-first. The current state is not a generic local-method cycle; it is an official-prior-first residual search with one unfinished local diagnostic thread. Continue only to complete the task75 second-prior gate or, if the user rejects local ignored artifacts as too stale, select a new residual source/identity/prior ecosystem. Do not retune BR-XVLA or MPR-XVLA and do not design Ours for task75 until the second-prior gate is complete.

## 20. Exact Resume Plan

`Exact Next Codex Prompt After User Review`

```text
Resume the autonomous VLA research campaign in C:\Users\jiheo\tca_map after reviewing reports/autonomous_research_full_history_audit.md.

Branch: codex/epoch5-official-prior-first
Last scientific HEAD before the Phase A audit report commit: cd3853285a5dfabdee1ab21524392acf9ad2bc64
Current scientific state: Epoch 5 official-prior-first, pushed LIBERO-90 tasks81/83 no-headroom state, with ignored/local LIBERO-90 task75 residual/headroom artifacts and no task75 second-prior screen yet
Current pushed decision: LIBERO90_TASKS81_83_HEADROOM_NOT_VERIFIED_NO_OURS_TARGET / POST_MPR_XVLA_IDENTITY_GRID_NO_FRESH_TARGET
Current local task75 decision: TASK75_TASK_LEVEL_EXPERT_HEADROOM_POSITIVE_SAME_RESET_UNAVAILABLE, second-prior screen missing
Previous selected methods: BR-XVLA and MPR-XVLA
Previous decisions: BR-XVLA validation no-pass; MPR-XVLA offline no-pass; PRC-XVLA not elevated
Selected audit recommendation: CONTINUE_CURRENT_CYCLE

Exact next scientific action:
Complete exactly one missing diagnostic before any Ours design: run the task75 second-prior screen for LIBERO-90 identity 20260725 against Quantized OpenVLA-OFT INT4 or another preregistered comparable official prior if OpenVLA-OFT does not support the suite. If the second prior solves task75, record no Ours target. If the second prior cleanly fails with no infrastructure failure, then and only then decide whether task75 authorizes candidate generation under official-prior-first governance. Do not train, retune, or launch Ours in this step.

Prohibited repeats:
Do not rescue or retune MCI-VLA, CSPR-VLA, R2R-OFT, CR-LightVLA, ATCD, BR-XVLA, MPR-XVLA, or PRC-XVLA. Do not generate a task75 Ours candidate before the second-prior gate. Do not treat prior success/failure, uniform-ablation success, diagnostic headroom, or prior scans as Ours. Do not use the old task8 residual, spatial task5, or LIBERO-90 tasks81/83 as Ours targets. Do not claim INT4 OpenVLA-OFT is a full-precision reproduction. Do not switch branches, stash, reset, clean, or delete untracked rollout artifacts without user approval.

Time-to-evidence requirement:
Produce one durable second-prior answer with artifact path, result JSON, stdout/stderr logs, heartbeat, exit code or process status, suite/task/reset identity, success/failure, infrastructure-failure classification, and explicit training/optimizer/checkpoint/Ours-design false booleans. Keep `reports/autonomous_compact_handoff.md` under 250 lines if updated.

LoRA role:
LoRA/QLoRA is only implementation infrastructure. For the next step, no LoRA training or PEFT attachment is authorized; the task75 gate is prior-only.

Reviewer false-negative safeguards:
Do not classify task75 solved or failed from unsupported-suite errors, wrong-runtime failures, missing optional packages, or stale partial files. Preserve invalid/repaired attempts separately and distinguish unsupported second-prior infrastructure from a clean policy failure.

Conditions for implementation and rollout:
Task75 Ours implementation may start only after matched Base/Prior residual, task-level headroom caveat, second-prior clean no-solve, no privileged inference signal, prior fairness, resource risk, and frozen decision thresholds are documented. Closed-loop Ours rollout may start only after a selected LoRA/QLoRA training artifact passes offline validation, checkpoint reload, bounded action deltas, key ablation, simple control, clean retention, and exact paired manifest gates.
```
