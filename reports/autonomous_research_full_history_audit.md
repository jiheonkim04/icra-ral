# Autonomous VLA Research — Full History Audit

This is the mandatory Phase A audit before any resumed VLA research. It refreshes the earlier audit at `b0ecb6e`, the Epoch 5 refresh at `541c822`, and the post-audit refresh at `f0e555b` using the live repository state at pushed scientific HEAD `0b3697f697f8ab83f80f568ea85e8b4855709d52` plus local uncommitted BR-XVLA gradient-smoke work. Evidence precedence: current local git/HEAD, `reports/current_research_governance.md`, current result artifacts, current campaign-state JSON, git history across branches, project state / next actions / decision logs, historical reports, then old prompts. Missing facts are recorded as `NOT_RECORDED`.

## 1. Executive Summary

No paper-ready method exists. No valid `PROTOTYPE_GO` method exists. The repository contains substantial reusable infrastructure, many valid scientific kills, many invalid or pre-rollout stops, and several official-prior diagnostics, but it does not contain `READY_TO_DRAFT_RAL_PAPER_PACKAGE`.

This audit finds 89 distinct research routes or diagnostic prior routes: the previous 87-route ledger plus the two task-1 X-VLA Ours candidates, `BR-XVLA` and unselected `OCB-XVLA`. Formal selected Ours methods are 49 after adding selected `BR-XVLA` to the previous 48-method autonomous campaign. Implemented route count is 79/89 using code, runner, or local execution evidence. Trained/checkpointed route count is 32. Closed-loop Stage A count remains 17 formal autonomous methods, or 19 route-level methods when historical prototypes are included. Stage B count remains 10 formal methods, or 11 route-level methods with the repaired PhaseBarrier prototype. Second-backbone Ours count remains 0.

Outcome totals: 26 valid scientific kills, 41 non-scientific failures or resource/preimplementation blockers, and 11 underpowered or unresolved results. The increase since the previous audit comes from Epoch 5: task-1 headroom is task-level positive but same-reset unavailable; the task-1 basket data audit passed; exactly two BR/OCB-XVLA candidates were generated with `BR-XVLA` selected; the BR-XVLA training spec and X-VLA-format data-adapter smoke passed; and the local one-batch gradient smoke is blocked before model load/backward by an optional dependency shim issue (`fastapi.__spec__ is None`). This is an environment/dependency blocker, not a scientific kill.

The strongest Ours result remains `CAVM-VLA`: full 24/58 versus nearest-success memory 23/58, Base 22/58, and no-contrast 21/58 after one allowed expansion. It is a near-miss, not paper-ready: the advantage is one episode, no third expansion is allowed, and there is no second-backbone or official-prior confirmation. The strongest official-prior result is X-VLA solving the earlier task-8 residual 8/8; that removes that Ours target rather than creating a paper method.

Current active state: Epoch 5, pushed stage `epoch_5_br_xvla_data_adapter_smoke_complete`, pushed decision `BR_XVLA_DATA_ADAPTER_SMOKE_PASS_GRADIENT_SMOKE_PENDING`, with local uncommitted gradient-smoke attempt `BR_XVLA_GRADIENT_SMOKE_BLOCKED_OR_FAIL`. Previous Ours method `MCI-VLA` remains closed as `MCI_STAGE_0_IMPLEMENTATION_FAILURE`. The next scientific action, after user review only, is to repair/record the BR-XVLA no-optimizer gradient-smoke dependency boundary and rerun only that one-batch gate; optimizer steps, checkpointing, and closed-loop Ours rollout remain unauthorized.

Main reasons the campaign is not paper-ready: no Ours method beats Base, closest prior/proxy, key ablation, and simple reviewer-killer control in a valid Stage B; official-prior comparison arrived late and is still diagnostic; many late routes failed from data/headroom/objective-scale/resource issues before rollout; the search repeatedly favored small frozen-SmolVLA attachments; and no same-method Ours evidence exists on Quantized OpenVLA-OFT INT4 or another second backbone.

## 2. Audit Snapshot

| Field | Value |
|---|---|
| Snapshot timestamp | `2026-07-17T18:44:06+09:00` |
| Repository | `C:\Users\jiheo\tca_map` |
| Current branch | `codex/epoch5-official-prior-first` |
| Scientific HEAD | `0b3697f697f8ab83f80f568ea85e8b4855709d52` |
| HEAD subject | `Record BR-XVLA data adapter smoke` |
| Git status | `## codex/epoch5-official-prior-first...origin/codex/epoch5-official-prior-first`; untracked `rollouts/2026_07_17/`, `tca_map/xvla_task1/gradient_smoke.py`, `tests/test_br_xvla_gradient_smoke.py` |
| `main` HEAD | `8dc4de2fdbf576ace8bdf3699d190b761553c1fa` |
| Active Windows research Python | none detected |
| Active WSL research Python | none detected; detached WSL PID `23232` is no longer alive |
| Worker classification | `NO_ACTIVE_SCIENTIFIC_WORKER_AT_AUDIT_SNAPSHOT` |
| CUDA/GPU snapshot | RTX 5080, 16,303 MiB total, 1,666 MiB used, 14% utilization; no research Python compute process detected |
| RAM snapshot | Windows about 24,288,100 KiB total / 10,076,748 KiB free; WSL 11 GiB total / 10 GiB available / 3.0 GiB swap free |
| Disk snapshot | Windows C: 737,070,182,400 bytes used / 262,064,406,528 free; WSL `/mnt/c`: 931G total / 687G used / 245G available; WSL `/`: 1007G total / 86G used / 870G available |
| Current epoch/cycle/stage | Epoch 5, cycle 0, pushed `epoch_5_br_xvla_data_adapter_smoke_complete`; local gradient-smoke attempt finished blocked |
| Current prior sequence | OpenVLA-OFT first, LightVLA second, X-VLA third |
| Current decision | pushed `BR_XVLA_DATA_ADAPTER_SMOKE_PASS_GRADIENT_SMOKE_PENDING`; local `BR_XVLA_GRADIENT_SMOKE_BLOCKED_OR_FAIL` |
| Current next action | after user review only, repair/record optional-dependency shim boundary and rerun one-batch no-optimizer BR-XVLA gradient smoke; no optimizer/checkpoint/rollout |
| Current reports | `reports/epoch5_prior_reproduction_result.md`, `reports/epoch5_prior_reproduction_result.json`, `reports/autonomous_compact_handoff.md` |
| Current X-VLA result | `runs/xvla_prior/diagnostic_xvla_libero10_task1_id20260724_20260731_20260717T1729KST/result.json` |
| Current Base result | `runs/xvla_prior/diagnostic_smolvla_base_libero10_task1_id20260724_20260731_officialenv_20260717T1739KST/result.json` |
| Current BR-XVLA adapter smoke | `runs/xvla_prior/br_xvla_data_adapter_smoke_20260717T183355KST/result.json` |
| Local BR-XVLA gradient-smoke attempt | `runs/xvla_prior/br_xvla_gradient_smoke_20260717T184139KST/result.json`, SHA-256 `07562DD5A031E1ADFDA4D10BCB31C6C80308194D568553C62CC8C18750E8D1EC` |
| State-file caveat | `reports/autonomous_until_paper_state.json` and `reports/autonomous_ral_campaign_state.json` still contain stale `current_commit` values from `b0ecb6e`; live HEAD and current artifacts are authoritative |

Current task-1 diagnostic: X-VLA 6/8 with failures `20260725` and `20260727`; matched SmolVLA Base 3/8 with failures `20260724`, `20260727`, `20260728`, `20260729`, `20260730`. Both succeed on `20260726` and `20260731`; X-VLA-only successes are `20260724`, `20260728`, `20260729`, `20260730`; Base-only success is `20260725`; the clean shared residual is `20260727`. Task-level expert headroom is positive on nearest HDF5 demo `demo_48`, but same-reset HDF5 headroom is unavailable because no HDF5 demo init-state hash matches the benchmark residual init-state hash. BR-XVLA has not loaded a model, attached PEFT, run backward, created an optimizer, written a checkpoint, or run closed-loop Ours evaluation.

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
| BR-XVLA bounded spec and data adapter | `04c5239`, `0b3697f` | `runs/xvla_prior/epoch5_br_xvla_training_spec_v1.json`, `runs/xvla_prior/br_xvla_data_adapter_smoke_20260717T183355KST/result.json` | official X-VLA PEFT rank-8 spec frozen; X-VLA reader smoke passes with local `mmengine.fileio` shim | valid pre-optimizer infrastructure | selected method not yet trained |
| BR-XVLA gradient-smoke local blocker | local uncommitted | `runs/xvla_prior/br_xvla_gradient_smoke_20260717T184139KST/result.json` | failed before model load because `fastapi.__spec__ is None`; no forward/backward/optimizer/checkpoint | dependency/environment blocker | no scientific method evidence |
| Partial/resume/durable worker infrastructure | many | `reports/*partial*.json`, `runs/*heartbeat*`, `*.pid` | 52 tracked report partials, 69 heartbeat files, 93 PID-like files | valid operational infrastructure | infrastructure only |

Tracked inventory at audit refresh: 2,666 tracked files; 1,776 under `reports`, 281 under `scripts`, 244 under `tests`, 210 under `tca_map`, 103 under `runs`, and 21 under `rollouts`. Machine-readable inventory from the prior audit remains the latest normalized count: 152 tracked `reports/*result.json`, 72 report manifests, 52 report partials, 196 `reports`/`runs` result JSONs, 57 exit-code files, 69 heartbeat files, and 93 PID/PID-like files.

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
| 88 | epoch5 | BR-XVLA | basket-remaining reweighted X-VLA | prior extension | X-VLA-Libero | `ae9505e`..`0b3697f`; local gradient attempt | Yes | No | No | No | Partial | No | No | No | data-adapter smoke passed; local gradient smoke blocked before model load/backward | active dependency blocker | INCONCLUSIVE | No | Yes | `reports/epoch5_task1_ours_candidate_design.md`, `runs/xvla_prior/br_xvla_data_adapter_smoke_20260717T183355KST/result.json`, `runs/xvla_prior/br_xvla_gradient_smoke_20260717T184139KST/result.json` |
| 89 | epoch5 candidate | OCB-XVLA | object-contrast basket X-VLA | prior extension candidate | X-VLA-Libero | `ae9505e` | No | No | No | No | No | No | No | No | candidate score 73/100, not selected | not selected | INCONCLUSIVE | No | No | `reports/epoch5_task1_ours_candidate_design.md` |

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
| 2026-07-17 | `04c5239`, `0b3697f` | BR-XVLA spec and adapter smoke | BR-XVLA | no-training spec frozen; X-VLA reader smoke passed | pre-optimizer infrastructure ready |
| 2026-07-17 | local uncommitted | no-optimizer gradient-smoke attempt | BR-XVLA | blocked before model load/backward by `fastapi.__spec__ is None` | dependency blocker, not scientific evidence |

## 6. Valid Scientific Kills

The 26 valid scientific kills are valid only at their scoped claims. None proves a broad family impossible.

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

## 7. Non-Scientific Failures

These 41 failures/blockers are not proof that a broad scientific family is impossible.

| Class | Routes | Why not a scientific kill |
|---|---|---|
| IMPLEMENTATION_OR_OPTIMIZATION_FAILURE | custom SmolVLA 7D adapter, COVI, IARC, FAMR, SPARC, HEST, HASTE, KITE, VDR, RAP, AMP, AFID, CSPR, MCI, R2R-OFT | local realization, objective scale, or validation/action-delta gate failed before a fair closed-loop formulation test |
| DATA_OR_SUPERVISION_FAILURE | SafeTrace, CensorCredit, G3P, PESA, NICE, S2C, MHS, DCCG | labels, contrast, caches, or targets collapsed |
| CONDITION_TOO_SEVERE_OR_RESOURCE | LIFT, OpenPI/pi0.5, PCD, RIPT-VLA, VLA-GSE, VLA-0, VLA-JEPA | compute/checkpoint/resource comparability blocked local fair execution |
| NO_USABLE_HEADROOM | PCAV, CFR, TSC, URF | claimed condition did not expose useful residual improvement |
| MEASUREMENT_INVALIDITY | ContactTube | measurement depended on missing/invalid pose or clipping semantics |
| ENVIRONMENT/DEPENDENCY FAILURE | BR-XVLA local gradient-smoke attempt | optional server/import shim left `fastapi.__spec__` unset, so Transformers import failed before model load, PEFT attachment, forward, backward, optimizer, or checkpoint |
| DESIGN_OR_PREPROTOTYPE_NO_GO | CCIF, LCG, BRID, CR-LightVLA, ATCD | design or pre-rollout mechanism did not define enough evidence for prototype GO |
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
| X-VLA task1 residual | X-VLA 6/8 vs Base 3/8; shared failure `20260727`; task-level expert replay positive but same-reset unavailable | correctly continued into BR-XVLA only with caveat | medium | Yes: continue current BR-XVLA gradient gate only |
| BR-XVLA | selected after task1 headroom/data gates; spec and data-adapter smoke passed; gradient smoke blocked before model load/backward | unresolved active pre-optimizer route | high until gradient/validation/closed-loop evidence exists | Yes: repair/rerun no-optimizer gradient smoke only |

## 9. Positive Signals and Near-Misses

| Rank | Route | Positive evidence | Novelty strength | Closed-loop evidence | Strongest comparison | Why not advanced | Reusable? |
|---:|---|---|---|---|---|---|---|
| 1 | CAVM | best Ours point estimate 24/58 | moderate memory mechanism | expanded Stage B | nearest 23/58, Base 22/58 | one-episode gain, no third expansion, no second backbone | concept reusable; result closed |
| 2 | BR-XVLA task1 path | official prior improves Base 6/8 vs 3/8; task-level headroom/data audit positive; adapter smoke passed | modest prior-extension novelty | prior/Base closed-loop only; no Ours rollout | X-VLA 6/8, Base 3/8 | gradient smoke blocked before model load/backward; no optimizer, checkpoint, validation pass, or closed-loop Ours evidence | reusable as current active problem condition |
| 3 | EAC | Stage A promise, Stage B near-tie | moderate adaptive controller | Stage B | Base/AAC/ablation 30/40 vs full 29/40 | controls beat by one | useful control set |
| 4 | RCV | full beat Base 20/40 vs 14/40 | moderate retrieval | Stage B | stateless/no-context 24/40 | ablations explain benefit | diagnostics reusable |
| 5 | OpenVLA/LightVLA complementarity | OpenVLA and LightVLA solved each other's task8 failures | prior-comparison signal | prior diagnostics | oracle union 8/8 | ATCD teacher signal below threshold, X-VLA later solved task8 | informs prior-first search |

Positive offline metrics alone are not ranked as paper results. CensorCredit's apparent positive prototype is invalid because later evidence found identical censored/uncensored labels and heads.

## 10. External-Prior Comparison Audit

Among the now 49 selected formal Ours methods, official external-prior reproduction count remains 0 as an Ours-vs-official-prior result. Proxy comparison count remains 26 for the original 47 formal methods; `R2R-OFT` used a real OpenVLA-OFT prior condition but failed before closed-loop Ours comparison, and `BR-XVLA` uses a real X-VLA prior condition but has not passed even the no-optimizer gradient gate. `OCB-XVLA` is counted as an unselected candidate route, not a selected formal method. No-external-prior experiment count for formal Ours methods is therefore still large and reviewer-facing.

Epoch 5 materially improved process quality: OpenVLA-OFT, LightVLA, RIPT-VLA, VLA-GSE, X-VLA, VLA-0, and VLA-JEPA were inspected or run before new Ours design. The important result is negative for overclaiming: X-VLA solved the task8 residual 8/8, so that condition cannot be used as an Ours target. Current task1 is a better prior-grounded residual candidate and has task-level headroom, but it still lacks same-reset upper-bound evidence and all Ours evidence beyond data-adapter compatibility.

Published numbers were generally not treated as direct baselines in later reports, but many methods used transparent proxies rather than official code/checkpoints. The current branch is the first serious correction toward official-prior-first comparison; it must finish the BR-XVLA gradient and offline gates before any optimizer-step scale-up or closed-loop Ours rollout.

## 11. LoRA / Low-Compute Strategy Audit

LoRA/QLoRA was intended as compute infrastructure, not the contribution. The campaign partially respected this, but it also underused official LoRA early and overemphasized lightweight frozen-policy attachments later.

Verified adapter facts: official SmolVLA rank-4 LoRA targeted `lm_expert` q/v plus state/action projections and had 185,664 trainable parameters, 0.0412% of 450,231,840. Rank-16 feasibility had 742,656 trainable. PatchGuard rank-4 had 9,984 trainable parameters. Custom 7D rank-4/rank-8 adapters had 128,007/131,975 trainable parameters. TG-7D rank-4 had 295,623 trainable. `R2R-OFT` used QLoRA/LoRA as implementation infrastructure for a prior extension but failed validation before closed-loop rollout. `BR-XVLA` freezes the official X-VLA PEFT intent at LoRA rank 8 / alpha 16 with primary `lambda=2` and uniform `lambda=0` arms, but no PEFT adapter was actually attached because the gradient smoke failed during import. Quantized OpenVLA-OFT INT4 and X-VLA are inference diagnostics, not QLoRA training results.

Failure attribution: many failures were not caused by LoRA capacity because they never reached a fair capacity test. Early failures were often action-interface and checkpoint persistence problems; later failures often came from data/headroom/objective-scale collapse. The campaign overemphasized trivial lightweight attachments relative to official prior anchoring. Future LoRA use should be conditional: only after Base/Prior/headroom are proven and only when the scientific mechanism is separable from PEFT.

## 12. Research-Funnel Statistics

| Funnel quantity | Count | Notes |
|---|---:|---|
| Total routes/diagnostic routes in ledger | 89 | previous 87 plus `BR-XVLA` and unselected `OCB-XVLA` |
| Formal selected Ours methods | 49 | previous 48 plus selected `BR-XVLA`; unselected `OCB-XVLA` counted as a candidate route only |
| Current task1 Ours candidates | 2 | `BR-XVLA` selected, `OCB-XVLA` not selected |
| Implemented routes | 79 | code/runner/local execution evidence; `BR-XVLA` has spec/data-adapter/gradient-smoke code but no successful backward |
| Trained/checkpointed routes | 32 | includes bounded `R2R-OFT`; prior diagnostics without training excluded |
| Formal Stage A count | 17 | unchanged |
| Formal Stage B count | 10 | unchanged |
| Route-level Stage-A-equivalent count | 19 | formal 17 plus PhaseBarrier/CensorCredit historical prototypes |
| Route-level Stage-B-equivalent count | 11 | formal 10 plus PhaseBarrier repaired prototype |
| Second-backbone Ours count | 0 | OpenVLA/X-VLA are priors, not Ours |
| Official-prior diagnostic routes | 11 | OpenVLA, OpenPI, PCD, LightVLA, RIPT, VLA-GSE, X-VLA, VLA-0, VLA-JEPA plus controls/scans |
| Paper-candidate GO count | 0 | no `PROTOTYPE_GO` |

Loss breakdown: 26 valid scientific kills; 41 non-scientific failures/blockers; 11 underpowered/unresolved; 11 infrastructure/diagnostic/no-claim rows. Formal selected proposal to Stage A: 17/49 = 34.7%. Formal selected proposal to Stage B: 10/49 = 20.4%. Stage B to GO: 0/10 = 0%. Formal selected proposal to second-backbone Ours: 0/49 = 0%.

## 13. Compute and Operational Audit

Current campaign-state records 5.21 GPU hours and 14.845 GiB downloaded for an earlier autonomous slice, but repo-wide GPU hours are `NOT_RECORDED`. The repository spans from first commit `07c823d` on 2026-06-27 10:30:15+09:00 to this audit on 2026-07-17 18:44:06+09:00, about 20.3 wall-clock days. Git history contains 835 commits across all refs and 831 ancestors of audit HEAD before this report commit.

Simulator episode lower bound from final artifacts remains at least 3,604 completed non-quarantined route-level episodes before adding all invalid attempts. Epoch 5 adds OpenVLA/SmolVLA residual episodes, short-requery episodes, LightVLA/CR/X-VLA/task1 scans, and task-level expert replay, but a globally normalized episode total is `NOT_RECORDED`. The BR-XVLA data-adapter and gradient-smoke attempts did not run simulator rollout.

Asset/storage notes: `C:\assets\data` and model/checkpoint caches dominate storage. Known large prior assets include OpenVLA-OFT around 15 GiB, X-VLA 3.28 GiB, VLA-0 21.46 GiB, VLA-JEPA 22.96 GiB, and SmolVLA under 1 GiB. Current untracked videos under `rollouts/2026_07_17/` include residual and short-requery OpenVLA videos.

Operational overhead includes context exhaustion, many documentation/report commits, durable worker launchers, invalid/repaired attempts, environment mismatch repairs, and branch proliferation. Duplicate/avoidable reruns include official LoRA drift/regeneration, PhaseBarrier invalid retrain then repair, COVI invalid v1 then repair, PCAV expansion resume, VDR self-worker confusion, RAP/KITE/SPARC launcher issues, wrong-env SmolVLA residual attempts, the X-VLA Base run first attempted in an incompatible OpenVLA environment, and two BR-XVLA gradient-smoke dependency failures before model load (`fastapi` missing, then `fastapi.__spec__ is None`).

## 14. Repetition and Search-Space Audit

Recurring families: candidate ranking/verifiers (`TCA-Select`, `PESA`, `PCAV`, `ECHO`); post-hoc residual correction (`FCAR`, `RAR`, `COVI`, `FAMR`, `SPARC`, `CFR`, `URF`, `BRID`, `CSPR`, `MCI`, `R2R-OFT`, `BR-XVLA`); action filtering/damping (`CSS`, `ExecSpec`, `PTC`, `SACF`, `RAC`, `EAC`, `AMP`, `PhaseBarrier`, `CR-LightVLA`); memory/retrieval (`RCV`, `CAVM`, `MTF`, `DAGR`, `RAP`); visual canonicalization/TTA (`PRISM`, `OCFN`, `SCVC`, `FANG`, `VDR`, `COVI`); temporal/history heads (`DICD`, `PSE`, `CALA`, `HEST`, `HASTE`, `TSC`, `MHS`, `NICE`); supervision/credit (`SafeTrace`, `CensorCredit`, `FEDO`, `G3P`, `ATCD`); representation/action generation (`TCA-Map`, `ActionMap`, `TG-7D`, `CBFD`, `EvoState`, `AFID`, `MCI`).

Common failed assumptions: small frozen-policy attachments would produce publishable gains; action L2/offline probes would predict closed-loop success; retrieval/memory would beat stateless/simple controls; visual canonicalization would fix brittleness without clean-behavior disruption; and proxy priors would satisfy reviewer-grade comparison. Epoch 5 improved this by running official priors first, but it has not yet produced Ours evidence.

## 15. Why the Campaign Is Not Finished

Ranked causes by impact:

1. Candidate quality and anchoring: 0/49 selected formal Ours methods have a completed official-prior Ours comparison. Evidence: ledger rows 27-88 and section 10.
2. No stable positive problem condition: many late routes found no usable headroom, collapsed labels, objective-scale failure, or only task-level rather than same-reset headroom. Evidence: PCAV/CFR/TSC/URF, NICE/CensorCredit, MCI, R2R-OFT, BR-XVLA.
3. Repeated narrow method families: residuals, gates, memories, and history heads were renamed more often than core assumptions changed. Evidence: section 14.
4. Pretrained-policy disruption/nonacting mechanisms: many Stage A/B methods lost to Base, ablation, or simple controls. Evidence: RCV, DAGR, MARC, MTF, EAC.
5. Late external-prior comparison: official priors became central only in Epoch 5. Evidence: OpenVLA/LightVLA/X-VLA sequence.
6. Low-compute strategy confusion: LoRA was useful infrastructure but often turned into a bias toward tiny local attachments; BR-XVLA is more prior-anchored but has not yet passed the import/gradient boundary. Evidence: section 11.
7. Underpowered early decisions: DICD, GCAP, CAVM, CALA, and RAR leave false-negative risk but no paper candidate. Evidence: section 8.
8. Documentation/state churn and context interruptions: state JSON lagged behind HEAD, and a previous Codex context exhausted. Evidence: snapshot caveat and this audit request.
9. Hardware/resource limits: RTX 5080 supports lightweight and INT4 diagnostics but not every large prior or full finetune. Evidence: LIFT, OpenPI/PCD/RIPT/VLA-GSE/VLA-0/VLA-JEPA blockers.

Scientific difficulty dominates, but process/governance mattered: broad search before stable official simulator evidence and late proxy-heavy comparisons generated many honest negatives without a reviewer-ready positive. The current BR-XVLA task1 condition is the most concrete live problem condition, but it is still only a pre-optimizer path with a same-reset-headroom caveat and a blocked gradient smoke.

## 16. False-Negative Audit

Potential false negatives exist, but none should be reopened immediately except the current BR-XVLA gate continuation. CAVM is strongest historically: 24/58 beat nearest 23/58, Base 22/58, and ablation 21/58, but no third expansion is allowed. CALA and RAR had small offline margins without closed-loop confidence. DICD and GCAP were underpowered Stage A archives, but later related methods tested richer variants.

Reviewer B overreach risk is real mainly for Stage 0 point-estimate or offline stops, not for completed Stage B kills. Later governance improved classification: MCI and R2R-OFT are implementation/validation failures, not scientific kills. BR-XVLA's current blocker is also not a scientific kill because no model load, PEFT attachment, forward, backward, optimizer, checkpoint, or rollout happened. Do not reopen a route merely because the campaign lacks a positive result.

## 17. Paper-Readiness Checklist

Nearest live route for checklist: `BR-XVLA`, because it is the selected prior-grounded Ours candidate for the task1 shared residual. It is an Ours mechanism on paper, but not an executed Ours result.

| Requirement | Status | Gap |
|---|---|---|
| Defensible novelty | PARTIAL | BR-XVLA is a narrow prior extension, but novelty has not survived implementation/evaluation |
| SmolVLA Base vs Base + Ours | MISSING | Base 3/8 exists; no Ours |
| Closest prior vs Ours | MISSING | X-VLA 6/8 exists; no Ours |
| Key ablation | MISSING | no executed BR-XVLA ablation |
| Relevant simple control | MISSING | no control for task1 residual |
| Clean retention | MISSING | not evaluated |
| Adequate paired statistics | MISSING | only matched Base/Prior diagnostic and task-level headroom so far |
| Quantized OpenVLA-OFT INT4 + Ours | MISSING | OpenVLA was previous prior diagnostic only |
| Second claim-specific condition | MISSING | not selected |
| Efficiency | MISSING | no trained/evaluated method |
| Reproducibility | PARTIAL | prior/Base/headroom/data-adapter artifacts exist, but gradient-smoke code is local uncommitted and blocked |
| Figure/table-ready artifacts | MISSING | no paper package |

Exact gap to `READY_TO_DRAFT_RAL_PAPER_PACKAGE`: no `PROTOTYPE_GO`, no successful BR-XVLA backward pass, no optimizer-step training, no offline validation pass, no closed-loop Ours result, no official-prior win, no positive Stage B, no second-backbone Ours result, no second condition, and no figure/table package.

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
- The BR-XVLA training spec was frozen and the X-VLA-format data-adapter smoke passed with a local `mmengine.fileio` shim.
- Two local BR-XVLA gradient-smoke attempts occurred after that; the latest durable result is `BR_XVLA_GRADIENT_SMOKE_BLOCKED_OR_FAIL` because `fastapi.__spec__ is None`, before model load/PEFT/forward/backward/optimizer/checkpoint.
- Local uncommitted files remain: `tca_map/xvla_task1/gradient_smoke.py` and `tests/test_br_xvla_gradient_smoke.py`; this audit records them but does not commit them.
- The first SmolVLA task1 run used an incompatible OpenVLA environment and failed before rollout; the official-env rerun is the valid Base result.
- Untracked rollout videos remain under `rollouts/2026_07_17/`; this audit did not move, stash, or delete them.

## 19. Recommended Strategic Decision

Recommendation: `CONTINUE_CURRENT_CYCLE`.

Justification: the prior `RESET_CANDIDATE_SELECTION_STRATEGY` recommendation has already been acted on by Epoch 5 official-prior-first. The current state is not a generic local-method cycle; it is a matched official-prior residual diagnosis that has already advanced through task-level headroom, task1 data audit, candidate selection, spec freeze, and data-adapter smoke. Continue only to the BR-XVLA no-optimizer gradient-smoke dependency boundary; do not start optimizer-step training, checkpointing, model selection, or closed-loop Ours rollout before that gate passes and is recorded.

## 20. Exact Resume Plan

`Exact Next Codex Prompt After User Review`

```text
Resume the autonomous VLA research campaign in C:\Users\jiheo\tca_map after reviewing reports/autonomous_research_full_history_audit.md.

Branch: codex/epoch5-official-prior-first
Last scientific HEAD before the Phase A audit report commit: 0b3697f697f8ab83f80f568ea85e8b4855709d52
Current scientific state: Epoch 5 official-prior-first, pushed stage epoch_5_br_xvla_data_adapter_smoke_complete, with local uncommitted BR-XVLA gradient-smoke attempt finished blocked
Current pushed decision: BR_XVLA_DATA_ADAPTER_SMOKE_PASS_GRADIENT_SMOKE_PENDING
Current local gradient-smoke decision: BR_XVLA_GRADIENT_SMOKE_BLOCKED_OR_FAIL because fastapi.__spec__ is None before model load/backward
Previous method: MCI-VLA
Previous decision: MCI_STAGE_0_IMPLEMENTATION_FAILURE
Selected audit recommendation: CONTINUE_CURRENT_CYCLE

Exact next scientific action:
Repair or accurately record the optional-dependency import shim boundary for the local BR-XVLA no-optimizer gradient smoke, then rerun only that one-batch gate. The gate must load cached X-VLA-Libero locally, attach official PEFT LoRA rank 8 / alpha 16, consume the X-VLA-format task1 adapter, compute the basket-remaining weighted supervised loss, call backward once, and prove finite nonzero trainable gradients. It must still not create an optimizer, call optimizer.step, write a checkpoint, or run closed-loop evaluation. If the dependency issue is not safely shim-only, stop and record the blocker.

Prohibited repeats:
Do not rescue or retune MCI-VLA, CSPR-VLA, R2R-OFT, CR-LightVLA, or ATCD. Do not generate a third task1 Ours candidate. Do not run optimizer-step training, checkpointing, model selection, or closed-loop Ours rollout before the BR-XVLA gradient smoke passes. Do not use the old task8 residual as an Ours target because X-VLA solved it 8/8. Do not claim INT4 OpenVLA-OFT is a full-precision reproduction. Do not switch branches, stash, reset, clean, or delete untracked rollout artifacts without user approval.

Time-to-evidence requirement:
Produce one durable gradient-smoke answer with artifact path, result JSON, stdout/stderr logs, heartbeat, exit code or process status, model-load/PEFT/forward/backward booleans, gradient finite/nonzero stats if reached, and explicit optimizer/checkpoint/training false booleans. Keep `reports/autonomous_compact_handoff.md` under 250 lines if updated.

LoRA role:
LoRA/QLoRA is only implementation infrastructure. For the next step, PEFT attachment and backward-smoke are authorized, but no optimizer-step LoRA training is authorized.

Reviewer false-negative safeguards:
Do not classify BR-XVLA dead from stale files, wrong-env failures, missing optional serving packages, or an import-only shim problem. Separate shared failure 20260727 from the X-VLA-only regression 20260725. Preserve invalid/repaired attempts separately.

Conditions for implementation and rollout:
Optimizer-step BR-XVLA training may start only after source, runner, focused tests, data-adapter smoke, successful no-optimizer gradient smoke, no privileged inference signal, prior fairness, resource risk, and frozen decision thresholds are documented. Closed-loop Ours rollout may start only after a training artifact passes offline validation, checkpoint reload, bounded action deltas, key ablation, simple control, clean retention, and exact paired manifest gates.
```
