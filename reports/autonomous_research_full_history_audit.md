# Autonomous VLA Research - Full History Audit

This report is the mandatory Phase A audit before any new candidate generation, implementation, training, or rollout. Evidence precedence follows the current Goal instruction, `reports/current_research_governance.md`, `AGENTS.md`, current result artifacts, current campaign state, git history, and historical reports. Missing facts are recorded as `NOT_RECORDED` rather than guessed.

## 1. Executive Summary

No paper-ready method exists. No valid `PROTOTYPE_GO` method exists. The repository contains substantial reusable infrastructure and a large body of negative, invalid, underpowered, and diagnostic evidence, but it does not contain `READY_TO_DRAFT_RAL_PAPER_PACKAGE`.

Route count: 73 distinct research routes were found, consisting of 26 historical or diagnostic routes plus 47 formal autonomous method proposals. Implemented route count: 70 of 73 have code-level or runner-level implementation evidence; `SafeLoRA-VLA`, `TG-VLA`, and `ISAC-VLA` did not reach an implemented local experiment. Trained route count: 31 have verified training or checkpoint artifacts; 23 of those are in the formal 47-method campaign. Closed-loop Stage A count: 17 formal autonomous methods reached Stage A; 19 route-level methods reached Stage-A-equivalent closed-loop evidence when historical prototypes are included. Stage B count: 10 formal autonomous methods reached Stage B; 11 route-level methods reached Stage-B-equivalent closed-loop evidence when the repaired PhaseBarrier prototype is included. Second-backbone Ours count: 0.

Outcome totals used in this audit: 26 valid scientific kills, 31 non-scientific failures, and 9 underpowered or unresolved results. Seven additional rows are infrastructure, diagnostic, no-claim, or preimplementation rejections and are not counted as scientific failures. The current active stage is `epoch_4_cycle_39_candidate_search_pending`; the previous method is `MCI-VLA`, closed as `MCI_STAGE_0_IMPLEMENTATION_FAILURE`.

The strongest result obtained is `CAVM-VLA`: after one allowed expansion, full reached 24/58 versus nearest-success memory 23/58, Base 22/58, and no-contrast 21/58. It is the best near-miss, but the effect is a one-episode advantage, no third expansion is allowed, and no second-backbone or external-prior confirmation exists.

The campaign is not paper-ready mainly because: no method beats Base, closest prior/proxy, key ablation, and simple reviewer-killer control in a valid Stage B; external-prior comparisons were usually local proxies, not official reproductions; many late methods died before rollout from data, headroom, objective-scale, or implementation failures; the search repeatedly favored lightweight frozen-SmolVLA attachments; and there is no same-method Quantized OpenVLA-OFT INT4 plus Ours result.

## 2. Audit Snapshot

| Field | Value |
|---|---|
| Snapshot timestamp | 2026-07-17T03:26:32.4080667+09:00 |
| Audit branch | `codex/full-history-audit-before-resume` |
| Scientific HEAD at audit start | `bc15132d74741c5b03c253c9bc062d8c5aaa5ddc` |
| HEAD commit subject | `Adjudicate MCI-VLA Stage 0 result` |
| Git status at audit start | clean, `## codex/full-history-audit-before-resume` |
| `main` HEAD | `8dc4de2fdbf576ace8bdf3699d190b761553c1fa` |
| Local/remote branch inventory | 256 local branches, 196 remote branches, 452 all-branch lines |
| Unmerged branch with unique work | `codex/execspec-repair-state0-state1`; one historical ExecSpec state1 kill-gate commit, superseded by later ExecSpec state2/3/3.5 evidence |
| Active Windows Python research worker | none detected |
| Active WSL Python research worker | none detected; only system `networkd-dispatcher` and unattended-upgrades Python services |
| CUDA compute snapshot | RTX 5080, 16,303 MiB total, 2,412 MiB used, 13,568 MiB free, 10 percent GPU utilization, 36 C; listed compute apps were desktop/system processes, not research Python |
| RAM snapshot | 24,288,100 KiB visible, 5,361,388 KiB free |
| Disk snapshot | C: 999,134,588,928 bytes total, 320,565,641,216 bytes free |
| Current epoch/cycle/stage | epoch 4, cycle 39, `epoch_4_cycle_39_candidate_search_pending` |
| Current/previous method | `MCI-VLA` |
| Current/previous decision | `MCI_STAGE_0_IMPLEMENTATION_FAILURE` |
| Current next action in state | Generate exactly three Cycle 39 candidates; do not rescue or retune MCI-VLA |
| Checkpoint path in state | `/mnt/c/assets/checkpoints/smolvla_libero` |
| Current MCI result paths | `reports/mci_vla/stage_0_result.json`, `reports/mci_vla/stage_0_partial.json`, `reports/mci_vla/stage_0_manifest.json`, `reports/mci_vla/stage_0_adjudication.md` |
| State-file commit caveat | `reports/autonomous_until_paper_state.json` records stale `current_commit` `c155008`; live HEAD/result artifacts are newer and authoritative |

## 3. Major Infrastructure Milestones

| Milestone | Commit | Artifact | Evidence | Current validity | Paper-method contribution |
|---|---|---|---|---|---|
| Official SmolVLA/LeRobot loader | `83e88a7` | `reports/official_smolvla_lerobot_model_load_status.md` | official checkpoint loaded locally | valid infrastructure | infrastructure only |
| Official LIBERO assets | `2a4cad2` | `reports/official_smolvla_libero_asset_verification.md` | dataset/checkpoint verification | valid infrastructure | infrastructure only |
| Official 8D state and 7D action semantics | `2efdd9e` | `reports/official_smolvla_libero_action_schema.md` | action/state schema audit | valid after later official run | infrastructure only |
| WSL/Linux CUDA rollout path | `54a80ff` | `tca_map/smolvla/official_wsl_libero_rollout.py` | 4/4 smoke plus 48/48 official pilot episodes | valid infrastructure | infrastructure only |
| Exact-init replay stabilization | `2b80cfb` | `reports/official_smolvla_exact_init_replay.md` | eligible exact expert successes reached 6/6, adapter 0/6 | valid infrastructure, method gap remained | infrastructure only |
| Deterministic evaluation protocol | `d12ef6d` | `reports/official_smolvla_eval_determinism_check.md` | fixed-seed exact max diff 0; unpinned regenerated artifacts differed | valid when fixed seed is used | infrastructure only |
| Persisted LoRA adapters | `15649d6` | `reports/official_smolvla_lora_checkpoint_manifest.md` | three seed bundles and checksums | valid but later canonical evaluation superseded drift | infrastructure only |
| Official closed-loop baseline/LoRA pilot | `54a80ff` | `reports/official_smolvla_libero_closed_loop_pilot.md` | Base 75 percent, LoRA seeds 83.3/66.7/75 over 12 each | valid infrastructure | no method claim |
| Quantized OpenVLA-OFT INT4 | `5c2a364` | `reports/openvla_oft_int4_hard_slice_result.md` | 20/20 hard-slice diagnostic where SmolVLA hard slice was 2/10 across two task groups | valid diagnostic | not Ours-on-second-backbone |
| Cross-backbone exact-state evaluation | `5c2a364` | `reports/openvla_oft_int4_hard_slice_result.md` | OpenVLA-OFT did not reproduce the same hard failures | valid diagnostic | blocks overclaiming SmolVLA-only failures |
| Partial checkpoint/resume infrastructure | `c4607b8` and later | `reports/pse_vla/stage_b_partial_result.json` plus many partials | 52 tracked partial JSON files, 23 exit-code files, 21 heartbeats, 29 PID files | valid operational infrastructure | infrastructure only |
| Durable WSL worker launchers | `e344233`, `9556599` | `scripts/*launch*`, `reports/*heartbeat*` | durable workers, heartbeats, missing-key-only resume behavior | valid for future long runs | infrastructure only |

Tracked-file inventory at audit time: 2,634 tracked files; 1,769 under `reports`, 274 under `scripts`, 237 under `tests`, 199 under `tca_map`, 103 under `runs`, 25 under `configs`, and 21 under `rollouts`. Machine-readable artifact inventory: 157 `*_result.json` files, 90 manifest JSON files, 19 state JSON files, 52 partial JSON files, 23 exit-code files, 21 heartbeat files, 29 PID files, and 97 checkpoint/checkpoint-related tracked paths.

## 4. Master Method Ledger

Legend: `Impl`, `Trained`, `GPU`, `Sim`, `S0`, `SA`, `SB`, and `2BB` are Yes/No/Partial/NA evidence columns. `Closure` means permanent scientific closure, not merely operational stop. `Reopen` is bounded reopen eligibility under current governance, not a recommendation.

| # | Epoch/cycle | Method/route | Core idea | Contribution class | Closest external prior | Branch/commit | Impl | Trained | GPU | Sim | S0 | SA | SB | 2BB | Principal metric/result | Final decision | Evidence classification | Closure | Reopen | Main artifact paths |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | historical | TCA-Map | target-conditioned action mapping | representation/action generation | TCA-style target conditioning | historical current history | Yes | Yes | Yes | No | NA | No | No | No | offline target-prior positive but 7D head lost to mean baseline | archive | VALID_HISTORICAL | Yes | No | `reports/final_project_state.md` |
| 2 | historical | TCA-Select | select target-conditioned candidate | candidate ranking | TCA target selection | historical current history | Yes | No | No | No | NA | No | No | No | no meaningful gain; representation collapse risk | archive | INCONCLUSIVE | No | No | `reports/killed_routes_summary.md` |
| 3 | historical | ActionMap approximation / mini-anchor | approximate action mapping from anchors | representation/action generation | action-map residual methods | historical current history | Yes | No | No | No | NA | No | No | No | oracle 0.06565, method 0.52993 vs mean 0.46677 and MLP 0.50193; top1 0.0185 | killed | VALID_HISTORICAL | Yes | No | `reports/actionmap_mini_anchor_state1_result.json` |
| 4 | historical | CSS-Shield | semantic safety shield | action filtering/damping | safety shielding | historical current history | Yes | No | No | No | NA | No | No | No | native Phase2 no wrong-target gain over safety-only; intervention rate 1.0 | killed | VALID_HISTORICAL | Yes | No | `reports/css_shield_autopilot_state.json` |
| 5 | historical | ExecSpec-Repair | executable-spec action repair | action filtering/damping | spec-guided repair | historical plus unmerged state1 branch | Yes | No | No | No | NA | No | No | No | full 17/19 degraded exact-init replays; diagonal affine also 17/19 | killed | VALID_HISTORICAL | Yes | No | `reports/execspec_state3_5_baseline_dominance_audit.json` |
| 6 | historical | AMP-GD | gradient-directed action-map planning | action optimization | action planning and gradient control | historical current history | Yes | No | No | Partial | NA | No | No | No | toy success 1.0 but LIBERO tiny did not beat safety/random | archive | INCONCLUSIVE | No | No | `reports/amp_gd_state2_report.json` |
| 7 | historical | ResetSpec-Retarget | retarget resets/object frames | retargeting | object-relative retargeting | historical current history | Yes | No | No | No | NA | No | No | No | global scale reward/success 1, object-relative 0 | archive | INCONCLUSIVE | No | No | `reports/resetspec_state1_result.json` |
| 8 | historical | Phase-locked/event retiming | retime actions by event phase | temporal/history | phase/event retiming | historical current history | Yes | No | No | No | NA | No | No | No | 0/9 recovery over raw/best simple | killed | VALID_HISTORICAL | Yes | No | `reports/phase_locked_retiming_state1_result.json` |
| 9 | historical | TL-ChunkRepair | temporal chunk-level repair | temporal/action repair | temporal localization repair | historical current history | Yes | No | No | No | NA | No | No | No | violations reduced 8/8 but safe success 0/8; no-repair best success 1/1 | killed | VALID_HISTORICAL | Yes | No | `reports/tl_chunkrepair_state1_result.json` |
| 10 | historical | ContactTube-Aug | contact tube augmentation | visual/contact representation | contact-tube imitation | historical current history | Yes | No | No | No | NA | No | No | No | validity 0.849265, clip 0.150735, simple retarget error 0.009154 vs full 0.015226; missing HDF5 pose | data/measurement failure | INVALID_QUARANTINED | No | No | `reports/contacttube_aug_state1_result.json` |
| 11 | historical | PRISM-VLA | visual canonicalization or sensitivity | visual canonicalization/TTA | PRISM-like image transforms | historical current history | Yes | No | No | No | NA | No | No | No | canonicalization 0.474066 beat best PRISM 0.436356; sensitivity weakened | killed | VALID_HISTORICAL | Yes | No | `reports/all_killed_routes_summary.md` |
| 12 | historical | ContactSet-VLA | contact set geometry | contact representation | contact-set policies | historical current history | Yes | No | No | No | NA | No | No | No | full L2 1.1050 worse than single 0.9305, destination 0.86372, no-geom 0.85145 on 6 demos | archive | INCONCLUSIVE | No | No | `reports/all_killed_routes_summary.md` |
| 13 | historical | SafeTrace-VLA | safety preference tracing | supervision/credit | DPO/safety preference VLA | historical current history | Yes | No | No | No | NA | No | No | No | 800 pairs but only 10 nontrivial; generic safety accuracy 1.0; no utility labels | data failure | INVALID_QUARANTINED | No | No | `reports/safetrace_vla_state1_result.json` |
| 14 | historical | SafeLoRA-VLA | safety LoRA adaptation | low-compute adaptation | SafeLoRA | historical current history | No | No | No | No | NA | No | No | No | no experiment or training; source/dependency route blocked | preimplementation rejection | UNKNOWN | No | No | `reports/all_killed_routes_summary.md` |
| 15 | historical | PatchGuard-VLA | rank-4 guarded patch adapter | low-compute adapter | adversarial or patch-guard LoRA | historical current history | Yes | Yes | Yes | No | NA | No | No | No | metric 0.13356 vs generic adv LoRA 0.142803 and cutout 0.02973 | killed | VALID_HISTORICAL | Yes | No | `reports/patchguard_vla_state1_result.json` |
| 16 | historical | Standard SmolVLA LoRA baseline route | official PEFT baseline | infrastructure/control | LoRA/PEFT | `0a15424`, `54a80ff` | Yes | Yes | Yes | Yes | NA | Partial | No | No | rank4 185,664 trainable; closed-loop pilot Base 75 percent, LoRA seeds 83.3/66.7/75 | infrastructure baseline | SUPERSEDED | No | No | `reports/official_smolvla_libero_baseline_scaleup_result.json` |
| 17 | historical | Custom SmolVLA 7D adapter | custom 7D action adapter | low-compute adapter | PEFT adapter | historical current history | Yes | Yes | Yes | Yes | NA | Partial | No | No | offline rank8 0.494959 beat mean 1.08245 but exact replay adapter 0/6 | implementation/control gap | INVALID_QUARANTINED | No | No | `reports/tg7d_adapter_state_gate.json` |
| 18 | historical | TG-VLA | trajectory generation VLA | representation/action generation | trajectory-generator VLA prior | historical current history | No | No | No | No | NA | No | No | No | no training; baseline/prior risk killed before experiment | preimplementation rejection | UNKNOWN | No | No | `reports/killed_routes_summary.md` |
| 19 | historical | TG-7D Adapter | trajectory-generation 7D adapter | representation/action generation | TG adapter | historical current history | Yes | Yes | Yes | No | NA | No | No | No | heldout L2 0.740922 vs canonical 0.587661, standard LoRA 0.600887, MLP 0.619985 | killed | VALID_HISTORICAL | Yes | No | `reports/tg7d_adapter_state_gate.json` |
| 20 | historical | Post-canonical residual mining | residual after canonicalization | residual correction | canonical residual mining | historical current history | Yes | No | No | No | NA | No | No | No | clean-paraphrase delta -0.000748, oracle headroom -0.137013 | no-headroom diagnostic | VALID_HISTORICAL | No | No | `reports/final_autonomous_method_decision.md` |
| 21 | historical | FCAR | feature-conditioned adaptive residual | residual correction | residual adapter policies | historical current history | Yes | Yes | Yes | No | NA | No | No | No | full 0.100145, static merge 0.09118, rank4 LoRA 0.07619, Base 0.123998 | killed | VALID_HISTORICAL | Yes | No | `reports/final_autonomous_method_decision.md` |
| 22 | historical | ECHO | effect-conditioned headroom search | candidate ranking/verifier | effect-conditioned repair | `8dc4de2` on main-era history | Yes | No | No | No | NA | No | No | No | 12 groups, 96 candidates; Base/oracle/random all 0.8333; recoverable failures 0 | no-headroom diagnostic | VALID_HISTORICAL | No | No | `reports/implementation_v2_final_decision.md` |
| 23 | historical | Quantized OpenVLA-OFT INT4 diagnostic | second-backbone hard-slice check | infrastructure/diagnostic | OpenVLA-OFT | `5c2a364` | Yes | No | Yes | Yes | NA | No | No | Yes | OpenVLA hard slice 20/20 while matched SmolVLA hard slice 2/10 | diagnostic only | VALID_CANONICAL | No | No | `reports/openvla_oft_quantized_hard_slice_result.md` |
| 24 | historical | PhaseBarrier-VLA | phase barrier intervention | action filtering/damping | phase barrier control | historical current history | Yes | Yes | Yes | Yes | NA | Yes | Yes | No | valid bounded repair full 0/20 vs Base 8/20, ablation 9/20, global damping 0 | killed | VALID_CANONICAL | Yes | No | `reports/phase_barrier_bounded_repair_result.json` |
| 25 | historical | CensorCredit-VLA | censored credit supervision | supervision/credit | censored credit assignment | historical current history | Yes | Yes | Yes | Yes | NA | Yes | No | No | initial full 0.5 vs baseline 0 but uncensored ablation 0.5; later 24/24 identical labels | data/label failure | INVALID_QUARANTINED | No | No | `reports/censor_credit_empirical_postmortem.md` |
| 26 | historical | ISAC-VLA | intervention state/action correction | supervision/action generation | SDP, TORL, ConRFT-like intervention learning | historical current history | No | No | No | No | NA | No | No | No | killed before implementation as near-exact prior/resource mismatch | preimplementation rejection | UNKNOWN | No | No | `reports/all_killed_routes_summary.md` |
| 27 | initial | DICD-VLA | demonstration-indexed context decoder | temporal/history | context-decoder prior | `a2154c2` | Yes | Yes | Yes | Yes | Partial | Yes | No | No | full 1/10 vs chunk-index 2/10 and no-history 1/10 | underpowered archive | INCONCLUSIVE | No | No | `reports/dicd_vla/stage_a_result.json` |
| 28 | initial | FEDO-VLA | failure-evidence dropout objective | supervision/credit | APEX-like proxy | `b2f7b50` | Yes | Yes | Yes | Yes | Partial | Yes | No | No | faulted full 1 vs strongest 2; clean frozen 4 vs full 0 | killed | VALID_CANONICAL | Yes | No | `reports/fedo_vla/stage_a_result.json` |
| 29 | initial | GCAP-VLA | goal-conditioned action prior | representation/action generation | goal-conditioned action policies | `e24a6a1` | Yes | Yes | Yes | Yes | Partial | Yes | No | No | occluded Base 4, Sobel 5, no-temporal 4, full 3; clean Base 1, full 5 | underpowered target-axis archive | INCONCLUSIVE | No | No | `reports/gcap_vla/stage_a_result.json` |
| 30 | epoch2/c1 | PTC-VLA | phase-target correction | temporal/action repair | phase target control | `ce7d455` | Yes | Yes | Yes | Yes | Partial | Yes | No | No | full 0/10 vs Base 3/10 with active mechanism | killed | VALID_CANONICAL | Yes | No | `reports/ptc_vla/stage_a_result.json` |
| 31 | epoch2/c2 | SACF-VLA | semantics-aware correction filter | action filtering/damping | CAG/null correction | `fc0fb1e` | Yes | Yes | Yes | Yes | Partial | Yes | No | No | full 0/10 vs Base 7/10 | killed | VALID_CANONICAL | Yes | No | `reports/sacf_vla/stage_a_result.json` |
| 32 | epoch2/c3 | OCFN-VLA | occlusion-conditioned feature normalization | visual canonicalization/TTA | occlusion feature normalization | `c183d15` | Yes | Yes | Yes | Yes | Partial | Yes | Yes | No | expanded full 26/80 vs zero-noise 27/80 | killed | VALID_CANONICAL | Yes | No | `reports/ocfn_vla/stage_b_result.json` |
| 33 | epoch3/c1 | CBFD-VLA | cross-backbone failure distillation | representation/action generation | stronger-backbone teacher | `c0fbca2` | Yes | Yes | Yes | Yes | Partial | Yes | No | No | full 0/10 vs Base 7/10 | killed | VALID_CANONICAL | Yes | No | `reports/cbfd_vla/stage_a_result.json` |
| 34 | epoch3/c2 | SCVC-VLA | scene canonicalization via visual consistency | visual canonicalization/TTA | visual consistency canonicalization | `2c733a3` | Yes | Yes | Yes | Yes | Partial | Yes | Yes | No | full 11/40 vs shifted Base 20/40, paired CI [-0.425,-0.025] | killed | VALID_CANONICAL | Yes | No | `reports/scvc_vla/stage_b_result.json` |
| 35 | epoch3/c3 | PSE-VLA | policy state estimator | temporal/history | state estimator prior | `c4607b8` | Yes | Yes | Yes | Yes | Partial | Yes | Yes | No | full 50/80 vs bright-single 51/80; useful upper CI excluded | killed | VALID_CANONICAL | Yes | No | `reports/pse_vla/stage_b_result.json` |
| 36 | epoch4/c1 | RCV-VLA | retrieval-conditioned voting | memory/retrieval | SV-deviation proxy | `3a8a815` | Yes | Yes | Yes | Yes | Partial | Yes | Yes | No | full 20/40, Base 14/40, proxy 16/40, no-context/stateless 24/40 | killed | VALID_CANONICAL | Yes | No | `reports/rcv_vla/stage_2b_result.json` |
| 37 | epoch4/c2 | CAVM-VLA | contrastive action-value memory | memory/retrieval | success-memory prior | `e69f64f` | Yes | Yes | Yes | Yes | Partial | Yes | Yes | No | expanded full 24/58 vs nearest-success 23/58, Base 22/58, ablation 21/58 | unresolved near-miss | INCONCLUSIVE | No | No | `reports/cavm_vla/stage_2b_expansion_result.json` |
| 38 | epoch4/c3 | FANG-VLA | failure-aware navigation gate | action filtering/damping | AFIL proxy | `de08f34` | Yes | Yes | Yes | Yes | Partial | Yes | Yes | No | full 11/40, Base 16/40, AFIL proxy 15/40, ablation 11/40 | killed | VALID_CANONICAL | Yes | No | `reports/fang_vla/stage_b_result.json` |
| 39 | epoch4/c4 | EvoState-VLA | evolutionary state feature probe | representation/action generation | evolutionary state prior | `a2e94c1` | Yes | No | No | No | Yes | No | No | No | 4,221 validation pairs; improvement over actionless 0.024689 below 0.05 gate | valid scoped design stop | VALID_CANONICAL | Yes | No | `reports/evostate_vla/development_audit.json` |
| 40 | epoch4/c5 | RAC-VLA | reflective action correction | action filtering/damping | Reflective proxy | `adf3d07` | Yes | Yes | Yes | Yes | Partial | Yes | Yes | No | Stage B full 1/40, Base 1/40, proxy 1/40, ablation 2/40, inverse-gain 2/40 | killed | VALID_CANONICAL | Yes | No | `reports/rac_vla/stage_b_result.json` |
| 41 | epoch4/c6 | MTF-VLA | memory-temporal filtering | memory/retrieval | FrameSkip prior | `0824ba8` | Yes | Yes | Yes | Yes | Partial | Yes | Yes | No | full 26/40 vs no-retention 32/40, delta -0.15 CI [-0.275,-0.025] | killed | VALID_CANONICAL | Yes | No | `reports/mtf_vla/stage_b_result.json` |
| 42 | epoch4/c7 | DAGR-VLA | dynamic action guidance from retrieval | memory/retrieval | DAM proxy | `1853080` | Yes | Yes | Yes | Yes | Partial | Yes | Yes | No | Stage B full 18/40, Base 28/40, heuristic 24/40, prior proxy 5/40, ablation 16/40 | killed | VALID_CANONICAL | Yes | No | `reports/dagr_vla/stage_b_result.json` |
| 43 | epoch4/c8 | MARC-VLA | mixture action residual correction | residual correction | OpenVLA L1 proxy | `0d5648d` | Yes | Yes | Yes | Yes | Partial | Yes | No | No | full 0/10, Base 8/10, no-gate 7/10, static L1 mixture 7/10 | killed | VALID_CANONICAL | Yes | No | `reports/marc_vla/stage_a_result.json` |
| 44 | epoch4/c9 | PESA-VLA | predictive episode-success assessor | candidate ranking/verifier | success assessor prior | `f6b65e6` | Yes | No | No | No | Yes | No | No | No | query probe accuracy 0.5225 vs majority 0.6 over 400 validation records | design/data failure | INVALID_QUARANTINED | No | No | `reports/pesa_vla/development_audit.json` |
| 45 | epoch4/c10 | EAC-VLA | evidence-adaptive controller | action filtering/damping | AAC proxy | `5e7bbdd` | Yes | Yes | Yes | Yes | Partial | Yes | Yes | No | Stage B full 29/40 vs Base/AAC/ablation 30/40, fixed-short 29/40 | killed | VALID_CANONICAL | Yes | No | `reports/eac_vla/stage_b_result.json` |
| 46 | epoch4/c11 | G3P-VLA | geometric goal-grounded policy probe | representation/action generation | geometric policy prior | `9079a25` | Yes | No | No | No | Yes | No | No | No | data/supervision failure before rollout | data failure | INVALID_QUARANTINED | No | No | `reports/g3p_vla/development_audit.json` |
| 47 | epoch4/c12 | CALA-VLA | context-aware latent alignment | temporal/history | latent alignment prior | `65fd947` | Yes | No | No | No | Yes | No | No | No | full probe RMSE 3.1988 vs action-history-only 3.14397; margin -0.011718 | unresolved false-negative risk | INCONCLUSIVE | No | Maybe | `reports/cala_vla/development_audit.json` |
| 48 | epoch4/c13 | RAR-VLA | residual action representation | residual correction | residual representation prior | `fbffc0a` | Yes | No | No | No | Yes | No | No | No | full RMSE 0.171954 vs zero-residual 0.165597; margin -0.038376 | unresolved false-negative risk | INCONCLUSIVE | No | Maybe | `reports/rar_vla/development_audit.json` |
| 49 | epoch4/c14 | COVI-VLA | contrastive visual intervention | visual canonicalization/TTA | contrastive visual intervention | `6027d10` | Yes | Yes | Yes | No | Yes | No | No | No | repaired final output valid 0.2, only two nonzero objectives; no scientific kill | implementation failure | INVALID_QUARANTINED | No | No | `reports/covi_vla/stage_0_result.json` |
| 50 | epoch4/c15 | LIFT-VLA | latent intervention fine-tuning | low-compute adaptation | LIFT-like fine-tuning | `b1adc67` | Yes | No | No | No | Yes | No | No | No | compute infeasible, no training/rollout | resource failure | INVALID_QUARANTINED | No | No | `reports/lift_vla/stage_0_result.json` |
| 51 | epoch4/c16 | IARC-VLA | inverse action-region correction | residual correction | inverse action correction | `aee7867` | Yes | Yes | Yes | No | Yes | No | No | No | training happened but action validity/data range invalid 0.3; no rollout | implementation/data failure | INVALID_QUARANTINED | No | No | `reports/iarc_vla/stage_0a_result.json` |
| 52 | epoch4/c17 | FAMR-VLA | feature-aligned model residual | residual correction | model-residual prior | `5725a68` | Yes | Yes | Yes | No | Yes | No | No | No | endpoint training happened; implementation/data failure and no headroom | implementation/data failure | INVALID_QUARANTINED | No | No | `reports/famr_vla/stage_0a_result.json` |
| 53 | epoch4/c18 | PCAV-VLA | policy-conditioned action verifier | candidate ranking/verifier | action verifier prior | `787fc7c` | Yes | No | No | No | Yes | No | No | No | no usable headroom; alternative fraction 0.9375 but headroom false | no usable headroom | INVALID_QUARANTINED | No | No | `reports/pcav_vla/stage_0a_result.json` |
| 54 | epoch4/c19 | SPARC-VLA | sparse action representation correction | residual correction | sparse action correction | `660db5a` | Yes | No | No | No | Yes | No | No | No | implementation/action-validity failure | implementation failure | INVALID_QUARANTINED | No | No | `reports/sparc_vla/stage_0a_result.json` |
| 55 | epoch4/c20 | NICE-VLA | noisy-invariant context encoder | temporal/history | invariant context encoder | `a814a7a` | Yes | Yes | Yes | No | Yes | No | No | No | Stage0a pass, Stage0b1 data failure; 1,792 pairs, collapsed action-regime contrast | data failure | INVALID_QUARANTINED | No | No | `reports/nice_vla/stage_0b1_result.json` |
| 56 | epoch4/c21 | HEST-VLA | history-enhanced state transformer | temporal/history | history transformer prior | `5124db4` | Yes | No | No | No | Yes | No | No | No | implementation/optimization failure before rollout | implementation failure | INVALID_QUARANTINED | No | No | `reports/hest_vla/stage_0a_result.json` |
| 57 | epoch4/c22 | HASTE-VLA | history-aware action state estimator | temporal/history | history estimator prior | `4c0a015` | Yes | No | No | No | Yes | No | No | No | implementation/optimization/support failure | implementation failure | INVALID_QUARANTINED | No | No | `reports/haste_vla/stage_0a_result.json` |
| 58 | epoch4/c23 | KITE-VLA | keyframe intervention transform | temporal/action repair | keyframe intervention prior | `500dfb1` | Yes | No | No | No | Yes | No | No | No | implementation/support failure before rollout | implementation failure | INVALID_QUARANTINED | No | No | `reports/kite_vla/stage_0a_result.json` |
| 59 | epoch4/c24 | VDR-VLA | visual discrepancy repair | visual canonicalization/TTA | visual discrepancy repair | `52dc9ae` | Yes | No | No | No | Yes | No | No | No | action-validity/pre-manifest failure | implementation failure | INVALID_QUARANTINED | No | No | `reports/vdr_vla/stage_0a_result.json` |
| 60 | epoch4/c25 | RAP-VLA | retrieval-augmented policy | memory/retrieval | retrieval-augmented control | `cc20e20` | Yes | No | No | No | Yes | No | No | No | launcher/preflight failure before rollout | implementation failure | INVALID_QUARANTINED | No | No | `reports/rap_vla/stage_0_result.json` |
| 61 | epoch4/c26 | AMP-VLA | action-manifold projection | action filtering/damping | action manifold projection | `ef1f20a` | Yes | No | No | No | Yes | No | No | No | implementation/optimization failure | implementation failure | INVALID_QUARANTINED | No | No | `reports/amp_vla/stage_0_result.json` |
| 62 | epoch4/c27 | CFR-VLA | counterfactual failure repair | residual correction | counterfactual repair | `bdec3dc` | Yes | No | No | No | Yes | No | No | No | no usable/residual headroom | no usable headroom | INVALID_QUARANTINED | No | No | `reports/cfr_vla/stage_0_result.json` |
| 63 | epoch4/c28 | TSC-VLA | temporal state correction | temporal/history | temporal correction prior | `e4a5abc` | Yes | No | No | No | Yes | No | No | No | no usable/residual headroom | no usable headroom | INVALID_QUARANTINED | No | No | `reports/tsc_vla/stage_0_result.json` |
| 64 | epoch4/c29 | CCIF-VLA | cross-context intervention filtering | action filtering/damping | context intervention prior | `77a470e` | Yes | No | No | No | Yes | No | No | No | design failure; no scientific result | design failure | INVALID_QUARANTINED | No | No | `reports/ccif_vla/stage_0_result.json` |
| 65 | epoch4/c30 | URF-VLA | uncertainty residual filtering | residual correction | uncertainty residual prior | `466511f` | Yes | No | No | No | Yes | No | No | No | no usable/residual headroom | no usable headroom | INVALID_QUARANTINED | No | No | `reports/urf_vla/stage_0_result.json` |
| 66 | epoch4/c31 | S2C-VLA | state-to-contact conversion | contact representation | contact conversion prior | `73dc3c0` | Yes | No | No | No | Yes | No | No | No | data/supervision/cache coverage failure | data failure | INVALID_QUARANTINED | No | No | `reports/s2c_vla/stage_0_result.json` |
| 67 | epoch4/c32 | LCG-VLA | latent contact guidance | contact representation | latent contact prior | `9d48f36` | Yes | No | No | No | Yes | No | No | No | design/no-headroom failure | design failure | INVALID_QUARANTINED | No | No | `reports/lcg_vla/stage_0_result.json` |
| 68 | epoch4/c33 | AFID-VLA | action-frequency invariant descriptor | representation/action generation | invariant descriptor prior | `f392a33` | Yes | No | No | No | Yes | No | No | No | implementation/objective-scale failure | implementation failure | INVALID_QUARANTINED | No | No | `reports/afid_vla/stage_0_result.md` |
| 69 | epoch4/c34 | BRID-VLA | behavior residual identity decomposition | residual correction | residual decomposition prior | `7348dd9` | Yes | No | No | No | Yes | No | No | No | no usable scientific headroom or design failure | design/no-headroom failure | INVALID_QUARANTINED | No | No | `reports/brid_vla/stage_0_result.json` |
| 70 | epoch4/c35 | MHS-VLA | multi-horizon state supervision | temporal/history | multi-horizon prior | `48c8239` | Yes | No | No | No | Yes | No | No | No | data/supervision/cache coverage failure | data failure | INVALID_QUARANTINED | No | No | `reports/mhs_vla/stage_0_result.json` |
| 71 | epoch4/c36 | DCCG-VLA | dense cross-context guidance | visual/action guidance | dense context guidance prior | `bdeed30` | Yes | No | No | No | Yes | No | No | No | data/supervision/cache coverage failure | data failure | INVALID_QUARANTINED | No | No | `reports/dccg_vla/stage_0_result.json` |
| 72 | epoch4/c37 | CSPR-VLA | consistency-scaled policy residual | residual correction | consistency residual prior | `dcfcfa6` | Yes | No | No | No | Yes | No | No | No | raw design failure corrected to implementation failure from gradient-scale issue | implementation failure | INVALID_QUARANTINED | No | No | `reports/cspr_vla/stage_0_result.json` |
| 73 | epoch4/c38 | MCI-VLA | multi-consistency invariance | representation/action generation | ROVLA multi-consistency proxy | `bc15132` | Yes | No | No | No | Yes | No | No | No | 17,280/17,280 rows, no exceptions, weighted gradient norm ratio 324.58 greater than 100 limit | implementation failure | INVALID_QUARANTINED | No | No | `reports/mci_vla/stage_0_result.json` |

## 5. Detailed Chronological Timeline

| Date KST | Commit | Action | Method | Result | Scientific meaning |
|---|---|---|---|---|---|
| 2026-06-27 | `07c823d` | repository initialized with agent instructions | workspace | autonomous research workspace begins | governance and evidence discipline become part of repo |
| 2026-07-04 to 2026-07-08 | historical commits | TCA, CSS, ExecSpec, AMP-GD, ResetSpec, retiming, TL, ContactTube, PRISM, ContactSet, ActionMap | historical routes | several offline and diagnostic kills | early routes showed action/interface and data gaps before official SmolVLA rollout stabilized |
| 2026-07-09 | `83e88a7` | official model loader | SmolVLA | official LeRobot model loads | infrastructure milestone |
| 2026-07-09 | `2a4cad2` | official LIBERO assets verified | SmolVLA/LIBERO | assets usable | infrastructure milestone |
| 2026-07-09 | `2efdd9e` | 8D state and 7D action schema documented | SmolVLA | action semantics known | corrected earlier interface uncertainty |
| 2026-07-09 | `0a15424` | official LoRA baseline scale-up | SmolVLA LoRA | rank4 185,664 trainable, loss improved | infrastructure, not paper method |
| 2026-07-10 | `d12ef6d` | deterministic evaluation checked | SmolVLA | fixed seed deterministic | reproducibility support |
| 2026-07-10 | `15649d6` | LoRA checkpoint persistence | SmolVLA LoRA | three seed bundles persisted | checkpointing support |
| 2026-07-11 | `54a80ff` | WSL closed-loop pilot | SmolVLA/LoRA | Base and LoRA pilot success measured over 48 episodes | official simulator path validated |
| 2026-07-11 | `5c2a364` | OpenVLA-OFT INT4 diagnostic | second backbone | hard slice 20/20 | diagnostic showed SmolVLA failure condition did not transfer directly |
| 2026-07-12 | `a2154c2` | DICD Stage A adjudicated | DICD | 1/10 vs 2/10 | underpowered archive |
| 2026-07-12 | `b2f7b50` | FEDO Stage A adjudicated | FEDO | clean retention collapse | valid kill |
| 2026-07-12 | `e24a6a1` | GCAP Stage A adjudicated | GCAP | target-axis mixed, clean positive | underpowered archive |
| 2026-07-12 | `ce7d455` | PTC Stage A adjudicated | PTC | 0/10 vs Base 3/10 | valid kill |
| 2026-07-13 | `fc0fb1e` | SACF Stage A adjudicated | SACF | 0/10 vs Base 7/10 | valid kill |
| 2026-07-13 | `c183d15` | OCFN Stage B expanded | OCFN | 26/80 vs 27/80 | valid kill despite scale |
| 2026-07-13 | `c0fbca2` | CBFD Stage A adjudicated | CBFD | 0/10 vs Base 7/10 | valid kill |
| 2026-07-13 | `2c733a3` | SCVC Stage B adjudicated | SCVC | 11/40 vs 20/40 | valid kill |
| 2026-07-13 | `c4607b8` | PSE Stage B adjudicated | PSE | 50/80 vs 51/80, upper CI excludes useful gain | valid kill |
| 2026-07-14 | `3a8a815` | RCV Stage B adjudicated | RCV | ablations/stateless 24/40 beat full 20/40 | valid kill |
| 2026-07-14 | `e69f64f` | CAVM expanded Stage B fixed | CAVM | full 24/58 best by one episode | strongest unresolved near-miss |
| 2026-07-14 | `de08f34` | FANG Stage B adjudicated | FANG | full 11/40 vs Base 16/40 | valid kill |
| 2026-07-14 | `a2e94c1` | EvoState design stop | EvoState | 4,221 pairs, margin 0.024689 below gate | valid scoped pre-rollout kill |
| 2026-07-14 | `adf3d07` | RAC Stage B adjudicated | RAC | no improvement, inverse/simple controls tie or beat | valid kill |
| 2026-07-14 | `0824ba8` | MTF Stage B adjudicated | MTF | no-retention ablation 32/40 beats full 26/40 | valid kill |
| 2026-07-15 | `1853080` | DAGR Stage B adjudicated | DAGR | Base 28/40 beats full 18/40 | valid kill |
| 2026-07-15 | `0d5648d` | MARC Stage A adjudicated | MARC | full 0/10 vs Base 8/10 | valid catastrophic kill |
| 2026-07-15 | `f6b65e6` | PESA Stage 0 closed | PESA | query probe below majority | design/data failure |
| 2026-07-15 | `5e7bbdd` | EAC Stage B adjudicated | EAC | full 29/40 vs 30/40 controls | valid kill after promising Stage A |
| 2026-07-15 to 2026-07-16 | many | G3P through DCCG | stage0-heavy methods | data, headroom, implementation failures | process pivoted to pre-rollout audits after repeated rollout losses |
| 2026-07-17 | `dcfcfa6` | CSPR Stage 0 adjudicated | CSPR | gradient-scale implementation failure | no scientific result |
| 2026-07-17 | `bc15132` | MCI Stage 0 adjudicated | MCI | objective-scale violation at 324.58 vs limit 100 | current campaign paused at cycle 39 candidate search |

## 6. Valid Scientific Kills

The 26 valid scientific kills are valid only at their scoped claims; none proves that an entire broad family is impossible.

| Method | Exact metric | Strongest baseline/control | Ablation evidence | Sample size | Confidence/paired evidence | Why closure is justified |
|---|---|---|---|---|---|---|
| TCA-Map | target-prior positive, 7D head lost to mean baseline | mean/MLP style baseline | action head failed | offline historical | NOT_RECORDED | representation did not survive action head/control path |
| ActionMap | oracle 0.06565, method 0.52993, mean 0.46677, MLP 0.50193 | MLP and mean | candidate top1 0.0185 | 1008 train, 432 eval | offline split | learned approximation worse than simple controls |
| CSS-Shield | no wrong-target gain; intervention rate 1.0 | safety-only | full shield acts everywhere | historical diagnostic | NOT_RECORDED | mechanism collapsed to over-intervention |
| ExecSpec-Repair | full 17/19 degraded; diagonal affine 17/19 | calibrated diagonal affine | full no better | 19 exact-init replays | paired replay | spec repair did not improve exact-init failures |
| Phase retiming | 0/9 recovery | raw/best simple | no recovery | 9 recovery cases | small but direct | retiming did not recover any case |
| TL-ChunkRepair | violations down 8/8, safe success 0/8 | no repair 1/1 | repair harms success | 8 repaired cases | direct paired diagnostic | reduced violation metric did not translate to success |
| PRISM | canonical 0.474066 vs best PRISM 0.436356 | canonicalization | sensitivity weakened | historical offline | NOT_RECORDED | PRISM-like transform did not beat simpler canonical route |
| PatchGuard | 0.13356 vs generic adv LoRA 0.142803 and cutout 0.02973 | generic adv LoRA | cutout strong | historical offline | NOT_RECORDED | guarded patch did not beat direct LoRA/control |
| TG-7D | heldout L2 0.740922 vs canonical 0.587661 and LoRA 0.600887 | canonical and standard LoRA | residual route worse | heldout plus 30 CF and 360 consistency pairs | offline split | adapter failed action prediction axis |
| FCAR | full 0.100145 vs static 0.09118, rank4 LoRA 0.07619, Base 0.123998 | rank4 LoRA and static merge | tiny gate not enough | 120/40/40 split | offline split | improvement not robust against simple controls |
| PhaseBarrier | valid repair full 0/20, Base 8/20, ablation 9/20, global damping 0/20 | Base and ablation | ablation beats full | 100 paired total | repaired canonical closed-loop | original positive rerun was invalid; valid repair killed component |
| FEDO | faulted full 1 vs strongest 2; clean frozen 4 vs full 0 | clean frozen | retention collapsed | Stage A | direct paired | valid Stage A catastrophic/retention kill |
| PTC | full 0/10 vs Base 3/10 | Base | mechanism active | Stage A 10 | direct paired | full failed while Base succeeded |
| SACF | full 0/10 vs Base 7/10 | Base | mechanism active | Stage A 10 | direct paired | catastrophic under governance |
| OCFN | full 26/80 vs zero-noise 27/80 | zero-noise | no useful improvement | expanded 80 | expanded paired | useful gain excluded after expansion |
| CBFD | full 0/10 vs Base 7/10 | Base | mechanism valid | Stage A 10 | direct paired | catastrophic under governance |
| SCVC | full 11/40 vs shifted Base 20/40 | shifted Base | full worse | Stage B 40 | paired CI [-0.425,-0.025] | CI excludes positive direction |
| PSE | full 50/80 vs bright-single 51/80 | bright single | no useful gain | expanded 80 | upper CI excludes useful improvement | retention did not produce improvement |
| RCV | full 20/40, Base 14/40, stateless/no-context 24/40 | stateless/no-context | ablations beat full | Stage B 40 | paired Stage B | retrieval explanation not needed |
| FANG | full 11/40, Base 16/40, AFIL 15/40, ablation 11/40 | Base/AFIL | ablation ties full | Stage B 40 | paired Stage B | gate did not add value |
| EvoState | 4,221 pairs, improvement 0.024689 below 0.05 | actionless baseline | mechanism/data valid | 4,221 validation pairs | scoped validation threshold | robust design stop before rollout |
| RAC | full 1/40, Base 1/40, proxy 1/40, ablation 2/40, inverse 2/40 | ablation/inverse | controls tie or beat | Stage B 40 | paired Stage B | reflective correction not useful |
| MTF | full 26/40 vs no-retention 32/40 | no-retention | ablation beats full | Stage B 40 | delta -0.15 CI [-0.275,-0.025] | retention component harmful |
| DAGR | full 18/40 vs Base 28/40, heuristic 24/40 | Base/heuristic | ablation 16, prior proxy 5 | Stage B 40 | CI full-base [-0.4,-0.1] | guidance lost to Base/simple |
| MARC | full 0/10 vs Base 8/10, no-gate/static 7/10 | Base | no-gate/static strong | Stage A 10 | direct paired | valid catastrophic Stage A kill |
| EAC | full 29/40 vs Base/AAC/ablation 30/40 | Base/AAC/ablation | ablation beats full by one | Stage B 40 | paired Stage B | Stage A promise did not survive fair Stage B |

## 7. Non-Scientific Failures

These 31 failures are not proof that the corresponding scientific family is impossible. They are process, data, environment, headroom, measurement, or implementation outcomes.

| Class | Routes | Why not a scientific kill |
|---|---|---|
| IMPLEMENTATION_OR_OPTIMIZATION_FAILURE | custom SmolVLA 7D adapter, COVI, IARC, FAMR, SPARC, HEST, HASTE, KITE, VDR, RAP, AMP, AFID, CSPR, MCI | the local realization or objective scale failed before a fair closed-loop formulation test |
| DATA_OR_SUPERVISION_FAILURE | SafeTrace, CensorCredit, G3P, PESA, NICE, S2C, MHS, DCCG | labels, contrast, caches, or targets collapsed; this invalidates the experiment rather than closing the method family |
| CONDITION_TOO_SEVERE | LIFT | compute or local condition made the planned experiment infeasible |
| NO_USABLE_HEADROOM | PCAV, CFR, TSC, URF | the claimed condition did not expose useful residual improvement |
| MEASUREMENT_INVALIDITY | ContactTube | measurement depended on missing/invalid pose or clipping semantics |
| DESIGN_FAILURE | CCIF, LCG, BRID | the design or pre-rollout mechanism did not define a usable scientific test |
| CONTEXT/SESSION INTERRUPTION | current audit context handoff and prior exhausted Codex thread | at least one context exhaustion occurred; exact count `NOT_RECORDED`; interruption is operational, not scientific evidence |

## 8. Underpowered, Ambiguous, or Potentially Misclassified Results

| Route | Evidence | Decision | False-negative risk | Bounded reopen? |
|---|---|---|---|---|
| TCA-Select | no meaningful gain and representation collapse risk, but no later official closed-loop protocol | correctly archived historically | low | No |
| AMP-GD | toy success 1.0 but LIBERO tiny did not beat safety/random | correctly archived | medium | No; novelty/headroom weak |
| ResetSpec-Retarget | global scale reward/success 1, object-relative 0 | correctly archived | medium | No; core claim unstable |
| ContactSet | only 6 demos; full worse than simple variants | underpowered archive | medium | No; evidence too weak and old |
| DICD | 1/10 vs 2/10 and 1/10 | underpowered Stage A archive | medium | No; later temporal methods supersede |
| GCAP | target-axis full 3 vs Base 4 and Sobel 5, but clean full 5 vs Base 1 | underpowered target-axis archive | medium | No; target axis failed |
| CAVM | full 24/58 vs nearest-success 23/58 | unresolved strongest near-miss | medium-high | No under current governance; one expansion already used and no third expansion allowed |
| CALA | full RMSE 3.1988 vs action-history-only 3.14397, margin -0.011718 | possible false negative | medium | Maybe scientifically, but not strategically recommended |
| RAR | full RMSE 0.171954 vs zero-residual 0.165597, margin -0.038376 | possible false negative | medium | Maybe scientifically, but not strategically recommended |

CALA and RAR are the clearest Stage 0 false-negative risks because their margins were small and not paired closed-loop evidence. They still do not justify the next campaign action because neither has enough surviving novelty or positive external-prior anchoring to outrank a reset of candidate selection.

## 9. Positive Signals and Near-Misses

| Rank | Route | Positive evidence | Novelty strength | Closed-loop evidence | Strongest comparison | Why not advanced | Reusable? |
|---|---|---|---|---|---|---|---|
| 1 | CAVM | best valid closed-loop point estimate in repository: 24/58 | moderate memory mechanism | expanded Stage B | nearest-success 23/58, Base 22/58 | one-episode gain, no third expansion, no second backbone, no external prior confirmation | conceptually reusable, result closed |
| 2 | EAC | Stage A looked promising and Stage B remained near-tie | moderate adaptive controller | Stage B | Base/AAC/ablation 30/40 vs full 29/40 | strongest controls beat full by one | reusable as cautionary control set |
| 3 | RCV | full beat Base 20/40 vs 14/40 | moderate retrieval mechanism | Stage B | no-context/stateless 24/40 | simple ablations explained the benefit | retrieval diagnostics reusable |
| 4 | DAGR | beat prior proxy and ablation, showing mechanism activity | moderate retrieval/guidance | Stage B | Base 28/40 and heuristic 24/40 | Base/simple dominated | protocol reusable |
| 5 | PSE | large expanded 80-pair evaluation, near tie 50/80 vs 51/80 | low-moderate state estimator | Stage B | bright-single 51/80 | upper CI excluded useful improvement | statistics template reusable |

Positive offline metrics alone were not ranked. CensorCredit had an apparent positive prototype, but the later postmortem found identical censored/uncensored labels and heads, so it is invalid rather than a near-paper result.

## 10. External-Prior Comparison Audit

Among the 47 formal autonomous methods, official external-prior reproduction count is 0. Transparent or local proxy comparison count is 26. No external-prior experiment count is 21. These counts are route-level formal-campaign counts; historical routes often used conceptual baselines but not comparable official prior implementations.

Serious methods did name closest priors, especially after governance tightened. However, the comparison was usually a proxy: APEX proxy for FEDO, CAG/null for SACF, SV-deviation for RCV, success-memory for CAVM, AFIL for FANG, Reflective proxy for RAC, FrameSkip for MTF, DAM for DAGR, OpenVLA L1 proxy for MARC, AAC proxy for EAC, and ROVLA-style multi-consistency proxy for MCI. Published numbers were generally not treated as direct baselines in later reports, but the absence of official prior code/checkpoint execution leaves a reviewer-facing gap.

Fairness summary: later proposals disclosed proxy status more consistently; earlier routes sometimes treated literature as motivation rather than an executable comparator; no formal method produced a direct official-prior reproduction under matched LIBERO/SmolVLA semantics.

## 11. LoRA / Low-Compute Strategy Audit

LoRA was intended as compute infrastructure, not the scientific contribution. The campaign partially respected this by later governance, but it also underused official LoRA early and overemphasized trivial frozen-policy attachments later.

Verified LoRA/adapter details: official SmolVLA rank4 LoRA targeted `lm_expert` q/v projections plus state/action projections and had 185,664 trainable parameters, 0.0412 percent of 450,231,840. Rank16 feasibility had 742,656 trainable parameters. PatchGuard used rank4 targets `state_proj`, `action_in_proj`, and `action_out_proj`, with 9,984 trainable of 450,056,160. Custom 7D rank4/rank8 adapters trained 128,007/131,975 trainable parameters; rank8 had the best offline result but failed exact replay. TG7D rank4 had 295,623 trainable. FCAR used a regenerated rank4 baseline plus a tiny CPU gate. FAMR used an official rank4 LoRA task vector. Quantized OpenVLA-OFT INT4 was inference diagnostic, not QLoRA training.

Failure attribution: many failures were not caused by LoRA capacity because the method never reached a fair capacity test. Some early failures were action-interface and checkpoint persistence problems. Later failures more often came from data/headroom/objective-scale collapse. The campaign should keep LoRA as an implementation scaffold only when it isolates the scientific mechanism; standard LoRA should remain conditional, not automatic.

## 12. Research-Funnel Statistics

| Funnel quantity | Count | Notes |
|---|---:|---|
| Formal candidate-generation artifacts | 45 | 44 epoch candidate-generation reports plus one initial autonomous candidate-discovery report |
| Formally enumerated candidates | 135 | exactly three per candidate artifact |
| Selected/proposed formal methods | 47 | includes FEDO/GCAP automatic pivots without separate exact-three artifacts |
| Total routes in full audit ledger | 73 | 26 historical/diagnostic plus 47 formal |
| Implemented routes | 70 | three unimplemented: SafeLoRA, TG-VLA, ISAC |
| Formal source/runner verified | 47 | every formal method has source and runner evidence |
| Formal focused tests verified | 45 | no focused test detected for G3P and PESA |
| Formal Stage 0/0a result JSON count | 27 by current suffix/glob inventory | late stage0-heavy epoch dominates |
| Formal Stage A count | 17 | DICD through EAC subset |
| Formal Stage B count | 10 | OCFN, SCVC, PSE, RCV, CAVM, FANG, RAC, MTF, DAGR, EAC |
| Route-level Stage-A-equivalent count | 19 | formal 17 plus PhaseBarrier and CensorCredit historical prototypes |
| Route-level Stage-B-equivalent count | 11 | formal 10 plus PhaseBarrier repaired prototype |
| Trained checkpoint route count | 31 | 23 formal plus 8 historical routes |
| Second-backbone Ours count | 0 | OpenVLA INT4 was diagnostic only |
| Paper-candidate GO count | 0 | no valid `PROTOTYPE_GO` |

Loss breakdown from ledger: 3 exact-prior/resource/preimplementation rejections; 5 no-headroom or condition-too-severe candidate failures counted as non-scientific; 8 data/supervision failures; 14 implementation/optimization failures; 11 simple-baseline/key-ablation/clean-retention scientific kills in formal Stage A/B; 5 underpowered formal/historical closed-loop archives; 4 broader historical valid kills from offline or replay evidence; and 7 infrastructure/diagnostic/no-claim rows. Conversion rates: selected formal proposal to Stage A, 17/47 = 36.2 percent; selected formal proposal to Stage B, 10/47 = 21.3 percent; Stage A to Stage B, 10/17 = 58.8 percent; Stage B to GO, 0/10 = 0 percent; selected formal proposal to second-backbone Ours, 0/47 = 0 percent.

## 13. Compute and Operational Audit

Current campaign state records 5.21 GPU hours and 14.845 GiB downloaded for the current autonomous campaign, but repo-wide GPU hours are `NOT_RECORDED`. The state start for the current autonomous campaign is 2026-07-12T16:32+09:00; the full repository campaign spans from first commit 2026-06-27T10:30:15+09:00 to audit snapshot 2026-07-17T03:26:32+09:00, about 19.7 wall-clock days.

Simulator episodes: formal autonomous method pipeline has 3,380 clearly countable paired closed-loop episodes from final artifacts: 890 Stage A/2a episodes plus 2,490 Stage B/2b episodes. Canonical non-quarantined route-level lower bound is at least 3,572 episodes after adding PhaseBarrier valid repair 100, official baseline pilot/smoke 52, and OpenVLA/SmolVLA hard-slice diagnostic 40. Including original exploratory PhaseBarrier/CensorCredit attempts gives at least 3,592, but repo-wide exact total is `NOT_RECORDED` because replay branches, invalid attempts, and partial reruns are not globally normalized.

Assets and storage: `C:\assets\data` is about 93.545 GiB, `hf_home` 1.902 GiB, `datasets` 1.862 GiB, `checkpoints` 1.696 GiB, and `repos` 1.028 GiB. WSL model assets include OpenVLA-OFT around 15 GiB and SmolVLA around 865 MiB. Largest known campaign download in state is 14.845 GiB for OpenVLA-related assets.

Operational interruptions: at least one Codex context exhaustion is known because this audit resumes from an exhausted previous thread; exact context compaction/restart count is `NOT_RECORDED`. Approval interruptions are `NOT_RECORDED`. Duplicate or avoidable rerun classes include official LoRA regeneration/drift due missing checkpoints/unpinned RNG, invalid retrained PhaseBarrier repair, COVI invalid v1 plus repair, PCAV expansion name error/resume, VDR self-worker confusion, RAP launcher preflight failure, KITE resume, and SPARC capture reset.

Git/commit scale: 802 commits exist across all refs; 798 are ancestors of audit-start HEAD. Approximate all-ref churn is dominated by generated report artifacts: `reports` +12,257,382/-31,110 lines across 4,695 file touches; `tca_map` +103,360/-737; `scripts` +83,016/-616; `tests` +44,551/-3,284.

## 14. Repetition and Search-Space Audit

Recurring families were repeatedly renamed. Candidate ranking/verifiers include TCA-Select, PESA, PCAV, and ECHO. Post-hoc residual correction includes FCAR, RAR, COVI, FAMR, SPARC, CFR, URF, BRID, CSPR, and MCI. Action filtering/damping includes CSS, ExecSpec, PTC, SACF, RAC, EAC, AMP, and PhaseBarrier. Memory/retrieval includes RCV, CAVM, MTF, DAGR, and RAP. Visual canonicalization/TTA includes PRISM, OCFN, SCVC, FANG, VDR, and COVI-adjacent routes. Temporal/history heads include DICD, PSE, CALA, HEST, HASTE, TSC, MHS, and NICE. Supervision/credit includes SafeTrace, CensorCredit, FEDO, and G3P. Representation/action-generation includes TCA-Map, ActionMap, TG7D, CBFD, EvoState, AFID, and MCI.

Common failed assumptions: small frozen-policy attachments would expose publishable gains; action L2 or offline probes would predict closed-loop success; memory/retrieval would beat stateless/simple controls; visual canonicalization would solve policy brittleness without disrupting clean behavior; and local transparent proxies would be enough for reviewer-grade prior comparisons. The campaign explored many names, but the underlying mechanism space narrowed too often around cached-feature heads, residual gates, small adapters, and verifier-style filters attached to frozen SmolVLA.

## 15. Why the Campaign Is Not Finished

Ranked causes by impact:

1. Candidate quality and anchoring: many candidates were not built around a strong positive external prior with official runnable code. Evidence: 0/47 formal methods have official prior reproduction; 26/47 only proxy.
2. No stable positive problem condition: several late Stage 0s found no usable headroom or collapsed labels. Evidence: PCAV/CFR/TSC/URF no-headroom, NICE/CensorCredit label collapse, MCI raw no-headroom corrected to implementation failure.
3. Repeated narrow method families: residuals, gates, memories, and history heads were renamed frequently without changing enough core dimensions. Evidence: ledger family clustering in section 14.
4. Pretrained-policy disruption or nonacting mechanisms: Stage A/B methods often lost to Base, ablation, or simple stateless controls. Evidence: RCV, DAGR, MARC, MTF, EAC.
5. Late or weak external-prior comparison: proxy priors arrived, but official comparators did not. Evidence: formal prior audit section 10.
6. Low-compute parameterization confusion: LoRA became both underused early due action-interface blockers and overgeneralized later as lightweight local attachments. Evidence: LoRA audit section 11.
7. Underpowered or ambiguous early decisions: DICD, GCAP, CAVM, CALA, and RAR leave false-negative risk, but not enough to form a paper. Evidence: section 8.
8. Documentation/state churn and context interruptions: state JSON commit lagged behind HEAD, and the prior thread exhausted context. Evidence: snapshot state-file caveat and this audit handoff.
9. Hardware constraints: RTX 5080 was sufficient for lightweight local work and INT4 diagnostics, but not for broad full fine-tuning or large external-prior reproduction. Evidence: LIFT compute infeasibility and LoRA/INT4 infrastructure reports.

Scientific difficulty dominates the outcome, but process/governance also mattered: early broad search before official simulator stabilization and later proxy-heavy comparisons created many honest negative artifacts without producing a reviewer-ready positive result.

## 16. False-Negative Audit

Potential false negatives exist, but none should drive immediate continuation. CAVM is the strongest: full 24/58 beat nearest-success 23/58, Base 22/58, and ablation 21/58. Missing evidence is a larger preregistered confirmation and second-backbone result, but current governance allows no third expansion after one 58-pair expansion. CALA and RAR had small Stage 0 margins without closed-loop confidence analysis; missing evidence is a cheap decisive paired validation or uncertainty analysis. DICD and GCAP were underpowered Stage A archives, but later related methods tested richer temporal and goal-conditioned variants.

Reviewer B overreach risk is real mainly for Stage 0 point-estimate stops, not for completed Stage B kills. Treating implementation/data failures as method failures was corrected in later governance; MCI explicitly classifies as implementation failure and not scientific kill. The audit does not recommend a bounded reopen because the best false-negative candidates are either governance-closed (CAVM) or insufficiently novel/anchored (CALA/RAR) compared with resetting candidate selection.

## 17. Paper-Readiness Checklist

Nearest method used for checklist: CAVM-VLA, because it is the strongest valid near-miss.

| Requirement | Status | Gap |
|---|---|---|
| Defensible novelty | PARTIAL | memory mechanism plausible but closest prior not officially reproduced |
| SmolVLA Base vs Base + Ours | PARTIAL | full 24/58 vs Base 22/58, too small |
| Closest prior vs Ours | PARTIAL | nearest-success memory proxy 23/58, not official prior |
| Key ablation | PARTIAL | no-contrast 21/58, but effect small |
| Relevant simple control | PARTIAL | nearest-success and Base tested |
| Clean retention | PARTIAL | no paper-grade clean-retention package |
| Adequate paired statistics | MISSING | one-episode gain after allowed expansion |
| Quantized OpenVLA-OFT INT4 + Ours | MISSING | diagnostic OpenVLA exists, no CAVM on OpenVLA |
| Second claim-specific condition | MISSING | not run |
| Efficiency | PARTIAL | local compute known, no final method efficiency table |
| Reproducibility | PARTIAL | artifacts exist, not a paper package |
| Figure/table-ready artifacts | MISSING | no final package |

Exact gap to `READY_TO_DRAFT_RAL_PAPER_PACKAGE`: no `PROTOTYPE_GO`, no official-prior win, no statistically adequate positive Stage B, no Ours-on-second-backbone result, no second condition, and no figure/table-ready reproducibility package.

## 18. Missed or Unreported Events

Progress likely occurred between user-visible updates because the previous thread exhausted context while the repo kept accumulating durable artifacts. Notable hidden or easy-to-miss events: CSPR and MCI completed immediately before this audit, both as Stage 0 implementation failures; DCCG/MHS/S2C ended as data or cache coverage failures; multiple late stage0-heavy cycles did not reach rollout; COVI v1 was invalid then repaired; PhaseBarrier had an invalid retrained positive result quarantined and a valid bounded repair that killed the component; CensorCredit looked positive until the label/head identity postmortem; EAC had a promising Stage A but failed/tied in Stage B; CAVM remained the strongest near-miss but is governance-closed after one expansion.

Automatic pivots included FEDO and GCAP in the initial autonomous set and many epoch 4 candidate selections. Branch audit found `codex/execspec-repair-state0-state1` with one unique historical ExecSpec state1 kill-gate commit absent from HEAD, but later ExecSpec state2/3/3.5 artifacts in current history supersede it. At audit start there was no uncommitted scientific work and no active research worker. The only local work created by this run is this audit report.

## 19. Recommended Strategic Decision

Recommendation: `RESET_CANDIDATE_SELECTION_STRATEGY`.

Justification: continuing the current candidate-selection style would likely produce another cached-feature head, residual gate, verifier, or local proxy that dies in Stage 0 or loses to Base/ablation. Resetting the gates alone is insufficient because many failures are not overly harsh gates; they are weak candidate anchoring, no headroom, collapsed labels, and proxy-only prior comparisons. A bounded reopen is not the best next action because the strongest reopen candidate, CAVM, already used its allowed expansion, while CALA/RAR lack enough novelty and closed-loop promise to outrank a better candidate search. A strategic new epoch may follow, but the immediate decision is to reset selection criteria before generating the next exact-three candidates.

## 20. Exact Resume Plan

`Exact Next Codex Prompt After User Review`

```text
Resume the autonomous VLA research campaign in C:\Users\jiheo\tca_map after the full-history audit.

Branch: codex/full-history-audit-before-resume
Last scientific HEAD before audit report: bc15132d74741c5b03c253c9bc062d8c5aaa5ddc
Current scientific state: epoch 4, cycle 39, epoch_4_cycle_39_candidate_search_pending
Previous method: MCI-VLA
Previous decision: MCI_STAGE_0_IMPLEMENTATION_FAILURE
Selected audit recommendation: RESET_CANDIDATE_SELECTION_STRATEGY

Exact next scientific action:
Generate exactly three Cycle 39 candidates under reports/current_research_governance.md, but reset candidate selection strategy before scoring. Each candidate must start from a strong positive external prior or a clearly justified exception, identify official code/checkpoint status, reconstruct the prior mechanism, and show a local fair comparison path. Do not implement, train, or rollout until the three candidates are documented, scored, and one is selected.

Prohibited repeats:
Do not rescue or retune MCI-VLA or CSPR-VLA. Do not propose another generic cached-feature residual, small frozen-policy gate, history head, verifier, visual canonicalizer, memory lookup, no-headroom probe, or proxy-only prior unless it changes at least two core dimensions and has clear headroom. Do not treat OpenVLA INT4 diagnostic success as an Ours result.

Time-to-evidence requirement:
Before implementation, require a cheap evidence plan that can falsify the candidate within one bounded development audit or Stage A. Prefer less than 4 hours for prior/headroom/data-health evidence and less than 12 hours to a decisive prototype stop or Stage A launch decision. Record GPU, disk, download, and resume paths.

LoRA role:
Use LoRA/QLoRA only as low-compute parameterization. Keep the scientific mechanism separable from PEFT. Include standard LoRA only when it is a real alternative explanation or shared scaffold control. If adapter capacity is the bottleneck, classify LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT and allow at most one bounded capacity adjustment before confirmatory testing.

Reviewer false-negative safeguards:
For any Stage 0 or pre-rollout stop, classify evidence as FATAL_PREIMPLEMENTATION, ROBUST_EMPIRICAL_DESIGN_FAILURE, UNDERPOWERED_OR_UNRESOLVED, or IMPLEMENTATION_OR_DATA_FAILURE. Do not kill on tiny point estimates without uncertainty, collapsed labels, missing headroom, or implementation defects. Preserve the strongest fair interpretation and the narrowest publishable claim.

Conditions for implementation and rollout:
Implementation may start only after source, runner, focused tests, data/contrast health, problem headroom, objective-scale audit, identity-preserving integration, prior/proxy fairness, and resource risk are documented. Closed-loop rollout may start only after checkpoint reload, finite gradients when applicable, bounded action deltas, full-vs-ablation difference, clean retention, no privileged inference signal, exact paired manifest, and frozen decision thresholds.
```
