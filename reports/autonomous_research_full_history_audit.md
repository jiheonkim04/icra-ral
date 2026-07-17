# Autonomous VLA Research - Full History Audit

This report is the mandatory Phase A audit before any new candidate generation, implementation, training, or rollout. It was first created at commit `b0ecb6e` and refreshed on the live `codex/epoch5-official-prior-first` branch after the repository moved into an official-prior-first Epoch 5. Evidence precedence follows the current Goal instruction, `reports/current_research_governance.md`, `AGENTS.md`, current result artifacts, current campaign state, git history, and historical reports. Missing facts are recorded as `NOT_RECORDED` rather than guessed.

## 1. Executive Summary

No paper-ready method exists. No valid `PROTOTYPE_GO` method exists. The repository contains substantial reusable infrastructure and a large body of negative, invalid, underpowered, and diagnostic evidence, but it does not contain `READY_TO_DRAFT_RAL_PAPER_PACKAGE`. The current Epoch 5 work is an official-prior-first diagnostic, not an Ours method and not a paper result.

Route count remains 73 distinct research routes, consisting of 26 historical or diagnostic routes plus 47 formal autonomous method proposals. The existing OpenVLA-OFT diagnostic route now has an Epoch 5 continuation, but it is not counted as an additional Ours method. Implemented route count remains 70 of 73 with code-level or runner-level implementation evidence; `SafeLoRA-VLA`, `TG-VLA`, and `ISAC-VLA` did not reach an implemented local experiment. Trained route count remains 31 with verified training or checkpoint artifacts; 23 of those are in the formal 47-method campaign. Closed-loop Stage A count remains 17 formal autonomous methods; 19 route-level methods reached Stage-A-equivalent closed-loop evidence when historical prototypes are included. Stage B count remains 10 formal autonomous methods; 11 route-level methods reached Stage-B-equivalent closed-loop evidence when the repaired PhaseBarrier prototype is included. Second-backbone Ours count remains 0.

Outcome totals used in this audit remain 26 valid scientific kills, 31 non-scientific failures, and 9 underpowered or unresolved results. Seven additional rows are infrastructure, diagnostic, no-claim, or preimplementation rejections and are not counted as scientific failures. The current active stage is Epoch 5 official-prior-first residual diagnosis: selected prior `OpenVLA-OFT on LIBERO`; previous method `MCI-VLA` remains closed as `MCI_STAGE_0_IMPLEMENTATION_FAILURE`.

The strongest result obtained is `CAVM-VLA`: after one allowed expansion, full reached 24/58 versus nearest-success memory 23/58, Base 22/58, and no-contrast 21/58. It is the best near-miss, but the effect is a one-episode advantage, no third expansion is allowed, and no second-backbone or external-prior confirmation exists.

The campaign is not paper-ready mainly because: no method beats Base, closest prior/proxy, key ablation, and simple reviewer-killer control in a valid Stage B; official-prior comparison arrived only as an Epoch 5 diagnostic after the prior method cycle; many late methods died before rollout from data, headroom, objective-scale, or implementation failures; the search repeatedly favored lightweight frozen-SmolVLA attachments; and there is still no same-method Quantized OpenVLA-OFT INT4 plus Ours result. Epoch 5 has found a promising prior-positive residual slice, OpenVLA-OFT INT4 14/16 versus SmolVLA Base 7/16, but the upper/headroom check and all Ours-side evidence are not yet complete.

## 2. Audit Snapshot

| Field | Value |
|---|---|
| Snapshot timestamp | 2026-07-17T12:45:29.5487505+09:00 |
| Audit branch | `codex/epoch5-official-prior-first` |
| Scientific HEAD at audit refresh | `ffb55f57e1cd978a35d0d84c9ede487559e02fa0` |
| HEAD commit subject | `Preregister epoch 5 OpenVLA residual diagnostic` |
| Git status at audit refresh | `## codex/epoch5-official-prior-first...origin/codex/epoch5-official-prior-first`; modified `tca_map/smolvla/official_wsl_libero_rollout.py`; untracked `scripts/launch_epoch5_residual_job.sh`; untracked `rollouts/2026_07_17/` OpenVLA videos |
| `main` HEAD | `8dc4de2fdbf576ace8bdf3699d190b761553c1fa` |
| Prior audit commit | `b0ecb6ea5f6eba2953b5bd842883c0474d634dff` (`Add full history research audit`) |
| New commits after prior audit | `d268a83` (`Start epoch 5 official-prior-first campaign`), `ffb55f5` (`Preregister epoch 5 OpenVLA residual diagnostic`) |
| Local/remote branch inventory | hundreds of branch refs; current branch is already pushed to `origin/codex/epoch5-official-prior-first` |
| Unmerged branch with unique work | `codex/execspec-repair-state0-state1`; one historical ExecSpec state1 kill-gate commit, superseded by later ExecSpec state2/3/3.5 evidence |
| Active Windows Python research worker | none detected |
| Active WSL Python research worker | none at refreshed snapshot; the previously active SmolVLA residual worker exited during the audit |
| Worker classification | `COMPLETED_DURING_AUDIT_WITHOUT_INTERVENTION`; launcher PID `394`, child Python PID `399`, exit code `0`, finished `2026-07-17T12:40:48+09:00` |
| Worker heartbeat/status | last observed heartbeat `2026-07-17T12:40:18+09:00`; `runs/openvla_oft_int4/epoch5_libero10_residual_v1/smolvla_exit_code.txt` is `0` |
| CUDA compute snapshot | RTX 5080, 16,303 MiB total, 2,309 MiB used, GPU utilization 10 percent, 37 C; no active research Python compute app listed after SmolVLA exit |
| RAM snapshot | WSL: 11 GiB total, 3.8 GiB used, 7.0 GiB available; swap 3.0 GiB |
| Disk snapshot | `/mnt/c`: 931 GiB total, 633 GiB used, 298 GiB available; `/home`: 1007 GiB total, 41 GiB used, 916 GiB available |
| Current epoch/cycle/stage | Epoch 5 official-prior-first residual diagnostic; state JSON still names `epoch_5_official_prior_ecosystem_selection` |
| Current selected prior | `OpenVLA-OFT on LIBERO`, quantized INT4 local diagnostic, not full-precision claim |
| Current/previous method | current route is prior diagnostic only; previous Ours method `MCI-VLA` |
| Current/previous decision | previous Ours decision `MCI_STAGE_0_IMPLEMENTATION_FAILURE`; current prior decision `OPENVLA_OFT_PRIOR_REPRODUCTION_RECOVERED_AND_VALIDATED_RESIDUAL_PENDING` |
| Current next action in state | stale state says select three official-prior ecosystems; current artifacts already selected OpenVLA-OFT, preregistered `epoch5_libero10_residual_v1`, and completed matched Base/Prior residual execution |
| Checkpoint path in state | `/mnt/c/assets/checkpoints/smolvla_libero` |
| Current MCI result paths | `reports/mci_vla/stage_0_result.json`, `reports/mci_vla/stage_0_partial.json`, `reports/mci_vla/stage_0_manifest.json`, `reports/mci_vla/stage_0_adjudication.md` |
| Current Epoch 5 report paths | `reports/epoch5_prior_ecosystem_selection.md`, `reports/epoch5_prior_reproduction_plan.md`, `reports/epoch5_prior_reproduction_result.md`, `reports/epoch5_prior_reproduction_result.json`, `reports/autonomous_compact_handoff.md` |
| Current residual artifacts | OpenVLA result `runs/openvla_oft_int4/epoch5_libero10_residual_openvla_int4.json`; SmolVLA result `runs/openvla_oft_int4/epoch5_libero10_residual_smolvla_exact.json`; OpenVLA manifest `runs/openvla_oft_int4/epoch5_libero10_residual_openvla_manifest.json`; SmolVLA manifest `runs/openvla_oft_int4/epoch5_libero10_residual_smolvla_manifest.json` |
| State-file commit caveat | `reports/autonomous_until_paper_state.json` and `reports/autonomous_ral_campaign_state.json` record stale `current_commit` `b0ecb6e`; live HEAD, Epoch 5 reports, and run artifacts are newer and authoritative |

Current Epoch 5 delta at the snapshot:

| Item | Evidence | Audit interpretation |
|---|---|---|
| Prior selection | `reports/epoch5_prior_ecosystem_selection.md` | OpenVLA-OFT selected first; pi0.5/OpenPI and PCD remain fallback ecosystems |
| Recovered prior hard slice | `reports/openvla_oft_quantized_hard_slice_result.json` | OpenVLA-OFT INT4 20/20 versus matched SmolVLA frozen Base 11/20; prior positive but saturated |
| Residual preregistration | `reports/epoch5_prior_reproduction_plan.md` | `epoch5_libero10_residual_v1` frozen before residual evaluation |
| OpenVLA residual result | `runs/openvla_oft_int4/epoch5_libero10_residual_openvla_int4.json` | completed 16/16, succeeded 14/16, no infrastructure failures; prior leaves two failures on task 8 |
| SmolVLA matched Base residual result | `runs/openvla_oft_int4/epoch5_libero10_residual_smolvla_exact.json` | completed 16/16, succeeded 7/16, no infrastructure failures; exit code `0` |
| Matched manifests | OpenVLA manifest SHA-256 `b2de1d683d7ab0c5aff7462857f0366bd72c9208c98b2e8566e6a42a296b5adf`; SmolVLA manifest SHA-256 `6defb7769a75b595bc8456e6938254d7185d2b03fd94a4bda4fd0a95464a837c` | both have label `epoch5_libero10_residual_v1`, 16 episodes, tasks 8/9, resets `20260716..20260723`, and identical first/last initial-state hashes |
| OpenVLA residual per-task outcome | task 8: 6/8; task 9: 8/8 | residual exists for prior on task 8 |
| SmolVLA Base per-task outcome | task 8: 3/8; task 9: 4/8 | Base has meaningful failure on both tasks |
| Matched Base/Prior outcome | OpenVLA 14/16 versus SmolVLA 7/16 | prior improves over Base and leaves residual failures; upper/headroom check remains unrun |
| Worker completion | `runs/openvla_oft_int4/epoch5_libero10_residual_v1/smolvla_finished_at.txt` | finished during audit at `2026-07-17T12:40:48+09:00`; audit read the result but did not launch follow-up work |
| Uncommitted scientific support code | `scripts/launch_epoch5_residual_job.sh`, `tca_map/smolvla/official_wsl_libero_rollout.py` diff | launcher chooses separate OpenVLA and official SmolVLA envs; optional PaliGemma stub is local compatibility support |
| Untracked rollout videos | `rollouts/2026_07_17/*.mp4` | 16 OpenVLA videos match the residual OpenVLA run; leave uncommitted unless explicitly curated |

The SmolVLA JSON at `runs/openvla_oft_int4/epoch5_libero10_residual_smolvla_exact.json` was initially treated as stale while the worker was active. It became interpretable only after `smolvla_exit_code.txt` appeared with exit code `0`; the audit then recorded the clean 7/16 Base result. No upper/headroom check or Ours work was started.

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
| Official-prior-first branch pivot | `d268a83` | `reports/epoch5_prior_ecosystem_selection.md` | selected OpenVLA-OFT before Ours; preserved pi0.5/OpenPI and PCD as fallbacks | valid strategy correction | process infrastructure, not a method |
| Residual-condition manifest control | `ffb55f5` | `reports/epoch5_prior_reproduction_plan.md`, `tests/test_openvla_oft_int4_gate.py` | frozen task/reset identities and manifest labels for OpenVLA/SmolVLA matched residual run | valid infrastructure | prerequisite before Ours, not paper method |
| OpenVLA-OFT residual diagnostic | uncommitted run artifacts after `ffb55f5` | `runs/openvla_oft_int4/epoch5_libero10_residual_openvla_int4.json`, `runs/openvla_oft_int4/epoch5_libero10_residual_smolvla_exact.json` | OpenVLA-OFT INT4 14/16 versus SmolVLA Base 7/16, matched manifests, no infrastructure failures | valid matched Base/Prior diagnostic; upper/headroom pending | prior diagnostic only |
| Epoch 5 durable worker launcher | uncommitted local file | `scripts/launch_epoch5_residual_job.sh` | detached WSL launcher writes PID, heartbeat, stdout/stderr, exit code, finished time, and resume command; separates OpenVLA and SmolVLA envs | useful but not yet committed | operational infrastructure |
| Partial checkpoint/resume infrastructure | `c4607b8` and later | `reports/pse_vla/stage_b_partial_result.json` plus many partials | 52 tracked partial JSON files, 23 exit-code files, 21 heartbeats, 29 PID files | valid operational infrastructure | infrastructure only |
| Durable WSL worker launchers | `e344233`, `9556599` | `scripts/*launch*`, `reports/*heartbeat*` | durable workers, heartbeats, missing-key-only resume behavior | valid for future long runs | infrastructure only |

Tracked-file inventory at audit refresh: 2,640 tracked files; 1,775 under `reports`, 274 under `scripts`, 237 under `tests`, 199 under `tca_map`, 103 under `runs`, 25 under `configs`, and 21 under `rollouts`. Current untracked files include the Epoch 5 launcher and 16 OpenVLA residual videos. Machine-readable artifact inventory under tracked `reports` includes 152 `*_result.json` files, 72 manifest JSON files, 34 state JSON files, and 52 partial JSON files; earlier all-ref inventory was higher because it included historical refs and tracked run artifacts outside `reports`.

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
| 23 | historical plus epoch5 | Quantized OpenVLA-OFT INT4 diagnostic | second-backbone hard-slice and residual-prior check | infrastructure/diagnostic | OpenVLA-OFT | `5c2a364`, `d268a83`, `ffb55f5` | Yes | No | Yes | Yes | NA | No | No | Yes | recovered hard slice OpenVLA 20/20 vs matched SmolVLA 11/20; Epoch 5 residual OpenVLA 14/16 vs matched SmolVLA Base 7/16 | diagnostic only, upper/headroom pending | VALID_CANONICAL_PARTIAL | No | No | `reports/openvla_oft_quantized_hard_slice_result.md`, `reports/epoch5_prior_reproduction_result.md`, `runs/openvla_oft_int4/epoch5_libero10_residual_openvla_int4.json`, `runs/openvla_oft_int4/epoch5_libero10_residual_smolvla_exact.json` |
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
| 2026-07-17 | `b0ecb6e` | first full-history audit committed | audit | 73 routes, 26 valid scientific kills, 31 non-scientific failures, 9 unresolved | recommended resetting candidate-selection strategy |
| 2026-07-17 | `d268a83` | Epoch 5 official-prior-first branch started | prior selection | selected OpenVLA-OFT and validated saturated 20/20 prior slice | implements the audit recommendation's spirit by anchoring before Ours |
| 2026-07-17 | `ffb55f5` | residual diagnostic preregistered | OpenVLA-OFT prior | tasks 8/9, resets `20260716..20260723`, 16 episodes per policy | freezes matched Base/Prior comparison before interpretation |
| 2026-07-17 | uncommitted run artifact | OpenVLA residual side completed | OpenVLA-OFT prior | 14/16 success, task 8 residual failures at reset identities `20260721` and `20260722` | prior leaves residual on frozen condition |
| 2026-07-17 | uncommitted run artifact | SmolVLA matched Base completed during audit | matched Base | 7/16 success, exit code 0, no infrastructure failures | Base fails meaningfully and prior improves; upper/headroom remains pending |

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

Among the 47 formal autonomous Ours methods, official external-prior reproduction count remains 0. Transparent or local proxy comparison count is 26. No external-prior experiment count is 21. These counts are formal Ours-method counts; historical routes often used conceptual baselines but not comparable official prior implementations.

Serious methods did name closest priors, especially after governance tightened. However, the comparison was usually a proxy: APEX proxy for FEDO, CAG/null for SACF, SV-deviation for RCV, success-memory for CAVM, AFIL for FANG, Reflective proxy for RAC, FrameSkip for MTF, DAM for DAGR, OpenVLA L1 proxy for MARC, AAC proxy for EAC, and ROVLA-style multi-consistency proxy for MCI. Published numbers were generally not treated as direct baselines in later reports, but the absence of official prior code/checkpoint execution leaves a reviewer-facing gap.

Epoch 5 changes the process state but not the Ours evidence state: it selected OpenVLA-OFT before designing any method, validated an existing official-stack INT4 hard-slice diagnostic, and completed a matched residual diagnostic during this audit. That is one official-prior diagnostic route, not an official-prior win by Ours. The hard-slice prior was positive but saturated at 20/20; the residual diagnostic found OpenVLA-OFT INT4 14/16 versus matched SmolVLA Base 7/16, so the Base/Prior residual structure is present, but upper/headroom remains pending.

Fairness summary: later proposals disclosed proxy status more consistently; earlier routes sometimes treated literature as motivation rather than an executable comparator; no formal Ours method produced a direct official-prior reproduction under matched LIBERO/SmolVLA semantics. The current branch is the first serious correction toward official-prior-first comparison, but it must finish the upper/headroom check before any Ours design.

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
| Official-prior diagnostic route count | 1 | OpenVLA-OFT INT4 diagnostic, including hard-slice and current residual-prior work |
| Official-prior Ours comparison count | 0 | no Ours method has been run against official OpenVLA-OFT, pi0.5/OpenPI, or PCD |
| Completed matched Base/Prior diagnostic count | 1 | `epoch5_libero10_residual_v1`; OpenVLA 14/16, SmolVLA Base 7/16, no infrastructure failures |
| Paper-candidate GO count | 0 | no valid `PROTOTYPE_GO` |

Loss breakdown from ledger: 3 exact-prior/resource/preimplementation rejections; 5 no-headroom or condition-too-severe candidate failures counted as non-scientific; 8 data/supervision failures; 14 implementation/optimization failures; 11 simple-baseline/key-ablation/clean-retention scientific kills in formal Stage A/B; 5 underpowered formal/historical closed-loop archives; 4 broader historical valid kills from offline or replay evidence; and 7 infrastructure/diagnostic/no-claim rows. Conversion rates: selected formal proposal to Stage A, 17/47 = 36.2 percent; selected formal proposal to Stage B, 10/47 = 21.3 percent; Stage A to Stage B, 10/17 = 58.8 percent; Stage B to GO, 0/10 = 0 percent; selected formal proposal to second-backbone Ours, 0/47 = 0 percent. Epoch 5 is not in the formal-method funnel yet because it is still prior reproduction and residual discovery.

## 13. Compute and Operational Audit

Current campaign state records 5.21 GPU hours and 14.845 GiB downloaded for the current autonomous campaign, but repo-wide GPU hours are `NOT_RECORDED`. The state start for the current autonomous campaign is 2026-07-12T16:32+09:00; the full repository campaign spans from first commit 2026-06-27T10:30:15+09:00 to refreshed audit snapshot 2026-07-17T12:45:29+09:00, about 20.1 wall-clock days.

Simulator episodes: formal autonomous method pipeline has 3,380 clearly countable paired closed-loop episodes from final artifacts: 890 Stage A/2a episodes plus 2,490 Stage B/2b episodes. Canonical non-quarantined route-level lower bound is at least 3,604 completed episodes after adding PhaseBarrier valid repair 100, official baseline pilot/smoke 52, OpenVLA/SmolVLA hard-slice diagnostic 40, and the completed Epoch 5 residual Base/Prior diagnostic 32. Including original exploratory PhaseBarrier/CensorCredit attempts gives at least 3,624, but repo-wide exact total is `NOT_RECORDED` because replay branches, invalid attempts, and partial reruns are not globally normalized.

Assets and storage: `C:\assets\data` is about 93.545 GiB, `hf_home` 1.902 GiB, `datasets` 1.862 GiB, `checkpoints` 1.696 GiB, and `repos` 1.028 GiB. WSL model assets include OpenVLA-OFT around 15 GiB and SmolVLA around 865 MiB. Largest known campaign download in state is 14.845 GiB for OpenVLA-related assets.

Operational interruptions: at least one Codex context exhaustion is known because this audit resumes from an exhausted previous thread; exact context compaction/restart count is `NOT_RECORDED`. Approval interruptions are `NOT_RECORDED`. Duplicate or avoidable rerun classes include official LoRA regeneration/drift due missing checkpoints/unpinned RNG, invalid retrained PhaseBarrier repair, COVI invalid v1 plus repair, PCAV expansion name error/resume, VDR self-worker confusion, RAP launcher preflight failure, KITE resume, SPARC capture reset, and two wrong-environment SmolVLA residual attempts that left stale failure JSON before the launcher was corrected to the official SmolVLA env.

Git/commit scale: 805 commits exist across all refs; 801 are ancestors of refreshed audit HEAD. Approximate all-ref churn is dominated by generated report artifacts: `reports` +12,257,382/-31,110 lines across 4,695 file touches in the prior audit's all-ref sample; exact refreshed churn is `NOT_RECORDED` because the current audit run did not regenerate the expensive all-ref diffstat.

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

Epoch 5 corrects one high-impact process failure, namely late/missing external-prior comparison, but it does not remove the scientific gap. The current OpenVLA-OFT residual diagnostic is encouraging because it has the desired Base/Prior residual structure: Base fails meaningfully, OpenVLA improves, and OpenVLA leaves failures. A paper method still needs an upper/teacher/oracle headroom check showing the residual is recoverable, an Ours mechanism, key ablation, clean retention, adequate paired statistics, and second-condition support. None of those Ours-side requirements exists at this audit snapshot.

## 16. False-Negative Audit

Potential false negatives exist, but none should drive immediate continuation. CAVM is the strongest: full 24/58 beat nearest-success 23/58, Base 22/58, and ablation 21/58. Missing evidence is a larger preregistered confirmation and second-backbone result, but current governance allows no third expansion after one 58-pair expansion. CALA and RAR had small Stage 0 margins without closed-loop confidence analysis; missing evidence is a cheap decisive paired validation or uncertainty analysis. DICD and GCAP were underpowered Stage A archives, but later related methods tested richer temporal and goal-conditioned variants.

Reviewer B overreach risk is real mainly for Stage 0 point-estimate stops, not for completed Stage B kills. Treating implementation/data failures as method failures was corrected in later governance; MCI explicitly classifies as implementation failure and not scientific kill. The audit does not recommend a bounded reopen because the best false-negative candidates are either governance-closed (CAVM) or insufficiently novel/anchored (CALA/RAR) compared with resetting candidate selection.

## 17. Paper-Readiness Checklist

Nearest Ours method used for checklist: CAVM-VLA, because it is the strongest valid Ours near-miss. Epoch 5 OpenVLA-OFT is not used as the checklist method because it is a selected external prior, not Ours.

| Requirement | Status | Gap |
|---|---|---|
| Defensible novelty | PARTIAL | memory mechanism plausible but closest prior not officially reproduced |
| SmolVLA Base vs Base + Ours | PARTIAL | full 24/58 vs Base 22/58, too small |
| Closest prior vs Ours | PARTIAL | nearest-success memory proxy 23/58, not official prior |
| Key ablation | PARTIAL | no-contrast 21/58, but effect small |
| Relevant simple control | PARTIAL | nearest-success and Base tested |
| Clean retention | PARTIAL | no paper-grade clean-retention package |
| Adequate paired statistics | MISSING | one-episode gain after allowed expansion |
| Quantized OpenVLA-OFT INT4 + Ours | MISSING | diagnostic OpenVLA exists; current Epoch 5 prior diagnostic is not CAVM/Ours |
| Second claim-specific condition | MISSING | not run |
| Efficiency | PARTIAL | local compute known, no final method efficiency table |
| Reproducibility | PARTIAL | artifacts exist, not a paper package |
| Figure/table-ready artifacts | MISSING | no final package |

Exact gap to `READY_TO_DRAFT_RAL_PAPER_PACKAGE`: no `PROTOTYPE_GO`, no official-prior win, no statistically adequate positive Stage B, no Ours-on-second-backbone result, no second condition, and no figure/table-ready reproducibility package.

## 18. Missed or Unreported Events

Progress likely occurred between user-visible updates because the previous thread exhausted context while the repo kept accumulating durable artifacts. Notable hidden or easy-to-miss events: CSPR and MCI completed immediately before this audit, both as Stage 0 implementation failures; DCCG/MHS/S2C ended as data or cache coverage failures; multiple late stage0-heavy cycles did not reach rollout; COVI v1 was invalid then repaired; PhaseBarrier had an invalid retrained positive result quarantined and a valid bounded repair that killed the component; CensorCredit looked positive until the label/head identity postmortem; EAC had a promising Stage A but failed/tied in Stage B; CAVM remained the strongest near-miss but is governance-closed after one expansion.

Automatic pivots included FEDO and GCAP in the initial autonomous set and many epoch 4 candidate selections. Branch audit found `codex/execspec-repair-state0-state1` with one unique historical ExecSpec state1 kill-gate commit absent from HEAD, but later ExecSpec state2/3/3.5 artifacts in current history supersede it.

Events after the first full-history audit that may be easy to miss:

- The prior audit itself was committed as `b0ecb6e` and recommended `RESET_CANDIDATE_SELECTION_STRATEGY`.
- The branch then moved to `codex/epoch5-official-prior-first`, so this refreshed audit stays on the current branch rather than switching.
- Epoch 5 selected OpenVLA-OFT before any Ours design and explicitly superseded ordinary Cycle 39 method search.
- The recovered OpenVLA hard-slice evidence is prior-positive but saturated: OpenVLA-OFT INT4 20/20 versus matched SmolVLA Base 11/20.
- A residual diagnostic was preregistered at `ffb55f5`; the selected condition is LIBERO-10 task 8/task 9, resets `20260716..20260723`.
- OpenVLA-OFT residual execution completed after preregistration: 14/16, with residual failures on task 8 reset identities `20260721` and `20260722`.
- The first SmolVLA residual attempts failed because the launcher used the OpenVLA environment; those are environment failures, not Base scientific results.
- The SmolVLA residual worker was active when this audit began and then completed cleanly during the audit: 7/16, exit code 0, using the official SmolVLA env.
- Local uncommitted work exists: a SmolVLA import-compatibility shim, an Epoch 5 residual launcher, and 16 OpenVLA rollout videos.
- This refreshed audit report is the only user-facing report created by the Phase A audit run.

## 19. Recommended Strategic Decision

Recommendation: `CONTINUE_CURRENT_CYCLE`.

Justification: the earlier audit's `RESET_CANDIDATE_SELECTION_STRATEGY` recommendation has already been acted on by moving to Epoch 5 official-prior-first. The current cycle is not another cached-feature Ours method; it is a selected external-prior diagnostic with frozen matched Base/Prior identities. The matched residual diagnostic completed during this audit and found OpenVLA-OFT INT4 14/16 versus SmolVLA Base 7/16. Continuing is justified only in the narrow sense of recording that result in the existing Epoch 5 reports and running or preregistering the smallest upper/teacher/oracle headroom check. It does not authorize Ours design, new training, a new method candidate, or a fallback ecosystem until recoverable headroom is adjudicated.

## 20. Exact Resume Plan

`Exact Next Codex Prompt After User Review`

```text
Resume the autonomous VLA research campaign in C:\Users\jiheo\tca_map after the full-history audit.

Branch: codex/epoch5-official-prior-first
Last scientific HEAD before refreshed audit report: ffb55f57e1cd978a35d0d84c9ede487559e02fa0
Current scientific state: Epoch 5 official-prior-first residual diagnostic complete for matched Base/Prior; upper/headroom pending
Previous method: MCI-VLA
Previous decision: MCI_STAGE_0_IMPLEMENTATION_FAILURE
Selected audit recommendation: CONTINUE_CURRENT_CYCLE

Exact next scientific action:
Resume only by recording the completed `epoch5_libero10_residual_v1` matched diagnostic in the existing Epoch 5 reports and compact handoff: OpenVLA-OFT INT4 14/16, SmolVLA Base 7/16, matched manifests, no infrastructure failures, Base/Prior residual structure present. Then run or preregister the smallest upper/teacher/oracle headroom check for the residual condition before any Ours design. If upper/headroom fails, classify the condition as not recoverable or too severe under the preregistered rules. If upper/headroom passes, only then proceed to Ours candidate design under official-prior-first governance.

Prohibited repeats:
Do not rescue or retune MCI-VLA or CSPR-VLA. Do not generate an Ours method, train, download, or launch a new candidate before upper/headroom evidence is complete. Do not treat OpenVLA-OFT INT4 diagnostic success as an Ours result. Do not reinterpret the wrong-env SmolVLA attempts as Base scientific results. Do not switch branches, stash, reset, or clean the uncommitted scientific work without explicit user approval.

Time-to-evidence requirement:
The current time-to-evidence target is a single upper/headroom answer. Preserve the matched Base/Prior table, worker PID, heartbeat, exit code, stderr/stdout summaries, manifest SHA-256s, per-task success counts, and stale/invalid attempts separately. If the headroom check fails from environment/runtime causes, classify it as infrastructure-blocked and repair only the runner/env within the existing preregistered boundary.

LoRA role:
Use LoRA/QLoRA only as low-compute parameterization after an actual Ours mechanism is authorized. For the current step, no LoRA training is authorized; this is Base/Prior/upper evidence collection only. If a later Ours mechanism uses adapters, keep the scientific mechanism separable from PEFT and include standard LoRA only when it is a real alternative explanation or shared scaffold control.

Reviewer false-negative safeguards:
For the current diagnostic, do not call the prior condition dead from stale files, wrong-env failures, or tiny point estimates without exact paired manifests. Classify outcomes as `RESIDUAL_FOUND_PRIOR_POSITIVE`, `PRIOR_SATURATED_NEXT_CONDITION`, `PRIOR_NOT_POSITIVE_ON_CONDITION`, or `INFRASTRUCTURE_BLOCKED` as preregistered. Preserve the strongest fair interpretation and the narrowest publishable claim.

Conditions for implementation and rollout:
Ours implementation may start only after selected-prior evidence satisfies Base fails -> Prior improves -> residual remains -> upper/headroom says recoverable. Closed-loop Ours rollout may start only after source, runner, focused tests, exact paired manifest, checkpoint reload, finite gradients when applicable, bounded action deltas, full-vs-ablation difference, clean retention, no privileged inference signal, prior fairness, resource risk, and frozen decision thresholds are documented.
```
