# Next Actions

## Immediate Autonomous Behavior

Codex should self-check current state instead of asking the user to confirm it.

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\14_plan_smolvla_download.ps1
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\13_check_smolvla_adapter_smoke.ps1
```

If checkpoint files are missing, report the missing file classes and stop at the checkpoint-file gate.

If checkpoint files are present, verify readiness and continue through the risk-assessed bounded SmolVLA pilot path. Load-only construction has passed; the next bounded step is selected by risk assessment. Do not perform rollout, simulator execution, real benchmark evaluation, token access, OpenVLA-OFT execution, or work outside the SmolVLA autonomous pilot budget unless the risk assessment says proceed.

The planning command is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\15_plan_smolvla_load_only_smoke.ps1
```

The bounded execution scaffold is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\16_smolvla_load_only_smoke.ps1
```

It will not load a model without `ALLOW_HEAVY_IMPORT=1`. Runtime dependencies are now installed, and the bounded load-only smoke has passed on CPU. `ALLOW_HEAVY_IMPORT=1` may be set by Codex only inside risk-assessed bounded SmolVLA tasks.

Check runtime dependency readiness:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\17_check_smolvla_runtime_deps.ps1
```

Run the bounded single-sample interface smoke only after load-only passes:

```powershell
$env:ALLOW_HEAVY_IMPORT="1"
$env:ALLOW_SINGLE_SAMPLE_INFERENCE="1"
powershell -ExecutionPolicy Bypass -File scripts\28_smolvla_single_sample_interface_smoke.ps1
Remove-Item Env:\ALLOW_SINGLE_SAMPLE_INFERENCE -ErrorAction SilentlyContinue
Remove-Item Env:\ALLOW_HEAVY_IMPORT -ErrorAction SilentlyContinue
```

Recheck the completed runtime install state without importing heavy VLA models:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\18_plan_smolvla_runtime_install.ps1
```

Plan or validate the dummy feature-cache interface without SmolVLA imports:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\19_plan_feature_cache.ps1
```

Run the eval-only cached-feature smoke with dummy cached features:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\25_eval_feature_cache_smoke.ps1 -PrepareDummyCache
```

Plan the tiny head-only pilot risk boundary without training:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\26_plan_tiny_head_only_pilot.ps1
```

Run the bounded tiny head-only smoke runner:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\29_tiny_head_only_smoke.ps1 -PrepareDummyCache
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

This is a tiny CPU head-only smoke on cached/dummy features. It is not a paper-grade result.

Summarize the current risk-gate choices:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\27_summarize_hard_stop_status.ps1
```

Generate the go/no-go status summary:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\31_generate_go_no_go_report.ps1
```

## Current Asset State

The approved SmolVLA checkpoint source has been acquired:

```text
lerobot/smolvla_base
```

Target directory:

```text
C:\assets\checkpoints\smolvla
```

Cache directory:

```text
C:\assets\hf_home
```

Detected groups:

- config: present (`config.json`),
- weights: present (`model.safetensors` and processor safetensors),
- tokenizer/processor dependency: present under `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`.

The dependency was acquired with full model weights avoided. Retained files include tokenizer JSON/config, vocab/merges, special tokens, chat template, preprocessor config, processor config, and config.

Bounded load-only status:

- passed on CPU,
- local checkpoint loaded,
- `load_vlm_weights=false`,
- no model inference,
- no training,
- no rollout,
- no OpenVLA-OFT execution,
- no downloads.

Verify files without loading the model:

```powershell
Test-Path C:\assets\checkpoints\smolvla
Get-ChildItem C:\assets\checkpoints\smolvla
Test-Path C:\assets\checkpoints\smolvla\config.json
Get-ChildItem C:\assets\checkpoints\smolvla -Filter *.safetensors
Get-ChildItem C:\assets\checkpoints\smolvla -Filter *.bin
```

## Expected Order

1. Manual SmolVLA acquisition checklist.
2. Readiness recheck.
3. Load-only adapter smoke planning.
4. Runtime dependency install completed under the earlier install gate. Done.
5. Run risk-assessed SmolVLA load-only heavy import/model construction. Done.
6. Create or run single-sample SmolVLA interface smoke with synthetic or dummy inputs. Done.
7. Feature cache interface validation with dummy cached features. Done.
8. Eval-only cached-feature head/metric smoke. Done.
9. Tiny head-only pilot planning and budget check. Done.
10. Tiny head-only smoke runner with bounded steps, no rollout, no OpenVLA-OFT, no paper claim, runtime<=30 minutes, and VRAM<=14GB. Done.
11. Summarize risk-gate choices. Done.
12. Generate go/no-go status summary. Done.
13. Required LoRA adapter construction plan. Done.
14. Required LoRA tiny smoke scaffold. Done.
15. Required TCA-Map + LoRA comparison plan. Done.
16. QLoRA feasibility check. Done.
17. Update LoRA/QLoRA go/no-go status. Done.
18. Bounded local pilot execution is risk-assessed autonomous if inside limits. Done: head-only ActionMap vs TCA-Map comparison report.
19. Tiny LoRA smoke runner inside risk-assessed budget. Done.
20. Tiny LoRA comparison report. Done.
21. Consolidated local pilot status/report inside risk-assessed budget. Done.
22. Approval-based hard-stops replaced by risk-assessed autonomous execution. Done.
23. LIBERO/LIBERO-CF-style dataset readiness/tiny subset risk planner. Done.
24. Simulator readiness planning without simulator import/render/rollout. Done.
25. Local pilot budget alignment to 300 steps. Done.
26. Bounded cached-feature local pilot extension inside the 300-step policy. Done.
27. Consolidated status/go-no-go summaries include the bounded extension report. Done.
28. LIBERO/RoboSuite official source resolution. Done.
29. Bounded source repo setup. Done: official LIBERO and RoboSuite source repos are path-ready.
30. Official LIBERO data acquisition. Done locally under `C:\assets\data\libero`; dataset files are not committed.
31. HDF5 reader dependency check. Done: `h5py>=3.11` was installed after a green dependency risk assessment, and `scripts\50_check_libero_hdf5_reader.ps1` reports ready.
32. LIBERO offline interface smoke gate. Done: `scripts\48_plan_libero_offline_interface_smoke.ps1` reports `ready_for_offline_interface_smoke=true` on local HDF5 files.
33. Current gate: tiny counterfactual split construction from acquired LIBERO metadata/HDF5 inventory with `scripts\51_build_libero_offline_counterfactual_split.ps1`.
34. Tiny real/offline ActionMap vs TCA-Map comparison with `scripts\52_compare_libero_offline_actionmap_tca.ps1`. Done if the ignored report says `libero_offline_head_comparison_passed=true`.
35. Required tiny real/offline LoRA comparison with `scripts\53_compare_libero_offline_lora.ps1`. Done if the ignored report says `libero_offline_lora_comparison_passed=true`. This is bounded NumPy adapter training only and requires `ALLOW_TINY_TRAINING=1`.
36. Bounded local pilot report with `scripts\54_generate_libero_offline_bounded_pilot_report.ps1`. Done if the ignored report says `libero_offline_bounded_pilot_report_passed=true`.
37. WSL simulator dependency ladder setup. Done: `scripts\57_setup_wsl_simulator_deps.ps1` created or reused the WSL venv at `$HOME/.venvs/tca_map_sim` and installed the bounded import-readiness dependencies without sudo or apt.
38. Bounded simulator source-link helper. Done: `scripts\60_link_wsl_simulator_sources.ps1` reuses `/home/jiheon/.venvs/tca_map_sim`, links local RoboSuite/LIBERO sources offline, writes the LIBERO `.pth` entry, and creates noninteractive WSL `~/.libero/config.yaml`.
39. Bounded simulator import-only smoke. Done: `scripts\55_bounded_simulator_import_smoke.ps1` imported `robosuite` and `libero` through the selected WSL venv. This is import-only readiness, not render evidence, rollout evidence, or paper-grade evidence.
40. Bounded simulator render/reset-step risk planning. Done: `scripts\58_plan_simulator_render_reset.ps1` now reports `ready_for_bounded_render_smoke_plan=true` and `ready_for_bounded_reset_step_smoke_plan=true` after passed import-only and render-smoke reports.
41. Bounded simulator render-smoke branch. Done: `scripts\59_bounded_simulator_render_smoke.ps1` passed one tiny MuJoCo 64x64 OSMesa render under the task-local gate. It did not create/reset/step LIBERO/RoboSuite environments, rollout, train, use GPU, download, execute OpenVLA-OFT, or make paper claims.
42. Bounded simulator reset/step smoke. Done: `scripts\61_bounded_simulator_reset_step_smoke.ps1` performed a tiny in-memory MuJoCo reset plus 3-step smoke through the selected WSL venv. It did not create LIBERO/RoboSuite environments, rollout, train, use GPU, download, execute OpenVLA-OFT, or make paper claims.
43. Tiny diagnostic rollout risk assessment. Done: `scripts\62_plan_tiny_diagnostic_rollout.ps1` reports a bounded planning envelope and authorizes execution only through task-local `ALLOW_TINY_ROLLOUT=1`.
44. Bounded tiny diagnostic rollout. Done: `scripts\63_bounded_tiny_diagnostic_rollout.ps1` passed 5 toy MuJoCo diagnostic tasks, 1 episode each, 5 steps each. This is simulator plumbing evidence only, not LIBERO/RoboSuite benchmark rollout evidence and not paper-grade evidence.
45. Bounded LIBERO/RoboSuite diagnostic rollout planning. Done: `scripts\64_plan_libero_robosuite_diagnostic_rollout.ps1` reports a bounded planning envelope for local official LIBERO/RoboSuite paths and data.
46. Bounded LIBERO/RoboSuite zero-action diagnostic rollout. Done: `scripts\65_bounded_libero_robosuite_diagnostic_rollout.ps1` passed 1 `libero_10` task with 3 zero-action steps, created/reset/stepped a real LIBERO/RoboSuite environment, and observed a finite 64x64 image. This is diagnostic simulator plumbing evidence only, not standard success, not benchmark success, and not paper-grade evidence.
47. Current gate: learned-policy LIBERO rollout readiness planning with `scripts\66_plan_libero_policy_rollout_readiness.ps1`. This is planning-only and decides whether the WSL-only simulator+SmolVLA topology is ready. If it says `proceed`, create a separately gated tiny learned-policy rollout runner. If it says `reduce_scope`, prepare WSL SmolVLA runtime readiness first. Stop only for ambiguous/out-of-budget risk, OpenVLA-OFT, token/secret/payment/license, credentialed/system-driver/license-gated changes, external upload/submission, multi-seed before a separate risk budget, or unsupported paper claims.
48. WSL SmolVLA runtime setup/readiness. Done: `scripts\67_plan_wsl_smolvla_runtime_setup.ps1` and `scripts\68_setup_wsl_smolvla_runtime_deps.ps1` prepared `/home/jiheon/.venvs/tca_map_sim` with CPU torch/torchvision plus SmolVLA runtime modules after a green risk assessment. The first setup attempt hit the 1800 second timeout, but a follow-up probe found all modules present and the second guard run reported setup complete without further installs. The later single-action smoke revealed and fixed additional LeRobot import/runtime dependencies: `draccus`, `datasets`, `imageio[ffmpeg]`, `diffusers`, `pyserial`, `deepdiff`, `av`, and `einops`.
49. WSL SmolVLA single-action smoke. Done: `scripts\69_plan_wsl_smolvla_single_action_smoke.ps1` planned a CPU synthetic action smoke, and `scripts\70_bounded_wsl_smolvla_single_action_smoke.ps1` passed under task-local `ALLOW_WSL_SMOLVLA_SINGLE_ACTION=1`. It loaded local SmolVLA in WSL, ran one synthetic `select_action`, produced a finite `[1, 6]` action, and did not run simulator rollout, training, GPU jobs, OpenVLA-OFT, downloads during the smoke, token access, or paper claims.
50. Current gate: create a separately gated tiny learned-policy LIBERO rollout runner. It must run at most 1 task initially, at most 5-10 steps, no training, no GPU by default, no OpenVLA-OFT, no multi-seed, no unsupported benchmark or paper claim, and must label output as bounded local pilot/benchmark diagnostic only.

Current status: the bounded tiny head-only smoke, ActionMap vs TCA-Map head-only comparison report, tiny LoRA smoke runner, tiny LoRA comparison report, consolidated local pilot status, go/no-go summary, required LoRA adapter construction plan, LoRA tiny-smoke scaffold, TCA-Map + LoRA comparison plan, QLoRA feasibility check, LoRA/QLoRA go/no-go update, LIBERO dataset risk planner, simulator readiness planner, 300-step local pilot budget alignment, bounded cached-feature local pilot extension, bounded-extension status consolidation, official LIBERO/RoboSuite source resolution, bounded source repo setup, toy MuJoCo diagnostic rollout, bounded LIBERO/RoboSuite zero-action diagnostic rollout, WSL SmolVLA runtime readiness, and WSL SmolVLA single-action smoke have passed. LoRA/QLoRA are required experimental tracks after the head-only path, but not the main novelty. The repository now uses end-to-end risk-assessed autonomous execution instead of broad approval-based hard-stops. Downloads, GPU tasks, bounded training, real dataset setup, WSL simulator dependency setup, simulator readiness, bounded diagnostic rollout, tiny learned-policy rollout, and bounded benchmark rollout should proceed automatically if a risk assessment is green and should stop only if risk is ambiguous, outside budget, external/irreversible, credentialed/system-driver/license-gated, OpenVLA-OFT-related, multi-seed outside budget, or would make an unsupported claim.

Planning command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\32_plan_lora_adapter_construction.ps1
powershell -ExecutionPolicy Bypass -File scripts\33_plan_lora_tiny_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\34_plan_lora_comparison.ps1
powershell -ExecutionPolicy Bypass -File scripts\35_check_qlora_feasibility.ps1
powershell -ExecutionPolicy Bypass -File scripts\36_compare_head_only_tiny_pilot.ps1
$env:ALLOW_TINY_TRAINING="1"; powershell -ExecutionPolicy Bypass -File scripts\37_tiny_lora_smoke.ps1 -PrepareDummyCache; Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
powershell -ExecutionPolicy Bypass -File scripts\38_compare_tiny_lora_pilot.ps1
powershell -ExecutionPolicy Bypass -File scripts\39_generate_local_pilot_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\45_resolve_libero_robosuite_sources.ps1
$env:ALLOW_DOWNLOADS="1"; powershell -ExecutionPolicy Bypass -File scripts\46_prepare_libero_robosuite_sources.ps1; Remove-Item Env:\ALLOW_DOWNLOADS -ErrorAction SilentlyContinue
powershell -ExecutionPolicy Bypass -File scripts\42_plan_libero_dataset_risk.ps1
powershell -ExecutionPolicy Bypass -File scripts\43_plan_simulator_readiness.ps1
powershell -ExecutionPolicy Bypass -File scripts\66_plan_libero_policy_rollout_readiness.ps1
powershell -ExecutionPolicy Bypass -File scripts\67_plan_wsl_smolvla_runtime_setup.ps1
powershell -ExecutionPolicy Bypass -File scripts\41_risk_assess_task.ps1 -Task "Bounded simulator import smoke" -Category simulator -Source "official local LIBERO and RoboSuite source checkouts" -TargetPath "C:\assets\repos" -ExpectedSizeGb 0 -ExpectedRuntimeMinutes 2 -ExpectedRamGb 2 -ExpectedVramGb 0 -SimulatorInstalled -OfficialSource
$env:ALLOW_SIMULATOR_IMPORT_SMOKE="1"; powershell -ExecutionPolicy Bypass -File scripts\55_bounded_simulator_import_smoke.ps1; Remove-Item Env:\ALLOW_SIMULATOR_IMPORT_SMOKE -ErrorAction SilentlyContinue
```

## Required LoRA/QLoRA Progression

A. SmolVLA load-only smoke. Done.
B. Single-sample interface smoke. Done.
C. Frozen/head-only TCA-Map tiny pilot. Done as bounded smoke.
D. Required LoRA adapter construction plan. Done.
E. Required LoRA tiny smoke scaffold. Done; bounded tiny execution runner and comparison report exist.
F. Required TCA-Map + LoRA comparison plan. Done.
G. QLoRA feasibility check. Done.
H. Go/no-go report. Done.

LoRA/QLoRA are required adaptation tracks, not optional nice-to-have items. Full fine-tuning remains forbidden locally.

## Self-Check Cases

Case A: SmolVLA checkpoint path missing or not configured. Codex reports exact missing path/config, updates state/action docs if needed, and stops at asset path gate.

Case B: SmolVLA path exists but config/tokenizer dependency/weights are missing. Codex reports exact missing file classes, updates state/action docs if needed, and stops at checkpoint-file gate.

Case C: Config/tokenizer dependency/weights and runtime dependencies are present and adapter-smoke-ready. Codex updates state/action docs if needed and continues to the next risk-assessed bounded SmolVLA pilot task. Since load-only construction, single-sample interface smoke, and dummy feature-cache/interface validation have passed, the next task is selected by risk assessment.

Case D: Checker fails due to Windows/PATH/tooling. Codex diagnoses and fixes minimally on a new branch if safe, then validates again.

Case E: Risk gate reached. Codex runs risk assessment. If the task is inside budget, proceed; if outside/ambiguous/external irreversible/OpenVLA/paper-claim-related, stop and report the blocker.

## Risk-Assessed Next Gates

Codex must not ask for routine approval when risk can be checked automatically. Before downloads, GPU, training, dataset setup, simulator readiness, or rollout, write/print a risk assessment with task, source, expected size, target path, disk free before/after estimate, expected runtime, expected RAM/VRAM, budget, source status, token/license/payment status, decision, and reason.

Structured helper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\41_risk_assess_task.ps1 -Task "next concrete task" -Category "generic"
```

Proceed automatically if inside budget. Stop and report if risk is ambiguous or outside budget.

Default autonomous budgets:

- download <=80GB per task and keep >=100GB free disk, except the official LIBERO dataset source `yifengzhu-hf/LIBERO-datasets`, which may use <=180GB only if >=250GB remains after acquisition,
- GPU VRAM <=14GB, runtime <=30 minutes, batch size 1,
- SmolVLA-only local training, frozen backbone or LoRA/QLoRA adapter only, max 300 steps after smaller smoke is stable,
- dataset setup only from official/documented unambiguous sources without token/login/payment/license click-through,
- simulator readiness/import-render smoke only after WSL dependency/import readiness is green and <=10 minutes,
- minimal WSL Python packaging setup after a green WSL simulator dependency ladder risk assessment,
- bounded rollout only after readiness smoke, task count <=5 for the first local benchmark rung, runtime <=30 minutes, no OpenVLA-OFT, no unsupported paper claim.

Always stop before:

- OpenVLA-OFT execution until a separate risk budget exists,
- token/secret/API key access,
- paid services,
- license click-through,
- external upload/submission/publishing,
- deleting user files outside approved cache/repo cleanup,
- system-wide CUDA/PyTorch/driver changes,
- credentialed/system-driver/license-gated system setup,
- unsupported paper-level empirical claims.

Minimal WSL Python packaging setup is standing-approved after risk assessment; a sudo password prompt remains a hard stop.

Codex should not ask routine questions such as whether files were placed, whether readiness should be checked, whether pytest should run, which branch is current, whether git is clean, or what is missing. It should inspect and report.

## Readiness Target

Proceed to a new load-only adapter smoke task only if:

```text
ready_for_smolvla_path_check=true
smolvla_checkpoint_files_present=true
ready_for_smolvla_adapter_smoke=true
```

## Later Task

After readiness, planning, load-only smoke, single-sample interface smoke, and feature-cache/interface validation are true, continue on a new branch for a tiny head-only smoke runner. That branch may run only bounded head-only smoke and must not train a backbone, rollout, evaluate real datasets, or execute OpenVLA-OFT.

After the LoRA/QLoRA go/no-go update, tiny LoRA smoke runner, tiny LoRA comparison report, consolidated local pilot status report, risk-assessed policy update, LIBERO dataset risk planner, simulator readiness planner, 300-step budget alignment, bounded local pilot extension, bounded-extension status consolidation, official source resolution, source repo setup, metadata-only subset construction, official LIBERO data acquisition, h5py reader readiness, offline interface smoke gate, counterfactual split construction, offline ActionMap/TCA-Map comparison, required offline LoRA comparison, bounded pilot report, simulator readiness status integration, WSL simulator dependency setup, local WSL source linking, bounded simulator import-only smoke, bounded render/reset-step risk planning, bounded render-smoke execution, bounded reset/step smoke, tiny diagnostic rollout risk planning, bounded tiny diagnostic rollout execution, and bounded LIBERO/RoboSuite zero-action diagnostic rollout execution, continue to learned-policy LIBERO rollout readiness planning. Stop only for multi-seed rollout before a separate risk budget, OpenVLA-OFT, token/secret/payment/license, credentialed/system-driver/license-gated changes, external irreversible actions, or unsupported paper claims.

After WSL SmolVLA runtime readiness and single-action smoke passed, the separately gated tiny learned-policy LIBERO rollout runner has also passed one bounded diagnostic run. Current next actions:

1. Generate a tiny learned-policy benchmark-metric diagnostic summary from the bounded rollout JSON without claiming benchmark success.
   Done.
2. Add a small learned-policy rollout matrix planner capped by the current risk budget.
   Done; current decision is `reduce_scope`, not multi-task proceed.
3. Create a separately gated one-task longer diagnostic runner with task count 1 and max steps 10.
   Done; current result is execution-pass but task-success false and reward 0.0.
4. Generate a reduced-scope rollout metric summary.
   Done; current result is diagnostic success rate 0.0, reward 0.0, action max abs about 0.793, gripper component 0.0.
5. Inspect likely action-interface causes before scaling: action normalization, gripper dimension, observation/state mapping, language prompt, and camera mapping.
   Planning done; high-priority risks are action dimension/gripper mapping, action normalization/scale, and observation state mapping.
6. Create a bounded metadata/report-only action-interface audit.
   Done; high-priority findings are action dim mismatch, gripper zero padding, state truncation risk, and nontrivial actions with zero reward.
7. Create a zero-action versus SmolVLA-action diagnostic comparison.
   Done as report-only comparison. Current result: zero-action and SmolVLA-action both have diagnostic success `false` and reward `0.0`; SmolVLA actions are nontrivial but do not outperform zero-action.
8. Create an explicit action/state adapter patch plan.
   Done as planning-only. Next: implement pure action/state/image adapter helpers with unit tests; do not run rollout until adapter tests and single-sample/interface smoke pass.
9. Implement pure action/state/image adapter helpers with unit tests.
   Done; pure helper tests pass and rollout behavior is not wired yet.
10. Wire adapter metadata into synthetic/single-sample interface smoke without rollout.
    Done; bounded synthetic single-sample smoke passed with `adapter_metadata_recorded=true` and no simulator or rollout.
11. Add a rollout bridge adapter wiring planner.
    Done; planner passed and keeps rollout execution blocked.
12. Wire pure action/state/image adapters into the learned-policy rollout bridge.
    Done in code/tests only; rollout execution remains a separate bounded diagnostic gate.
13. Rerun a bounded learned-policy diagnostic rollout with explicit adapter metadata only after the rollout gate is green.
    Done; execution passed, explicit adapter metadata is present, but diagnostic success and reward remain 0.0.
14. Run adapter-strategy and action-scale diagnostics before rollout scaling.
    Done for the first gripper-strategy runner. `scripts\83_bounded_adapter_strategy_diagnostic.ps1` executed zero-hold, open, and close variants for one task and at most 10 steps each. All variants passed the wrapper but produced diagnostic success rate 0.0 and reward sum 0.0.
15. Keep evidence labels as diagnostic or local pilot until baselines, ablations, and repeated benchmark protocol are implemented.
16. Stop before multi-seed rollout, paper-grade claims, OpenVLA-OFT, full fine-tuning, external upload, token/secret access, payment/license click-through, or destructive/system-level changes.
17. Bounded action-scale diagnostic.
    Done for scales `0.25`, `0.5`, and `1.0` under the zero-hold gripper strategy. All variants passed the wrapper and scaled action magnitude as expected, but diagnostic success rate and reward sum remained 0.0.
18. Bounded prompt-format diagnostic.
    Done for `stem_spaces`, `bddl_language`, and `bddl_language_period`. All variants passed the wrapper and changed action previews, but diagnostic success rate and reward sum remained 0.0.
19. Bounded camera-source diagnostic.
    Done for `current_aliases`, `camera3_eye_in_hand`, and `all_agentview`. All variants passed the wrapper and changed image-source metadata/action previews, but diagnostic success rate and reward sum remained 0.0.
20. Bounded state-sufficiency diagnostic.
    Done for `eef_pos_quat_first3`, `eef_pos_quat_last3`, and `eef_pos_zero_rot`. All variants passed the wrapper and changed explicit state metadata/action previews, but diagnostic success rate and reward sum remained 0.0.
21. Learned-policy diagnostic synthesis/no-go report.
    Done. The synthesis decision is `no_go_rollout_scaling`: the diagnostic ladder is complete, but no axis produced nonzero reward or diagnostic success, and every source report keeps `ready_for_rollout_scaling=false`.
22. Bounded environment-policy compatibility audit.
    Done. The audit decision is `no_go_rollout_scaling` with high-severity blockers in task/checkpoint alignment, `load_vlm_weights=false` diagnostic loading, 6D policy action versus 7D environment action convention, and repeated zero-reward diagnostic evidence.
23. Bounded offline LIBERO HDF5 demonstration interface audit.
    Done. The audit confirms 7D demonstration actions versus 6D policy actions, 6D `obs/ee_states` matching policy state, two HDF5 RGB streams versus three policy image inputs, and 128x128 HDF5 images versus 256x256 policy image inputs.
24. Report-only offline adapter reproduction check.
    Done. The first demonstration action is best reproduced by `policy_6d_delta_pose_plus_gripper_close`, while the current zero-hold gripper default mismatches the first demonstration gripper value `-1.0`.
25. Next safe research-engineering step: plan a bounded one-task gripper-close compatibility diagnostic. Run it only as a specific compatibility hypothesis, with no downloads, no training, no GPU job, no OpenVLA-OFT, no multi-seed, no rollout scaling, and no paper claims.
26. Gripper-close compatibility diagnostic planning command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\96_plan_gripper_close_compat_diagnostic.ps1
```

If the planner reports `decision=proceed`, a future runner may test exactly one gripper-close compatibility diagnostic under a task-local gate. If it reports `decision=reduce_scope`, do not rerun an identical close-strategy rollout; instead plan a narrower HDF5-aligned task/initial-state/action-sign compatibility check. In all cases, rollout scaling and paper claims remain blocked.
27. HDF5-to-rollout alignment audit command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\97_audit_hdf5_rollout_alignment.ps1
```

If this audit reports `ready_for_hdf5_initial_state_replay_plan=true`, the next safe task is a planning-only HDF5 initial-state or first-action replay diagnostic. Do not run another learned-policy rollout from the same reset-only setup until the HDF5 initial-state convention has been checked.
28. HDF5 initial-state replay planning command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\98_plan_hdf5_initial_state_replay.ps1
```

If this planner reports `ready_for_bounded_hdf5_replay_runner=true`, the next safe implementation is a separately gated replay runner with `ALLOW_HDF5_REPLAY_DIAGNOSTIC=1`. The first runner should set one HDF5 demo initial state and replay only the first demonstration action, with no learned-policy inference, no training, no GPU job, no OpenVLA-OFT, and no paper claim.
29. Bounded HDF5 first-action replay command:

```powershell
$env:ALLOW_HDF5_REPLAY_DIAGNOSTIC="1"
powershell -ExecutionPolicy Bypass -File scripts\100_bounded_hdf5_initial_state_replay.ps1
Remove-Item Env:\ALLOW_HDF5_REPLAY_DIAGNOSTIC -ErrorAction SilentlyContinue
```

If this runner passes, it only proves that the HDF5 initial-state/action replay convention is operational. It does not prove learned-policy success. The next safe learned-policy task should be a narrow rollout recheck using a documented initial-state convention, not rollout scaling or paper claims.

30. Init-state learned-policy recheck planning command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\101_plan_init_state_learned_policy_recheck.ps1
```

If this planner reports `ready_for_bounded_init_state_learned_policy_recheck_runner=true`, the next safe implementation is a separately gated runner using `ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK=1`. It must use one HDF5 demonstration initial state, one task, at most five policy-controlled steps, WSL CPU by default, no downloads, no installs, no training, no GPU job, no OpenVLA-OFT, no multi-seed evaluation, and no benchmark/SOTA/paper-grade claim.

31. Bounded init-state learned-policy recheck command:

```powershell
$env:ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK="1"
powershell -ExecutionPolicy Bypass -File scripts\102_bounded_init_state_learned_policy_recheck.ps1
Remove-Item Env:\ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK -ErrorAction SilentlyContinue
```

Current local result: passed as diagnostic execution evidence only. It loaded local SmolVLA in WSL CPU, set the local HDF5 demo `init_state`, ran one `libero_10` task for 3 policy-controlled steps with the gripper-close adapter, and kept downloads, installs, training, GPU jobs, OpenVLA-OFT, multi-seed, benchmark claims, and paper claims false. Task success remained `false` and reward sum remained `0.0`.

Next safe step: generate a report-only metric summary comparing this init-state recheck against previous reset-only learned-policy diagnostics. Do not scale rollout or make benchmark/paper claims.

32. Init-state recheck metric summary command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\103_generate_init_state_recheck_metric_summary.ps1
```

Current local result: summary passed with `decision=no_go_rollout_scaling`. Reset-only 3-step, reset-only 10-step, and HDF5-init-state 3-step diagnostics all passed their wrappers, but all had diagnostic success `false` and reward sum `0.0`. The HDF5-init-state run correctly set the demonstration initial state and used the gripper-close adapter, but it did not improve reward or task success.

Next safe step: stop rollout scaling and inspect checkpoint/task alignment, VLM loading policy, and offline demonstration-conditioned action decoding before more learned-policy rollouts.

33. SmolVLA/LIBERO checkpoint-task alignment audit command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\104_audit_smolvla_libero_checkpoint_task_alignment.ps1
```

Expected local result: report-only audit with `decision=no_go_rollout_scaling` and `ready_for_offline_demonstration_conditioned_action_decoding_plan=true`. If this passes, the next safe task is a planning-only offline demonstration-conditioned action-decoding gate. That future gate should use one HDF5 observation and one expert action target, must not create a simulator environment or rollout, and may authorize a later one-sample CPU SmolVLA inference only after a green risk assessment.

34. Offline demonstration-conditioned action decoding planning command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\105_plan_offline_demo_conditioned_action_decoding.ps1
```

Expected local result: `decision=proceed` and `ready_for_bounded_offline_demo_action_decoding_runner=true` if local checkpoint files, the selected HDF5 file, and the checkpoint-task alignment audit are all present. This planner does not load SmolVLA or run inference. If it passes, the next safe implementation is a separately gated one-sample offline action-decoding runner; it must not create a simulator environment, rollout, train, download, use OpenVLA-OFT, or make paper claims.

35. Bounded offline demonstration action decoding command:

```powershell
$env:ALLOW_OFFLINE_DEMO_ACTION_DECODING="1"
powershell -ExecutionPolicy Bypass -File scripts\106_bounded_offline_demo_action_decoding.ps1
Remove-Item Env:\ALLOW_OFFLINE_DEMO_ACTION_DECODING -ErrorAction SilentlyContinue
```

This runner may load local SmolVLA on CPU and perform exactly one offline `select_action` call on one local HDF5 observation/action pair. It must not create a simulator environment, rollout, train, download, use GPU jobs, execute OpenVLA-OFT, or make paper claims. After it runs, summarize the diagnostic before deciding whether any further rollout is justified.

36. Offline demonstration action decoding summary command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\107_summarize_offline_demo_action_decoding.ps1
```

If the summary reports `offline_alignment_signal=weak`, keep learned-policy rollout scaling blocked and inspect VLM loading policy, checkpoint provenance, and action normalization before another learned-policy rollout. If it reports moderate or strong alignment, it is still diagnostic-only; plan a tiny repeated offline decoding check before rollout.

37. VLM loading policy and action-normalization audit command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\108_plan_vlm_loading_policy_action_normalization_audit.ps1
```

Current expected interpretation: if the audit reports `decision=no_go_rollout_scaling` and `ready_for_repeated_offline_decoding_plan=true`, do not run another learned-policy rollout yet. Plan a tiny repeated offline demonstration action-decoding diagnostic over a few HDF5 timesteps, explicitly logging `load_vlm_weights`, action unnormalization, clipping, gripper strategy, and image aliases. Treat VLM-enabled loading or full SmolVLM2 weight acquisition as a separate risk-assessed task.

38. Repeated offline demonstration action-decoding planning command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\109_plan_repeated_offline_demo_action_decoding.ps1
```

If the planner reports `decision=proceed` and `ready_for_bounded_repeated_offline_demo_action_decoding_runner=true`, the next safe implementation is a separately gated runner using `ALLOW_REPEATED_OFFLINE_DEMO_DECODING=1`. It may load local SmolVLA on CPU and run at most three HDF5 timestep action decodes, with no simulator environment, no rollout, no training, no downloads, no GPU job, no OpenVLA-OFT, and no paper claim.

39. Bounded repeated offline demonstration action-decoding command:

```powershell
$env:ALLOW_REPEATED_OFFLINE_DEMO_DECODING="1"
powershell -ExecutionPolicy Bypass -File scripts\110_bounded_repeated_offline_demo_action_decoding.ps1
Remove-Item Env:\ALLOW_REPEATED_OFFLINE_DEMO_DECODING -ErrorAction SilentlyContinue
```

Current expected interpretation: this runner is diagnostic-only even when it passes. If repeated offline alignment remains weak, keep rollout scaling blocked and inspect VLM-enabled loading risk, checkpoint provenance, and action normalization before another learned-policy rollout. If repeated alignment is moderate or strong, plan a tiny offline baseline comparison before rollout scaling.

Current local result: the runner passed on three HDF5 timesteps, but repeated offline alignment remained weak (`mean_action_l1_to_expert=0.412322`, `mean_action_mse_to_expert=0.286972`, clipped action values total `3`, `load_vlm_weights=false`). Keep learned-policy rollout scaling blocked.

40. Next safe step: create a report-only VLM-enabled loading risk/provenance plan. It should estimate whether acquiring/loading the full `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` weights is official, token-free, size-bounded, and memory-safe. Do not download full VLM weights or run VLM-enabled loading until that risk assessment is green.

41. VLM-enabled loading risk/provenance planning command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\111_plan_vlm_enabled_loading_risk.ps1
```

If this planner reports `decision=proceed` and `ready_for_vlm_weight_acquisition_plan=true`, the next safe step is a separately gated VLM weight acquisition plan/runner for required `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` files only. Do not load VLM weights until acquisition and a bounded load-smoke plan pass.

Current local result: green. Metadata reports `apache-2.0`, public/ungated, required files about `1.895GB`, free disk after estimate about `419GB`, and no token/login/license/payment requirement. Next safe step: create and run a separately gated VLM weight acquisition runner for required files only under `ALLOW_DOWNLOADS=1`; still do not load the model until a later bounded load-smoke plan passes.

42. VLM required-file acquisition command:

```powershell
$env:ALLOW_DOWNLOADS="1"
powershell -ExecutionPolicy Bypass -File scripts\112_acquire_vlm_required_files.ps1
Remove-Item Env:\ALLOW_DOWNLOADS -ErrorAction SilentlyContinue
```

Current local result: passed. The runner acquired the bounded required files from `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` into `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`, including root `model.safetensors` plus config/tokenizer/processor files. Target size after acquisition is about `1.895GB`. The task performed no model load, inference, training, rollout, GPU job, OpenVLA-OFT execution, package install, token access, or paper claim.

Next safe step: create a bounded VLM-enabled load-smoke planner. It must estimate CPU RAM/runtime risk before any model load and keep rollout scaling blocked unless offline action-decoding alignment improves.

43. VLM-enabled load-smoke planning command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\113_plan_vlm_enabled_load_smoke.ps1
```

If this reports `decision=proceed`, the next safe implementation is a separately gated CPU-first load-only runner for `load_vlm_weights=true`. The future runner must require `ALLOW_HEAVY_IMPORT=1` and `ALLOW_VLM_ENABLED_LOAD_SMOKE=1`, perform no inference, no training, no rollout, no GPU job by default, no OpenVLA-OFT, no token access, and no paper claim.

Current local result: `decision=proceed` and `ready_for_bounded_vlm_enabled_load_smoke_runner=true`. The next safe task is to implement `scripts\114_bounded_vlm_enabled_load_smoke.ps1` as CPU-first load-only construction with `load_vlm_weights=true`, capped at 15 minutes, no inference, no training, no rollout, no GPU job by default, no OpenVLA-OFT, no token access, and no paper claim.

44. Bounded VLM-enabled load-only smoke command:

```powershell
$env:ALLOW_HEAVY_IMPORT="1"
$env:ALLOW_VLM_ENABLED_LOAD_SMOKE="1"
powershell -ExecutionPolicy Bypass -File scripts\114_bounded_vlm_enabled_load_smoke.ps1
Remove-Item Env:\ALLOW_VLM_ENABLED_LOAD_SMOKE -ErrorAction SilentlyContinue
Remove-Item Env:\ALLOW_HEAVY_IMPORT -ErrorAction SilentlyContinue
```

This runner is allowed only after the `113` planner is green. It is load-only and CPU-first. If it passes, the next safe step is a planning-only repeated offline demonstration action-decoding recheck using VLM-enabled loading. Do not run rollout scaling until offline alignment improves.

Current local result: passed. The runner constructed local SmolVLA with `load_vlm_weights=true` on CPU, loaded about `450M` parameters, allocated `0MB` CUDA memory, completed in about `14.5s`, and performed no inference, training, rollout, download, install, GPU job, OpenVLA-OFT execution, token access, or paper claim.

45. Next safe step: plan a bounded repeated offline demonstration action-decoding recheck with VLM-enabled loading. The planner should compare against the previous `load_vlm_weights=false` repeated offline diagnostic and authorize at most three local HDF5 timesteps, CPU-only, no simulator, no rollout, no training, no GPU job, no OpenVLA-OFT, and no paper claim.

Planning command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\115_plan_vlm_enabled_repeated_offline_decoding.ps1
```

If this reports `decision=proceed`, the next safe implementation is a separately gated runner using `ALLOW_HEAVY_IMPORT=1` and `ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING=1`. The runner may load local SmolVLA with VLM weights on CPU and decode at most three HDF5 timesteps, but must not create simulator environments, rollout, train, download, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

Current local result: `decision=proceed`, runner-ready true. It selects the same LIBERO HDF5 file and timesteps `0`, `136`, and `271`. The future runner should compare VLM-enabled action L1/MSE, clipping, and alignment signal against the previous `load_vlm_weights=false` repeated offline report.

46. Bounded VLM-enabled repeated offline decoding command:

```powershell
$env:ALLOW_HEAVY_IMPORT="1"
$env:ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING="1"
powershell -ExecutionPolicy Bypass -File scripts\116_bounded_vlm_enabled_repeated_offline_decoding.ps1
Remove-Item Env:\ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING -ErrorAction SilentlyContinue
Remove-Item Env:\ALLOW_HEAVY_IMPORT -ErrorAction SilentlyContinue
```

If this passes, summarize VLM-enabled versus no-VLM offline metrics before any rollout decision.

Current local result: passed. With `load_vlm_weights=true`, the bounded offline recheck decoded timesteps `0`, `136`, and `271` on CPU. Mean action L1/MSE improved from the previous no-VLM repeated diagnostic (`0.412322` / `0.286972`) to `0.301665` / `0.216188`, but the alignment signal remains `weak`, every adapted action still clipped one value, and rollout scaling remains blocked.

47. Next safe step: generate a report-only VLM-enabled versus no-VLM offline decoding summary and action-normalization/provenance diagnosis. Do not scale learned-policy rollouts until the offline alignment issue is explained or a separate green risk gate identifies a narrower rollout hypothesis.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\117_summarize_vlm_enabled_offline_decoding.ps1
```

If this summary confirms weak alignment and clipping, the next autonomous task is a report-only action-normalization/provenance audit over the SmolVLA processor stats, 6D policy action convention, 7D LIBERO action convention, and adapter clipping behavior.

Current local result: summary passed. VLM-enabled loading reduced mean action L1/MSE by `26.838%` / `24.666%`, but alignment remained `weak`, clipping persisted, ACTION `MEAN_STD` normalization is active, and the 6D policy action convention still needs provenance analysis against the 7D LIBERO action convention.

48. Next safe step: create the report-only action-normalization/provenance audit. It should inspect local processor stats, action unnormalizer metadata, 6D policy action dimensions, 7D LIBERO expert-action dimensions, adapter clipping, and whether the action scale/statistics explain the weak offline alignment. Do not load models, infer, train, rollout, download, use GPU jobs, execute OpenVLA-OFT, or make paper claims for this audit.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\118_audit_action_normalization_provenance.ps1
```

If this audit finds action-stat provenance or scale mismatch, the next autonomous step is a planning-only action-stat mapping/checkpoint-task provenance correction plan, not another learned-policy rollout variant.

Current local result: audit passed with `decision=no_go_rollout_scaling`. The processor action-stat prefixes are `so100`, `so100-blue`, and `so100-red`; action mean/std magnitudes are far larger than local LIBERO expert-action previews; the policy action shape is `[6]`; the local adapter path remains 7D; clipping persists.

49. Next safe step: create a planning-only action-stat mapping/checkpoint-task provenance correction plan. It should decide whether to compare in normalized action space, bypass or replace mismatched unnormalizer stats for LIBERO diagnostics, seek a LIBERO-aligned SmolVLA checkpoint/source, or keep learned-policy rollouts blocked and pivot to offline head/TCA-Map evidence. Do not alter model weights, train, rollout, download new checkpoints, use GPU jobs, execute OpenVLA-OFT, or make paper claims in the planning step.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\119_plan_action_stat_provenance_correction.ps1
```

Expected next selection: a report-only LIBERO action-stat subset audit over local HDF5 files.

Current local result: plan passed with `decision=reduce_scope` and selected `libero_action_stat_subset_audit`.

50. Next safe step: implement the report-only LIBERO action-stat subset audit. It should read a bounded number of local HDF5 files under `LIBERO_DATA_ROOT`, compute action mean/std/min/max and action dimension statistics, compare them against checkpoint SO100 processor stats, write ignored runtime reports, and keep rollouts/model loading/training/downloads/GPU/OpenVLA/paper claims false.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\120_audit_libero_action_stats.ps1
```

If this confirms mismatch, the next autonomous step is a planning-only normalized-action-space probe or checkpoint/task provenance resolution plan.

Current local result: audit passed with `decision=no_go_rollout_scaling`. It sampled 5 files and 2500 actions, confirmed LIBERO action dim `7`, LIBERO max abs `1.0`, checkpoint SO100 action-stat prefixes, checkpoint mean max abs `125.720543`, checkpoint std max `59.359951`, scale mismatch true, and dimension mismatch true.

51. Next safe step: create a planning-only normalized-action-space probe / checkpoint-task provenance resolution plan. It should choose between a bounded normalized action-space diagnostic, a checkpoint source/provenance check, or pivoting learned-policy rollout work away from this checkpoint. Do not run model loading, inference, rollout, training, downloads, GPU jobs, OpenVLA-OFT, or paper claims in the planning step.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\121_plan_normalized_action_space_probe.ps1
```

Expected interpretation: if the plan reports `decision=reduce_scope` and `selected_next_step=checkpoint_task_provenance_resolution`, do not run a normalized-action-space runner yet. First create a report-only checkpoint/task provenance resolution audit. Keep rollout scaling, postprocessor bypass/replacement, model loading, inference, training, downloads, GPU jobs, OpenVLA-OFT, and paper claims blocked.

Current local result: plan passed with `decision=reduce_scope`, `selected_next_step=checkpoint_task_provenance_resolution`, `ready_for_checkpoint_task_provenance_resolution=true`, and `ready_for_bounded_normalized_action_space_probe_runner=false`. The next safe task is a report-only checkpoint/task provenance resolution audit.

52. Next safe step: resolve checkpoint/task provenance in a report-only audit. It should inspect local SmolVLA checkpoint config, policy preprocessor/postprocessor metadata, local model-card README, the normalized-action plan, and the LIBERO action-stat subset audit. It must not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, alter policy behavior, access tokens, or make paper claims.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\122_resolve_checkpoint_task_provenance.ps1
```

Expected interpretation: if the audit reports `decision=no_go_learned_policy_rollout_scaling`, do not use the current base checkpoint as LIBERO learned-policy rollout evidence. Continue through offline/head TCA-Map and required LoRA evidence, or create a separate source-resolution plan for a LIBERO-action-aligned SmolVLA checkpoint.

Current local result: audit passed with `decision=no_go_learned_policy_rollout_scaling`. The current checkpoint is not valid as LIBERO learned-policy rollout evidence because its action shape/stat provenance remains 6D/SO100-like while local LIBERO actions are 7D/unit-scale. It selected `pivot_to_offline_head_tca_map_and_lora_or_find_libero_aligned_checkpoint`.

53. Next safe step: do not run more learned-policy LIBERO rollout scaling with the current base checkpoint. Choose the stronger low-compute paper path: either continue offline/head TCA-Map plus required LoRA evidence on real LIBERO data, or create a separate source-resolution plan for a LIBERO-action-aligned SmolVLA checkpoint. Prefer a report-only pivot plan first; it should not download, train, rollout, load models, use GPU jobs, execute OpenVLA-OFT, or make paper claims.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\123_plan_offline_tca_map_lora_pivot.ps1
```

Expected interpretation: if the plan reports `decision=pivot_offline_evidence_ladder`, the next safe task is a report-only offline evidence table and gap report consolidating ActionMap, TCA-Map, Distributional TCA-Select, required LoRA, and remaining rollout/checkpoint blockers.

Current local result: pivot plan passed with `decision=pivot_offline_evidence_ladder`, `ready_for_offline_evidence_table=true`, `ready_for_lora_scaleup_plan=true`, and `ready_for_learned_policy_rollout_scaling=false`. It selected `consolidate_offline_tca_lora_evidence_table_and_gap_report`.

54. Next safe step: create a report-only offline evidence table and gap report. It should consolidate ActionMap, TCA-Map, Distributional TCA-Select, required LoRA, and remaining rollout/checkpoint blockers. It must not train, rollout, download, load models, infer, use GPU jobs, execute OpenVLA-OFT, or make paper claims.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\124_generate_offline_evidence_gap_report.ps1
```

Expected interpretation: if this report passes, the next safe task is a bounded LoRA/offline-proxy scale-up plan on real LIBERO HDF5 subsets. Keep current-checkpoint learned-policy rollout scaling and paper claims blocked.

Current local result: evidence gap report passed with `decision=offline_evidence_table_ready`, `ready_for_lora_scaleup_plan=true`, `ready_for_offline_proxy_extension=true`, and `ready_for_learned_policy_rollout_scaling=false`.

55. Next safe step: plan a bounded LoRA/offline-proxy scale-up on real LIBERO HDF5 subsets. Keep it planning-only first. It should define max files/pairs/samples/steps, CPU-first defaults, no full fine-tuning, no rollout, no model loading unless separately gated, no GPU job unless a later budget is green, no OpenVLA-OFT, and no paper claim.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\125_plan_bounded_lora_offline_scaleup.ps1
```

Expected interpretation: if the plan reports `decision=proceed_bounded_offline_lora_scaleup_runner`, the next safe task is a separately gated CPU-only offline LoRA scale-up runner under `ALLOW_TINY_TRAINING=1`. It must keep outputs as offline proxy diagnostics only.

Current local result: plan passed with `decision=proceed_bounded_offline_lora_scaleup_runner`, `ready_for_bounded_lora_offline_scaleup_runner=true`, limits `max_pairs=16`, `max_samples=64`, `max_steps=64`, `lora_rank=4`, `device=cpu`, and required future gate `ALLOW_TINY_TRAINING=1`.

56. Next safe step: implement the separately gated CPU-only offline LoRA scale-up runner. It may run only under task-local `ALLOW_TINY_TRAINING=1`, may use local real LIBERO HDF5 data, and must remain offline proxy only. It must not load SmolVLA, import heavy VLA models, use GPU jobs, rollout, execute OpenVLA-OFT, full fine-tune, download, install packages, access tokens, or make paper claims.

Command:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\126_bounded_lora_offline_scaleup.ps1
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

Expected interpretation: if the runner reports `bounded_lora_offline_scaleup_passed=true`, refresh the offline evidence table and gap report to include the bounded LoRA scale-up result. This remains offline proxy evidence only, not standard success, not rollout success, and not paper-grade evidence.

Current local result: runner passed with `bounded_lora_offline_scaleup_passed=true`, `record_count=16`, `max_steps=64`, CPU-only execution, `ready_for_offline_evidence_refresh=true`, `ready_for_rollout=false`, and `ready_for_paper_claim=false`. The next safe step is to refresh the offline evidence table/gap report with this bounded scale-up result.

57. Next safe step: refresh the offline evidence table and gap report so it includes `reports\bounded_lora_offline_scaleup_report.json`. This should be report-only and must not train, download, load models, infer, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\124_generate_offline_evidence_gap_report.ps1
```

Expected interpretation: if `bounded_lora_scaleup_included=true`, use the refreshed table as scale-up-aware offline proxy evidence only. The next safe task is a report-only attribution-gap synthesis, not rollout scaling or a paper claim.

Current local result: evidence gap refresh passed with `bounded_lora_scaleup_included=true`, `bounded_lora_scaleup_record_count=16`, 9 evidence rows, bounded TCA-Map + LoRA vs ActionMap + LoRA action L1 delta `-0.004018`, wrong-target proxy delta `-0.4375`, and paper/rollout readiness still false.

58. Next safe step: create a report-only scale-up-aware attribution-gap synthesis. It should explain what the bounded offline evidence supports, what it does not support, why Distributional TCA-Select currently shows no additional LoRA proxy gain in this runner, and what must be resolved before paper-grade rollout claims. It must not train, download, load models, infer, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\127_synthesize_scaleup_attribution_gaps.ps1
```

Expected interpretation: if the synthesis passes, the next safe task is a report-only offline TCA-Select ambiguity/stress-test plan. Do not run rollout scaling or make paper claims from the offline proxy evidence.

Current local result: synthesis passed with `decision=scaleup_attribution_gaps_ready`, `bounded_lora_scaleup_included=true`, bounded TCA-Map + LoRA vs ActionMap + LoRA wrong-target proxy delta `-0.4375`, and bounded TCA-Select + LoRA vs TCA-Map + LoRA deltas `0.0`. The next safe task is a report-only TCA-Select candidate-ambiguity stress-test plan.

59. Next safe step: plan an offline TCA-Select ambiguity/stress test that can isolate inference-time selection gain without training or rollout. It should define candidate diversity, ambiguous target/action pairs, scoring metrics, and pass/fail criteria, but remain planning-only first.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\128_plan_tca_select_ambiguity_stress_test.ps1
```

Expected interpretation: if the plan reports `decision=proceed_offline_tca_select_ambiguity_stress_runner`, the next safe task is an offline CPU-only stress-test runner over existing local counterfactual artifacts. It must remain offline proxy only.

Current local result: plan passed with `decision=proceed_offline_tca_select_ambiguity_stress_runner`, `ready_for_offline_tca_select_ambiguity_stress_runner=true`, `candidate_count=8`, `max_records=64`, CPU-only, and paper/rollout readiness false.

60. Next safe step: implement the CPU-only offline TCA-Select ambiguity stress-test runner. It should use existing local counterfactual artifacts, generate ambiguous candidate heatmaps without loading SmolVLA, and report offline proxy metrics only.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\129_run_tca_select_ambiguity_stress_test.ps1
```

Expected interpretation: if the runner passes, update the attribution synthesis/evidence table with stress-test results. It remains offline proxy evidence only.

Current local result: runner passed with `tca_select_ambiguity_stress_passed=true`, `record_count=16`, selected wrong-target proxy rate `0.0`, top-heatmap wrong-target proxy rate `1.0`, wrong-target proxy delta `-1.0`, selected action L1 `0.0`, top-heatmap action L1 `0.164299`, action L1 delta `-0.164299`, CPU-only execution, no model loading, no training, no rollout, no GPU job, no OpenVLA-OFT, and paper readiness false.

61. Next safe step: refresh the scale-up attribution synthesis and/or offline evidence table so it includes `reports\tca_select_ambiguity_stress_report.json`. This should remain report-only and must not train, download, load models, infer with SmolVLA, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\127_synthesize_scaleup_attribution_gaps.ps1
```

Expected interpretation: if the refreshed synthesis reports `tca_select_ambiguity_stress_included=true`, the next safe task is a report-only refresh of the consolidated offline evidence table with a dedicated TCA-Select ambiguity-stress row.

62. Next safe step: refresh the consolidated offline evidence table with a dedicated Distributional TCA-Select ambiguity-stress row from `reports\tca_select_ambiguity_stress_report.json`.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\124_generate_offline_evidence_gap_report.ps1
```

Expected interpretation: the refreshed evidence table should report `tca_select_ambiguity_stress_included=true`, preserve offline-proxy labels, and keep learned-policy rollout scaling and paper claims blocked.

Current local result: evidence refresh passed with `evidence_row_count=10`, `tca_select_ambiguity_stress_included=true`, wrong-target proxy delta `-1.0`, action L1 delta `-0.164299`, and paper/rollout readiness false. The refreshed attribution synthesis now recommends a report-only learned-policy candidate-generation readiness check.

63. Next safe step: plan a report-only learned-policy candidate-generation readiness check. It should determine what would be needed to generate real candidate action heatmaps from the local SmolVLA/TCA-Map stack without running model inference yet.

Constraints:

- no model inference,
- no heavy VLA import,
- no training,
- no rollout,
- no simulator execution,
- no GPU job,
- no OpenVLA-OFT,
- no paper claim.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\130_plan_candidate_generation_readiness.ps1
```

Expected interpretation: if the plan passes, implement a synthetic-tensor candidate-generation contract checker before any real model inference.

Current local result: candidate-generation readiness planning passed with `ready_for_candidate_generation_contract_checker=true`, `ready_for_real_candidate_generation_smoke_plan=true`, and `ready_for_real_candidate_generation_smoke_execution=false`. Prior load-only, single-sample interface, and feature-cache reports were green. No model load, model inference, training, rollout, GPU job, simulator execution, OpenVLA-OFT, or paper claim was performed.

64. Next safe step: implement a synthetic-tensor candidate-generation contract checker. It should validate candidate list, low-resolution heatmap, masked heatmap, metadata, and TCA-Select input/output contracts without loading SmolVLA or running model inference.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\131_check_candidate_generation_contract.ps1
```

Expected interpretation: if the contract checker passes, plan a separately gated real candidate-generation smoke. Do not run model inference in the contract checker.

Current local result: synthetic contract checker passed with `candidate_generation_contract_check_passed=true`, candidate count `4`, heatmap grid `8`, selected candidate index `0`, max GPU memory `0.0 MB`, and no model load/inference, training, rollout, GPU job, simulator execution, OpenVLA-OFT, external verifier, privileged inference, or paper claim.

65. Next safe step: create a planning-only risk gate for a separately bounded real candidate-generation smoke. It must decide whether a future task may set `ALLOW_HEAVY_IMPORT=1` and `ALLOW_SINGLE_SAMPLE_INFERENCE=1` for a single-sample candidate heatmap smoke, while keeping rollout, training, GPU-heavy execution, OpenVLA-OFT, and paper claims blocked.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\132_plan_real_candidate_generation_smoke.ps1
```

Expected interpretation: if the plan is green, implement the bounded real candidate-generation smoke in a separate branch, but do not execute it unless all required task-local gates are set.

Current local result: plan passed with `decision=proceed_bounded_real_candidate_generation_smoke_implementation`, `ready_for_real_candidate_generation_smoke_implementation=true`, `ready_for_real_candidate_generation_smoke_execution=false`, no blockers, and required future gates `ALLOW_REAL_CANDIDATE_GENERATION_SMOKE=1`, `ALLOW_HEAVY_IMPORT=1`, and `ALLOW_SINGLE_SAMPLE_INFERENCE=1`.

66. Next safe step: implement the bounded real candidate-generation smoke script and tests without executing it by default. The script must refuse to run unless all three required gates are set task-locally and must still forbid rollout, training, simulator execution, OpenVLA-OFT, downloads, external verifiers, privileged state, and paper claims.

Current local result: implementation scaffold added in `scripts\133_bounded_real_candidate_generation_smoke.ps1` and `tca_map.smolvla.real_candidate_generation_smoke`. Default execution remains blocked unless `ALLOW_REAL_CANDIDATE_GENERATION_SMOKE=1`, `ALLOW_HEAVY_IMPORT=1`, and `ALLOW_SINGLE_SAMPLE_INFERENCE=1` are all set for the task. The script writes ignored runtime reports, uses CPU by default, caps candidates at 4 and grid size at 8, and still forbids rollout, training, simulator execution, downloads, OpenVLA-OFT, external verifiers, privileged state, and paper claims.

67. Next safe step: run a planning-only synthesis for the real candidate-generation smoke scaffold or, if the autonomous risk assessment is green, run the bounded real candidate-generation smoke with all three task-local gates and label it engineering evidence only. Do not treat a passing smoke as standard success, rollout success, or paper-grade evidence.

## WSL Simulator Dependency Ladder Standing Approval

Current autonomous simulator-readiness sequence:

1. WSL simulator dependency risk assessment. Done.
2. Create or use `~/.venvs/tca_map_sim`. Done.
3. Install only minimal import-readiness Python dependencies. Done, without sudo or apt.
4. Link local RoboSuite/LIBERO source checkouts into the existing WSL venv without downloads. Done.
5. Rerun bounded simulator import smoke. Done.
6. Run a separate bounded render/reset-step risk planner. Done.
7. Create and run a separate bounded render-smoke branch only if the render-smoke assessment is green. Done; current local result passed with OSMesa.
8. Run a separate bounded reset/step-smoke risk assessment and smoke. Done.
9. Run a separate bounded tiny diagnostic rollout risk assessment. Done.
10. Run a separate bounded tiny diagnostic rollout with task-local `ALLOW_TINY_ROLLOUT=1` only if the risk assessment is green. Done.
11. Run a separate bounded LIBERO/RoboSuite zero-action diagnostic rollout planner and runner only if risk assessment is green. Done.
12. Run `scripts\66_plan_libero_policy_rollout_readiness.ps1` before any learned-policy rollout. Current local state is green after WSL SmolVLA runtime setup/readiness and WSL single-action smoke.
13. Create a separately gated tiny learned-policy rollout runner. Stop before multi-seed rollout, OpenVLA-OFT, full fine-tuning, external upload, or unsupported paper-level claims.

Allowed WSL apt packages are limited to `python3-pip`, `python3-venv`, `python3-dev` if needed, `build-essential` only if required for Python package builds, `git` if missing, and `curl` or `wget` only for official setup checks. Stop for sudo password, token/login, license/payment, CUDA/driver/toolkit, graphics-stack changes, OpenVLA-OFT, paper claims, or rollout beyond the tiny diagnostic limits.
