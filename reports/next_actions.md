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
11. Keep evidence labels as diagnostic or local pilot until baselines, ablations, and repeated benchmark protocol are implemented.
12. Stop before multi-seed rollout, paper-grade claims, OpenVLA-OFT, full fine-tuning, external upload, token/secret access, payment/license click-through, or destructive/system-level changes.

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
