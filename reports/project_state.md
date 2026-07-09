# Project State

Date: 2026-07-10 KST

Target branch: `main`

Implementation branch: `codex/fix-official-smolvla-rollout-protocol`

Current decision: `LORA_CHECKPOINTS_MISSING_REGENERATION_REQUIRED`

## Current Route

The archived custom SmolVLA 7D adapter route remains stopped. The valid route is official SmolVLA/LeRobot reproduction first, using official preprocessing, normalization, action conventions, dataset format, and evaluation stack.

This state update is a no-experiment protocol fix. It did not run experiments, train, use GPU, download assets, run simulator rollout, run model inference, run OpenVLA-OFT, design a new method, revive FCAR, rerun LoRA seeds, modify past results, delete artifacts, or overwrite artifacts.

## Fixed In This Protocol Pass

- model revision lock: `lerobot/smolvla_libero` at `31d453f7edd78c839a8bbc39744a292686daf0de`
- dataset revision lock: `lerobot/libero` at `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
- package/env version lock recorded in `configs/official_smolvla_repro_lock.yaml`
- package source-commit gap recorded as `VERSION_LOCKED_SOURCE_COMMIT_UNAVAILABLE`
- baseline naming policy frozen
- LoRA checkpoint persistence policy frozen
- official rollout action semantics frozen
- static-mix compute accounting requirement frozen
- two-stage official closed-loop rollout protocol frozen
- official eval readiness classified without rollout

## Current Locks

- lock file: `configs/official_smolvla_repro_lock.yaml`
- split manifest SHA256: `1279F939648CF13E2F599084E42631681E1DFA5606B5D9B0851FFEB32710934B`
- metric protocol SHA256: `64430225940C5168B3734BB40F9F48AD02877E0BA04DC804367AFBB214AE486E`
- stable prediction artifact SHA256: `88DCA06AA05D69E8BC4FB3F1C5A7C7D22B1DC4438C65103EFD2389F24D35D59C`
- LoRA seed repro result SHA256: `BAA9BD61DA4631F8CF7020198147A52F66435DBFCDDF02717BE2188CC8E79505`

## Canonical Baselines

Future reports must use:

- `frozen_base`: Frozen SmolVLA-LIBERO Base Policy
- `rank4_lora`: Standard Rank-4 LoRA
- `validation_selected_action_space_static_mix`: validation-selected action-space interpolation, not adapter soup
- `task_or_instruction_router_proxy`: local proxy, not official MoIRA
- `frame_oracle_upper_bound`: oracle upper bound, not deployable
- `task_oracle_upper_bound`: oracle upper bound, not deployable

Historical metric artifacts are preserved as-is. Use the legacy mapping in `reports/official_smolvla_baseline_naming_policy.md` instead of rewriting old metrics.

## LoRA Checkpoint Status

Seed-specific official LoRA adapter checkpoint bundles are required for any official rollout or final reported LoRA result.

| Seed | Status |
| --- | --- |
| `11` | `CHECKPOINT_MISSING` |
| `22` | `CHECKPOINT_MISSING` |
| `33` | `CHECKPOINT_MISSING` |

Prediction JSON artifacts cannot replace adapter checkpoints.

## Official Eval Status

Classification: `MISSING_OFFICIAL_EVAL_DEPENDENCY`

- `lerobot-eval` exists and maps to `lerobot.scripts.lerobot_eval:main`
- local help lists `--env.type=libero`
- local source imports official LIBERO env modules
- local Python environment lacks `libero`
- local Python environment lacks `robosuite`
- native Windows official rollout remains unproven; WSL/Linux or dependency repair is recommended

## Rollout Protocol Status

The two-stage official closed-loop LIBERO rollout protocol is defined but not executable yet.

Stage A is a bounded readiness pilot over:

- `frozen_base`
- `rank4_lora`
- `validation_selected_action_space_static_mix`

Stage B is scaleup after Stage A passes with expanded tasks/seeds and confidence intervals.

## Conclusion

`LORA_CHECKPOINTS_MISSING_REGENERATION_REQUIRED`

The revision and protocol gaps are now recorded, but official rollout must not start until the seed LoRA adapter bundles exist and the official LIBERO eval environment is fixed.
