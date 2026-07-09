# PatchGuard-VLA Autopilot State

Date: 2026-07-09 KST

Branch: `codex/patchguard-vla-state0-state1`

## Scope

This bounded run is STATE 0-1 only:

- define PatchGuard-VLA around kinematic-consistent defense against physical patch attacks,
- compare against recent VLA patch and robustness literature,
- check local SmolVLA/LIBERO runtime and signal availability,
- measure a tiny clean-vs-patched action-divergence gate if local assets allow,
- do not train.

## Current Evidence Before New STATE 1 Runner

Existing local reports show:

- SmolVLA runtime dependencies are present.
- VLM-enabled local SmolVLA load-only smoke passed on CPU.
- VLM-enabled repeated offline action decoding over three local LIBERO HDF5 timesteps passed as diagnostic evidence.
- Local OpenVLA/OpenVLA-OFT assets were not found under `C:\assets\checkpoints` or `C:\assets\hf_home`; OpenVLA-OFT was not executed because it is outside the local bounded run.
- Local LIBERO HDF5 exposes `agentview_rgb`, `eye_in_hand_rgb`, `ee_states`, `joint_states`, and 7D expert actions.
- Existing audits block rollout scaling because action-stat provenance and 6D-to-7D action adaptation remain unresolved.
- QLoRA tooling is not locally available without installs because `peft` and `bitsandbytes` are absent.

## New Artifacts

- `tca_map/patchguard_vla/diagnostic.py`
- `scripts/230_patchguard_vla_state1_diagnostic.ps1`
- `tests/test_patchguard_vla_diagnostic.py`
- `reports/patchguard_vla_state1_result.json` after runner execution
- `reports/patchguard_vla_state1_result.md` after runner execution

## Safety State

Training happened: no.

Full benchmark happened: no.

OpenVLA-OFT happened: no.

Downloads happened: no.

Paper claims made: no.

## STATE 1 Result

Runner:

```powershell
$env:ALLOW_HEAVY_IMPORT="1"
$env:ALLOW_PATCHGUARD_VLA_STATE1="1"
powershell -ExecutionPolicy Bypass -File scripts\230_patchguard_vla_state1_diagnostic.ps1
```

Result file:

- `reports/patchguard_vla_state1_result.json`
- `reports/patchguard_vla_state1_result.md`

Observed decision: `TOO_HEAVY_LOCAL`

Key evidence:

- real local SmolVLA used: yes
- training happened: no
- rollouts happened: no
- downloads happened: no
- GPU jobs happened: no
- task text: `turn on the stove and put the moka pot on it`
- patch effect measured: yes
- max attacked policy-action L1 vs clean: `0.181765`
- max attacked translation-action L2 vs clean: `0.213965`
- kinematic/proprioceptive signal available: yes
- cutout baseline dominated fixed patch: no
- local LoRA/adapter path feasible now without installs: no
- missing local adapter tooling: `peft` and `bitsandbytes`

Interpretation: STATE 1 found enough attack and kinematic-signal evidence to avoid `KILL_ATTACK_NOT_REPRODUCIBLE` and `KILL_NO_KINEMATIC_SIGNAL`, and the cutout proxy did not fully solve the fixed-patch effect. The route still cannot proceed to PatchGuard LoRA smoke locally because real adapter tooling is absent under the no-install/no-training constraints.

## Decision Alternatives

If the new diagnostic finds no patch effect, decision is `KILL_ATTACK_NOT_REPRODUCIBLE`.

If it finds a patch effect but the cutout baseline removes it, decision is `KILL_BASELINE_DOMINATED`.

If it finds a patch effect and kinematic signal but local real-adapter tooling is absent, decision is `TOO_HEAVY_LOCAL`.

Only if all gates pass is the decision `READY_FOR_PATCHGUARD_LORA_SMOKE`.
