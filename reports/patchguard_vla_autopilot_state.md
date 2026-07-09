# PatchGuard-VLA Autopilot State

Date: 2026-07-09 KST

Branch: `codex/patchguard-vla-state1b`

## Scope

This bounded sequence is STATE 0-1 plus STATE 1B:

- define PatchGuard-VLA around kinematic-consistent defense against physical patch attacks,
- compare against recent VLA patch and robustness literature,
- check local SmolVLA/LIBERO runtime and signal availability,
- measure a tiny clean-vs-patched action-divergence gate if local assets allow,
- resolve installable adapter tooling if it is missing,
- run only a tiny adapter feasibility smoke after explicit gates are set.

## Current Evidence Before New STATE 1 Runner

Existing local reports show:

- SmolVLA runtime dependencies are present.
- VLM-enabled local SmolVLA load-only smoke passed on CPU.
- VLM-enabled repeated offline action decoding over three local LIBERO HDF5 timesteps passed as diagnostic evidence.
- Local OpenVLA/OpenVLA-OFT assets were not found under `C:\assets\checkpoints` or `C:\assets\hf_home`; OpenVLA-OFT was not executed because it is outside the local bounded run.
- Local LIBERO HDF5 exposes `agentview_rgb`, `eye_in_hand_rgb`, `ee_states`, `joint_states`, and 7D expert actions.
- Prior bounded RoboSuite/LIBERO visual checks are present: MuJoCo offscreen render smoke passed with a 64x64 RGB image, and camera-source diagnostics recorded `agentview_image` and `robot0_eye_in_hand_image` sources.
- Existing audits block rollout scaling because action-stat provenance and 6D-to-7D action adaptation remain unresolved.
- The prior STATE 1 `TOO_HEAVY_LOCAL` is an installable environment blocker, not a PatchGuard method kill.
- QLoRA tooling was not locally available in STATE 1 because `peft` and `bitsandbytes` were absent.

## New Artifacts

- `tca_map/patchguard_vla/diagnostic.py`
- `scripts/230_patchguard_vla_state1_diagnostic.ps1`
- `tests/test_patchguard_vla_diagnostic.py`
- `reports/patchguard_vla_state1_result.json` after runner execution
- `reports/patchguard_vla_state1_result.md` after runner execution
- `tca_map/patchguard_vla/state1b.py`
- `scripts/231_patchguard_vla_state1b_probe.ps1`
- `reports/patchguard_vla_state1b_result.json`
- `reports/patchguard_vla_state1b_result.md`

## Safety State

Full PatchGuard training happened: no.

Bounded tiny LoRA smoke happened: yes, batch size 1, rank 4, 10 steps per variant.

Full benchmark happened: no.

OpenVLA-OFT happened: no.

Large model or dataset downloads happened: no.

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

Interpretation: STATE 1 found enough attack and kinematic-signal evidence to avoid `KILL_ATTACK_NOT_REPRODUCIBLE` and `KILL_NO_KINEMATIC_SIGNAL`, and the cutout proxy did not fully solve the fixed-patch effect. Its `TOO_HEAVY_LOCAL` result is now reclassified as an installable environment blocker because `peft` and `bitsandbytes` were allowed and installed in STATE 1B.

## STATE 1B Result

Runner:

```powershell
$env:ALLOW_HEAVY_IMPORT="1"
$env:ALLOW_PATCHGUARD_VLA_STATE1B="1"
$env:ALLOW_PATCHGUARD_TINY_LORA_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\231_patchguard_vla_state1b_probe.ps1 -MaxSteps 10 -DependencyInstallHappened
```

Result file:

- `reports/patchguard_vla_state1b_result.json`
- `reports/patchguard_vla_state1b_result.md`

Observed decision: `KILL_BASELINE_DOMINATED`

Key evidence:

- install happened: yes, `peft` and `bitsandbytes`
- PEFT status: `0.19.1`, import and dummy LoRA smoke passed
- bitsandbytes status: `0.49.2`, 4-bit and 8-bit CUDA kernel smokes passed
- CUDA/GPU: PyTorch `2.10.0+cu128`, CUDA runtime `12.8`, NVIDIA GeForce RTX 5080
- model path: `C:\assets\checkpoints\smolvla`
- LoRA injection happened: yes, target modules `state_proj`, `action_in_proj`, `action_out_proj`
- trainable params: `9984`
- tiny training smoke happened: yes
- loss computed: yes
- VRAM peak MB: `2224.845`
- runtime sec: `57.438`
- clean metric: `0.422465`
- patched metric: `0.134391`
- cutout/random-erasing metric: `0.02973`
- generic adversarial LoRA metric: `0.142803`
- PatchGuard metric: `0.13356`
- PatchGuard beats baseline: no, because it did not beat both generic adversarial augmentation and cutout/random erasing

Interpretation: STATE 1B proves a local adapter path exists, so PatchGuard is not environment-blocked. The bounded tiny smoke does not justify STATE 2 because the PatchGuard variant failed the declared baseline-dominance gate.

## Decision Alternatives

If the new diagnostic finds no patch effect, decision is `KILL_ATTACK_NOT_REPRODUCIBLE`.

If it finds a patch effect but the cutout baseline removes it, decision is `KILL_BASELINE_DOMINATED`.

If it finds a patch effect and kinematic signal but local real-adapter tooling is absent, decision is `TOO_HEAVY_LOCAL`.

Only if the STATE 1B adapter and baseline gates pass is the decision `READY_FOR_PATCHGUARD_LORA_STATE2`.
