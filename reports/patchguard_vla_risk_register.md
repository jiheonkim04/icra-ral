# PatchGuard-VLA Risk Register

Date: 2026-07-09 KST

| Risk | Severity | Evidence to check | Mitigation |
| --- | --- | --- | --- |
| Patch has no measurable effect on local SmolVLA actions | High | `patchguard_vla_state1_result` action divergence | Kill as `KILL_ATTACK_NOT_REPRODUCIBLE`; do not train. |
| Cutout/random erasing removes the fixed-patch effect | High | cutout policy L1 vs clean compared with fixed-patch L1 | Kill as `KILL_BASELINE_DOMINATED`; PatchGuard is unnecessary locally. |
| Local SmolVLA checkpoint is action-provenance mismatched for LIBERO | High | existing action-normalization and HDF5 interface audits | Keep evidence diagnostic only; do not claim task success. |
| No segmentation or arm mask is available | Medium | HDF5 observation keys and simulator rendering support | Use EEF/proprio state as first non-leaking signal; treat arm mask as future work unless verified. |
| Kinematic signal leaks eval labels or privileged state | High | state adapter metadata and source keys | Use only observation.state/EEF/joint/gripper fields available at inference. |
| PatchGuard collapses to generic augmentation | High | comparison against cutout and visual augmentation proxy | Require improvement over generic baselines before any adapter smoke. |
| Real LoRA/adapter path is unavailable locally | High | `peft`/`bitsandbytes` availability and LeRobot adapter wiring | Return `TOO_HEAVY_LOCAL` unless tooling is approved or moved to WSL/Linux/cloud. |
| RAM pressure during local SmolVLA CPU inference | Medium | runtime RSS in state1 JSON | Keep policy calls bounded to 15 and use CPU-only execution. |
| Accidental training or rollout | High | runner gates and policy flags | Refuse training/rollout/OpenVLA-OFT/download gates in the runner. |
| Overclaiming offline proxy evidence | High | final report language | Label all outputs as diagnostic only; no benchmark, SOTA, or paper-grade claims. |

