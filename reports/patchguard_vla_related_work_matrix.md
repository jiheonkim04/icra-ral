# PatchGuard-VLA Related Work Matrix

Date: 2026-07-09 KST

Primary sources checked in this run:

- VLA-Hijack: https://arxiv.org/abs/2605.28083
- Partially Observable Adversarial Patch Attacks on VLA: https://arxiv.org/html/2606.03556v1
- Universal transferable patch attacks on VLA: https://arxiv.org/abs/2511.21192
- STRONG-VLA: https://arxiv.org/abs/2604.10055
- RobustVLA: https://arxiv.org/abs/2511.01331
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645 and https://openvla-oft.github.io/
- SmolVLA: https://arxiv.org/abs/2506.01844 and https://huggingface.co/lerobot/smolvla_base
- Standard adversarial training: https://arxiv.org/abs/1706.06083
- Adversarial Patch: https://arxiv.org/abs/1712.09665
- Random Erasing: https://arxiv.org/abs/1708.04896
- Cutout: https://arxiv.org/abs/1708.04552

| Anchor | What it solves | What it does not solve | Strongest novelty threat to PatchGuard-VLA | Required baseline or check |
| --- | --- | --- | --- | --- |
| VLA-Hijack | Targets visual proprioception and self-localization, creating a phantom embodiment patch attack across VLA architectures. | It is an attack, not a defense. | PatchGuard's novelty is only credible if it directly counters visual-proprioceptive hijacking rather than generic corruption. | Reproduce or approximate an image patch that changes real VLA actions; measure unchanged proprioception with changed action. |
| Partially Observable Adversarial Patch Attacks on VLA | Studies short-prefix or partial-observation physical patch attacks that can cause longer-horizon failures. | Does not provide a kinematic consistency defense. | Shows that a small local patch may matter even without full-trajectory optimization. | Start with random/fixed patches and only then consider cheap optimization if safe. |
| Universal transferable patch attacks on VLA | Studies universal and transferable physical patches for unknown architectures and sim-to-real settings. | Does not use robot kinematics as a defense signal. | Raises the attack baseline bar; a defense should not be tuned only to one local synthetic patch. | Random/fixed patch evidence is only STATE 1; future work needs stronger transfer attacks. |
| STRONG-VLA | Decouples robustness learning from task optimization under multimodal perturbations. | Generic multimodal perturbation robustness does not specifically enforce visual-proprioceptive or kinematic consistency. | If generic perturbation training matches PatchGuard, the method is baseline dominated. | Generic visual augmentation proxy is mandatory. |
| RobustVLA | Uses robustness-aware reinforcement post-training for VLA robustness. | Does not target physical patch-induced phantom embodiment as a kinematic consistency problem. | RL robustness could dominate if PatchGuard only adds generic robustness. | Compare future adapter smoke against generic robustness/augmentation when feasible. |
| OpenVLA / OpenVLA-OFT | Provides strong VLA and optimized fine-tuning recipes; OpenVLA-OFT reports major LIBERO gains and faster action generation. | Too heavy for this local STATE 1 run and not specific to physical patch defense. | A strong fine-tuning recipe may explain gains unless PatchGuard beats standard adaptation under attack. | Do not run OpenVLA-OFT locally; use it as a baseline threat only. |
| SmolVLA | Compact VLA that can run on consumer hardware and local CPU/GPU; local checkpoint already loads and decodes tiny LIBERO samples. | The local checkpoint has action-stat/provenance mismatch risk for LIBERO and is not itself a defense. | If patch effects are absent or dominated by checkpoint mismatch, PatchGuard has no local empirical basis. | Use SmolVLA only for bounded offline action divergence in STATE 1. |
| Standard adversarial training | Provides a robust optimization frame for adversarial examples. | Does not encode robot EEF/joint consistency or phantom embodiment suppression. | PatchGuard is not novel if it reduces to adversarial training on patches. | Require clean retention, attacked improvement, and kinematic signal ablations. |
| Random erasing, cutout, visual augmentation | Cheap, parameter-free occlusion and augmentation baselines. | They do not reason about robot proprioception, and may damage clean performance. | If they remove the patch effect, PatchGuard is unnecessary. | Required in STATE 1 as fixed-patch cutout and cheap visual augmentation proxies. |

## Novelty Gate

PatchGuard-VLA is worth running only if the local attack signal survives the simple-baseline gate:

- clean and patched actions diverge under the same robot proprioceptive state,
- a non-leaking EEF/joint/proprio signal exists,
- cutout/random erasing and generic augmentation do not trivially restore the clean action,
- a real future adapter path exists without turning the claim into toy-only evidence.

