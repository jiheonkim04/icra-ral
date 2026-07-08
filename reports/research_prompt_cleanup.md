# Research Prompt Cleanup

Date: 2026-07-08

This cleanup consolidates the previous research attempts, prompt resets, killed routes, and current selection criteria. It is documentation-only. No method implementation, experiment, training, rollout, download, GPU job, OpenVLA-OFT execution, local proxy diagnostic, or NumPy surrogate was run in this pass.

## Current Instruction Boundary

- Do not implement a new method in this run.
- Do not revive stale ActionMap work as an extension.
- Do not revive stale SafeLoRA work except as rejected-direction context.
- Do not make LoRA, QLoRA, or adapters the research novelty.
- Use LoRA or adapters only as later training tools if a real VLA path becomes green.
- Do not continue TCA-Select, weak 7D MLP TCA, CSS-Shield, ExecSpec-Repair, AMP-GD, ResetSpec, Phase-Locked Retiming, TL-ChunkRepair, ContactTube-Aug, PRISM-VLA, ContactSet-VLA, SafeTrace-VLA, or SafeLoRA-VLA as active routes.

## Consolidated User Criteria

The next RA-L-stable direction must satisfy four hard requirements:

1. Novelty first: method-level novelty against the latest relevant VLA/action/grounding/robustness papers.
2. Strong experiments: eventual multi-dataset, multi-task, multi-model, SOTA-axis evidence, with real VLA or official benchmark evaluation.
3. Baseline robustness: must survive mean action, linear/L1, simple MLP, canonicalization-only, safety-only, stop-on-risk, generic DPO/ORPO, diagonal affine, global scale, gripper-only, fixed shift, linear time warp, nearest-demo, single-point, destination-only, object-relative retargeting, and no-repair baselines where relevant.
4. RA-L stability: reject local-proxy-only, toy-only, diagnosis-only, wrapper-only, benchmark-only, and no-real-VLA-path topics.

## Reset Summary

Previous work repeatedly generated useful diagnostics but failed after stronger simple baselines were added. The durable lesson is not "try harder on the same route"; it is to require official anchors, strong trivial baselines, and a direct robotics evidence path before method invention.

The only salvageable research family is the target-prior/action-decoder family, and only if reframed away from TCA-Select and weak 7D heads into:

**Target-Grounded ActionMap / Language-Grounded Action Heatmap for VLA Manipulation**

That direction is not green as an implementation task yet. It must first clear an ActionMap anchor reproduction gate, because the local ActionMap-style mini-anchor already failed mean-action and cheap-MLP baselines.

## Prompt Cleanup Decisions

| Prompt or route family | Cleanup decision | Reason |
| --- | --- | --- |
| Target-Prior TCA-Map | salvage only as Target-Grounded ActionMap | Fixed-prior target evidence was real, but old TCA-Select and weak 7D heads are dead. |
| TCA-Select | do not revive | No measurable contribution beyond corrected target prior. |
| Weak 7D TCA MLP head | do not revive | Failed mean-action gate. |
| CSS-Shield | do not revive | Native-action semantic value collapsed to safety-only. |
| ExecSpec-Repair | do not revive | Diagonal affine matched full repair. |
| AMP-GD | do not revive | Informative/random probe and safety baselines matched or beat it outside toy evidence. |
| ResetSpec | do not revive | Fixed global scale beat object-relative retargeting. |
| Phase-Locked Retiming | do not revive | Fixed shift, gripper-only, linear warp, and related simple baselines explained slices. |
| TL-ChunkRepair | do not revive | Symbolic violation reduction did not improve replay/control utility and lost to no-repair. |
| ContactTube-Aug | do not revive | Augmented actions were not valid enough and lost to object-relative retargeting. |
| PRISM-VLA | do not revive | Canonicalization-only beat PRISM on primary paraphrase robustness and PRIDE. |
| ContactSet-VLA | do not revive | Full contact set lost to active single-point, destination-only, and no-geometry baselines. |
| Local ActionMap mini-anchor | do not extend directly | Failed mean-action, cheap-MLP, and candidate-collapse gates. |
| SafeTrace-VLA | do not revive | Safety-only/risk-only and generic DPO/preference proxy matched the signal. |
| SafeLoRA-VLA | do not revive as topic | No clear official LoRA path; LoRA is a tool, not novelty. |

## Current Clean Recommendation

The project should stop method-first local proxy routes. The best next research direction remains Target-Grounded ActionMap, but the correct immediate decision is:

`NEED_ACTIONMAP_ANCHOR_REPRO_FIRST`

This keeps the target-grounded heatmap idea alive while refusing to build on a failed local heatmap approximation.
