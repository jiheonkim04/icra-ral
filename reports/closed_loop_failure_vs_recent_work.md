# Closed-Loop Failure Vs Recent Work

Date: 2026-07-11 KST

Primary landscape file: `reports/latest_vla_method_landscape_2026.md`.

## Local Failure Mechanisms

The bounded visual review found two different local mechanisms:

- `libero_spatial/task_4`: drawer-contained black bowl extraction fails at `stable_grasp` / `contact_transition`.
- `libero_10/task_4`: two-mug, two-plate sequence fails as `long_horizon_compounding` with incomplete placement/release.

These are not one shared mechanism.

## Closest Recent Work

| Local idea that might be tempting | Closest recent work | Why it does not survive as novelty |
| --- | --- | --- |
| Add a confidence/failure head to detect bad rollout states | VLAConf, https://arxiv.org/abs/2605.29605 | VLAConf already targets calibrated VLA task-success confidence from frozen representations. |
| Verify action candidates before execution | CoVer, https://arxiv.org/html/2602.12281v2; Pre-VLA, https://arxiv.org/abs/2605.22446 | Generic test-time verification, prompt/action selection, and pre-execution action validation are already occupied. |
| Use 3D/spatial verification for the drawer or plate state | VeriSpace, https://arxiv.org/abs/2606.10568 | Spatially grounded action verification is already a direct neighboring claim. |
| Detect visual divergence and replan/truncate the current chunk | VLA-Corrector, https://arxiv.org/abs/2607.01804 | Event-triggered truncation plus corrective replanning is already the direct generic correction route. |
| Adjust action chunk length around contact or placement | AAC, https://arxiv.org/abs/2604.04161; SEAM, https://arxiv.org/abs/2607.04609; Legato, https://arxiv.org/abs/2602.12978 | Adaptive chunking, chunk-boundary smoothing, and continuation training are all crowded. |
| Add progress monitoring or recovery for the two-mug task | SPR, https://arxiv.org/abs/2603.09292; ProgressVLA, https://arxiv.org/abs/2603.27670; ProgVLA, https://arxiv.org/abs/2605.28231 | Progress estimation, progress-guided action correction, spatial subgoals, and rewind are already recent VLA claims. |
| Train on failures as negative examples | AFIL, https://arxiv.org/abs/2605.08434 | Online failure rollouts as adaptive negative guidance already covers this broad route. |
| Use frozen priors, adapters, experts, or routing | PriorVLA, https://arxiv.org/abs/2605.10925; CLARE, https://arxiv.org/abs/2601.09512; VLA-GSE, https://arxiv.org/abs/2605.06175 | Prior-preserving adaptation, adapter expansion/routing, and PEFT expert specialization are not open novelty slots here. |
| Claim robustness on LIBERO perturbations or occlusions | LIBERO-Plus, https://arxiv.org/abs/2510.13626; LIBERO-Occ, https://arxiv.org/abs/2606.10862 | These are benchmark comparisons, not local method novelty. |
| Claim counterfactual language/vision grounding from the mug task | LIBERO-CF / Counterfactual Action Guidance (CAG), https://arxiv.org/abs/2602.17659 | Counterfactual action guidance already addresses language-following shortcuts. |
| Long-horizon reflection/replanning for the two-mug task | REMAC, https://arxiv.org/abs/2503.22122 | Broad reflection/replanning is already represented and is not supported by a single local mechanism. |

## Novelty Outcome

The local evidence does not support a clean mechanism-specific method that avoids these recent works. The best visible issue, drawer-bowl stable grasp, would need a much narrower physical intervention and at least one additional independent reset seed or second task before it could be compared against VLA-Corrector, VeriSpace, AAC, and SPR without looking like a small local variant.

The two-mug failure is even less suitable: progress, subgoal, replanning, and correction claims are already covered by SPR, ProgressVLA, ProgVLA, REMAC, and VLA-Corrector.

## Bottom Line

No surviving RA-L method claim is available from the current review. The literature audit kills generic confidence, verification, adaptive chunking, progress, replanning, failure-negative, and adapter-routing formulations, while the visual review fails to identify a single repeated cross-task mechanism.
