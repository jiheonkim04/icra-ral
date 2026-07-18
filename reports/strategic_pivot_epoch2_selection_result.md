# Strategic Pivot Epoch 2 Selection

Decision: `NO_DEFENSIBLE_PIVOT_FOUND`

Terminal campaign status: `NO_DEFENSIBLE_LOCAL_RESEARCH_PATH_FOUND`

Execution type: `REPORT_ONLY`. This audit did not run a model, optimizer, simulator episode, Ours candidate, physical experiment, or paper-package generation.

## Boundary

- Branch and pushed source HEAD: `codex/epoch5-official-prior-first` at `2bf992a01e23179b01e94ad1060dbce00acd864d`.
- Pivot Epoch 1 is locally closed as `PRIOR_INFRASTRUCTURE_BLOCKED`. Its mechanism-faithful A2C2 module trained validly, but no Base or Prior closed-loop outcome was counted and the paper claim remains scientifically unadjudicated.
- The wrist-dropout axis remains `CLOSED`; RL4IL, RIFA, CVLR, and action-consistent missing-view decisions remain unchanged.
- This is exactly the second and final authorized pivot epoch. It contains exactly two theses, both outside wrist dropout and asynchronous-delay correction.

## Candidate 1 — active-view goal disambiguation

Problem: when a static view is occluded or visually aliased, a VLA may not have enough evidence to identify the task-critical target or affordance. A bounded camera action would gather another view before manipulation.

- Closest prior: [ActiveVLA](https://arxiv.org/abs/2601.08325), with [official artifact page](https://huggingface.co/ZhenyangLiu/ActiveVLA). The official page still marks training/inference/evaluation code and pretrained models as pending.
- Related prior: [Observe Then Act](https://doi.org/10.1109/LRA.2025.3541334), which serializes a camera next-best-view policy and a gripper next-best-pose policy on eight RLBench tasks.
- Protocol: RLBench/COLOSSEUM/GemBench active-camera evaluation with no-probe, random-probe, and heuristic-probe controls.
- Expected Base/Prior: static-view VLA versus multi-view 3D critical-region localization, view selection, and zoom.
- Residual: a lower-cost discrete probe might match the prior with fewer views and no full 3D projection stack.
- Legal inputs: benchmark-authorized live RGB or RGB-D/point clouds, language, proprioception, past camera observations/poses, and authorized camera actions; no task oracle.
- Stage 0: require target/action consistency above no-probe, random-probe, and heuristic-probe on frozen ambiguous identities.
- Stage A: matched active-camera simulator success for Base, Prior, Ours, ablation, and controls.
- Generalization/Pareto: held-out COLOSSEUM or GemBench condition; noninferior success at materially lower view/latency cost.
- Camera-only evidence could measure target consistency under non-actuated viewpoint changes, but cannot establish manipulation success.
- Archive overlap: the old tournament mentioned micro-probes but never executed them; current ActiveVLA and Observe Then Act create direct mechanism overlap. AMP-GD also failed against simple controls locally.
- Decisive-time estimate: `24` active hours.
- Strongest objection: without released ActiveVLA code/checkpoints or a local active-camera Base, there is no official-prior residual to extend; the idea risks becoming a weaker reimplementation.

Scores: `N=3.5, P=2.5, R=3.0, H=4.5, F=1.0, C=2.5, G=4.5, A=5.0, D=5.0`; total `46.0`.

Hard-filter result: **fail**. A locally runnable Base and feasible Ours path are absent, and a decisive result is not credible within 12 active hours.

## Candidate 2 — autonomous retry/recovery after execution failure

Problem: after a missed grasp, drop, misplacement, or state-breaking contact event, a success-only imitation policy may continue an invalid nominal trajectory instead of retrying or resetting.

- Closest prior: [FLARE](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_FLARE_A_Failure-Aware_Framework_for_Autonomous_Correction_and_Recovery_in_CVPR_2026_paper.html), which combines perturbed/bridging Retry demonstrations, object-centric Reset skills, and an online MLLM monitor. Its official CVPR page exposes paper and supplement but no code or checkpoint.
- Related prior: [ReTVL](https://arxiv.org/abs/2606.24633), which learns retry-sensitive values from mixed-quality real-robot demonstrations; no official code/checkpoint was located in the primary release.
- Executable detector only: [SAFE](https://arxiv.org/abs/2506.09937) has [official code](https://github.com/vla-safe/SAFE) at `b6036abe07b2b2bb9996afb2c07f13d6a9f507c0`, but detects failure and raises an alert—it is not an autonomous recovery comparator and releases no trained detector checkpoint.
- Expected Base/Prior: nominal VLA versus FLARE Retry/Reset or ReTVL-weighted imitation.
- Residual: lightweight simulator-only recovery without an online MLLM or reset-skill library.
- Legal inputs: live RGB, instruction, proprioception, legal history, policy features, and detector score; never expert/reset or success oracles at inference.
- Stage 0: induced reversible failures must show both timely detection and a causal recovery advantage over stop-only, restart, random retry, and nearest/mean controls.
- Stage A: matched official-simulator Base, SAFE-stop, faithful recovery Prior, Ours, ablation, and simple-retry episodes.
- Generalization/Pareto: held-out failure type; comparable recovery without an online MLLM, reset library, or second runtime policy.
- Camera-only video can test alert timing, not recovery success.
- Archive overlap: high with CSS-Shield, Phase Retiming, TL-ChunkRepair, SACF, RAC, EAC, verifier/ranking, and retrieval/memory recovery families.
- Decisive-time estimate: `30` active hours.
- Strongest objection: this would be a SAFE-detector plus FLARE/Retry-data composition, with no runnable closest recovery prior and substantial reuse of closed local families.

Scores: `N=2.0, P=3.0, R=4.0, H=4.5, F=1.5, C=2.5, G=4.0, A=5.0, D=1.0`; total `42.25`.

Hard-filter result: **fail**. The faithful prior path depends on unavailable recovery code/checkpoints and real-robot retry/reset data, no official simulator recovery comparator is locally executable, the Ours path is not feasible, closed families would be reused, and the 12-hour boundary cannot be met.

## Adjudication

Neither candidate passes every hard filter. Candidate 1 clears the novelty and residual minima but fails `F >= 4`; Candidate 2 fails both novelty and feasibility minima. Therefore score margin and tie-break rules cannot authorize selection.

Selecting either would replace official-prior-first evidence with a speculative local mechanism or manufacture novelty from a closed family. No Ours candidate, Stage 0, Stage A/B, generalization run, or paper package is authorized.

The frozen outcome is `NO_DEFENSIBLE_LOCAL_RESEARCH_PATH_FOUND`. Autonomous local method development stops here; there is no Pivot Epoch 3.
